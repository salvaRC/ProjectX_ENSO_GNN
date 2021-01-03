import time
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import numpy as np
from GNN2.net import gtnet
from GNN2.trainer import Optim
from data_handling.data_handler_CNN_data import data_loading
from utilities.utilities import rmse


def evaluate(dataloader, model, evaluateL2, evaluateL1, device, return_oni_preds=False):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    preds = None
    Ytrue = None

    for i, (X, Y) in enumerate(dataloader):
        assert len(X.size()) == 4, "Expected X to have shape (batch_size, #channels, window, #nodes)"
        X, Y = X.to(device), Y.to(device)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if preds is None:
            preds = output
            Ytrue = Y
        else:
            preds = torch.cat((preds, output))
            Ytrue = torch.cat((Ytrue, Y))
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()

    mmse = total_loss / i

    preds = preds.data.cpu().numpy()
    Ytest = Ytrue.data.cpu().numpy()
    oni_corr = np.corrcoef(Ytest, preds)[0, 1]
    rmse_val = rmse(Ytest, preds)
    r, p = pearsonr(Ytest, preds)
    oni_stats = {"Corrcoef": oni_corr, "RMSE": rmse_val, "Pearson_r": r, "Pearson_p": p}
    if return_oni_preds:
        return total_loss / i, rmse_val, oni_corr, oni_stats, preds, Ytest
    else:
        return total_loss / i, rmse_val, oni_corr, oni_stats


def train(dataloader, model, criterion, optim, args, nth_step=100):
    model.train()
    total_loss = 0
    for iter, (X, Y, finetuning_scaling) in enumerate(dataloader):
        X, Y, finetuning_scaling = X.to(args.device), Y.to(args.device), finetuning_scaling.to(args.device)
        model.zero_grad()
        assert len(X.size()) == 4, "Expected X to have shape (batch_size, #channels, window, #nodes)"
        X = X.transpose(2, 3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.LongTensor(id).to(args.device)
            tx = X[:, :, id, :]
            ty = Y  # [:, id]
            preds = model(tx, id)  # shape = (batch_size x _ x #nodes x _)
            preds = torch.squeeze(preds)
            loss = (((preds - ty)*finetuning_scaling)**2).mean()
            # loss = criterion(preds, ty) # * finetuning_scaling
            loss.backward()

            total_loss += loss.item()
            grad_norm = optim.step()

        if iter % nth_step == 1:
            print('Iter:{:3d} | loss: {:.4f}'.format(iter, loss.item() / iter))
    return total_loss / iter


def train_model(trainloader, valloader, testloader, model, train_data_name, criterion, optim, args, eval_L1, eval_L2, epochs,
                save_every_nth_model=100):
    """

    :param Data: DataLoader object
    :param model: model to train with
    :param train_data_name: either "train" or "pre_train" for the respective purpose
    :param criterion: Loss function
    :param optim: Optimizer
    """
    if train_data_name not in ["train", "pre_train"]:
        raise ValueError("Unknown data attribute, must be train or pre_train!")
    nth_step = 1000 #250 if train_data_name == "train" else 1000
    best_val = 10000000
    print("--" * 15, f'Begin {train_data_name}ing...')
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(trainloader, model, criterion, optim, args, nth_step=nth_step)

        val_loss, val_rmse, val_corr, oni_stats = \
            evaluate(valloader,  model, eval_L2, eval_L1, device=args.device)
        print('--> End of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | Val. stats: loss {:5.4f} | valid RMSE '
              '{:5.4f} | corr {:5.4f} | ONI corr {:5.4f} | ONI RMSE {:5.4f}'.format(epoch,
                                                                                    (time.time() - epoch_start_time),
                                                                                    train_loss, val_loss, val_rmse,
                                                                                    val_corr,
                                                                                    oni_stats["Corrcoef"],
                                                                                    oni_stats["RMSE"]), flush=True)
        # Save the model if the validation loss is the best we've seen so far.
        if oni_stats["RMSE"] < best_val:
            print(f"Model will be saved...")
            with open(args.save, 'wb') as f:
                torch.save(model, f)
                best_val = oni_stats["RMSE"]
        if epoch % save_every_nth_model == 0:
            print(f"Model will be saved at epoch =", epoch)
            with open(args.save.replace(".pt", f"_{epoch}EP.pt"), 'wb') as f:
                torch.save(model, f)

        if epoch % 5 == 0:
            test_acc, test_rae, test_corr, oni_stats = \
                evaluate(testloader, model, eval_L2, eval_L1, device=args.device)
            print("-------> Test stats: rse {:5.4f} | rae {:5.4f} | corr {:5.4f} |"
                  " ONI corr {:5.4f} | ONI RMSE {:5.4f}"
                  .format(test_acc, test_rae, test_corr, oni_stats["Corrcoef"], oni_stats["RMSE"]), flush=True)


def main(args, cmip5, soda, godas, transfer=True, model=None, save_every_nth_model=200):
    device = torch.device(args.device)
    args.device = device
    torch.set_num_threads(3)
    args.num_nodes = soda[0].shape[3]
    assert args.num_nodes == cmip5[0].shape[3] and args.num_nodes == godas[0].shape[3]
    print(f'Inferred {args.num_nodes} nodes from the data.\n')
    trainloader, valloader, testloader = \
        data_loading(cmip5, soda, godas, batch_size=args.batch_size, valid_split=args.validation_frac,
                     concat_cmip5_and_soda=True, scale_finetuning_loss=args.scale_SODA_loss)
    if model is None:
        model = gtnet(args.gcn_true, args.adaptive_edges, args.gcn_depth, args.num_nodes, device, args,
                      predefined_A=None, dropout=args.dropout, subgraph_size=args.subgraph_size,
                      node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                      conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                      skip_channels=args.skip_channels, end_channels=args.end_channels,
                      seq_length=args.window, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
        model = model.to(device)

    print('The receptive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)
    if args.L1Loss:
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)

    evaluateL2 = nn.MSELoss().to(device)
    evaluateL1 = nn.L1Loss().to(device)

    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    args.save = args.save.replace(".pt", f"_{args.epochs}epTRAIN-CONCAT.pt")
    try:
        train_model(trainloader, valloader, testloader, model, "train",
                    criterion, optim, args, evaluateL1, evaluateL2, epochs=args.epochs, save_every_nth_model=save_every_nth_model)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    loop_over = (args.save, "BEST")
    for saved, title in loop_over:
        with open(saved, 'rb') as f:
            model = torch.load(f).to(device)

        val_acc, val_rae, val_corr, val_oni = \
            evaluate(valloader, model, evaluateL2, evaluateL1, device=args.device)
        test_acc, test_rae, test_corr, oni_stats = \
            evaluate(testloader, model, evaluateL2, evaluateL1, device=args.device)
        print(
            f"+++++++++++++++++++++  {title} MODEL STATS (best w.r.t to validation RMSE): +++++++++++++++++++++++++++++++")
        print("-------> Valid stats: rse {:5.4f} | rae {:5.4f} | corr {:5.4f} |"
              " ONI corr {:5.4f} | ONI RMSE {:5.4f}"
              .format(val_acc, val_rae, val_corr, val_oni["Corrcoef"], val_oni["RMSE"]), flush=True)
        print("-------> Test stats: rse {:5.4f} | rae {:5.4f} | corr {:5.4f} |"
              " ONI corr {:5.4f} | ONI RMSE {:5.4f}"
              .format(test_acc, test_rae, test_corr, oni_stats["Corrcoef"], oni_stats["RMSE"]), flush=True)
        print("Saved in", saved)
