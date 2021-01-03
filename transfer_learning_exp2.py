from hyperparams_and_args import get_argparser
from utilities.utilities import load_cnn_data, get_filename, season_to_int
from GNN2.train_single_step_CNN_Data import main
import torch
import warnings
warnings.filterwarnings("ignore")

save_every_nth_model = 50

if __name__ == "__main__":
    parser = get_argparser(experiment="cnn_data")
    args = parser.parse_args()
    reload_model_from = args.reload_pretrained_from

    assert args.resolution == 5, "5deg resolution is a must for CNN data."
    torch.cuda.empty_cache()
    if reload_model_from is None:
        fine_tune = False
        model = None
        args.save = get_filename(args)
    else:
        fine_tune = ("FINETUNED" in reload_model_from) or (not args.transfer_learning)

        print(f"CONTINUE TRAINING MODEL RELOADED FROM", reload_model_from,
              ", args besides epochs, finetune_with_cmip5_too & lr will be overwritten by the model's one...")
        model = torch.load(reload_model_from)
        model.args.finetune_with_cmip5_too = args.finetune_with_cmip5_too
        model.args.lr = args.lr
        model.args.validation_frac = args.validation_frac
        if fine_tune:
            model.args.epochs += args.epochs
            transfer = False
            model.args.save = model.args.save.replace(".pt", f"_{args.epochs}FINETUNED.pt")
        else:
            model.args.transfer_epochs += args.transfer_epochs
            model.args.save = model.args.save.replace(".pt", f"_{args.transfer_epochs}PRETRAINED.pt")
        args = model.args

    print(args, "\n")

    cmip5, SODA, GODAS, cords, cnn_mask \
        = load_cnn_data(window=args.window, lead_months=args.horizon, lon_min=args.lon_min,
                        lon_max=args.lon_max, lat_min=args.lat_min, lat_max=args.lat_max,
                        data_dir=args.data_dir, use_heat_content=args.use_heat_content,
                        return_new_coordinates=True, return_mask=True, target_months=season_to_int(args.target_month),
                        )
    args.cnn_data_mask = cnn_mask
    args.geo_coords = cords
    print(f"Shapes are (#timesteps, #features, window, #nodes), CMIP5: {cmip5[0].shape}, SODA: {SODA[0].shape}")

    if args.adaptive_edges:
        print("Using an adaptive adjacency matrix.")

    main(args, cmip5, SODA, GODAS, transfer=not fine_tune, model=model, save_every_nth_model=save_every_nth_model)
