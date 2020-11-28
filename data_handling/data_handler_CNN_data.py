import numpy as np
import torch
from torch.autograd import Variable


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, cmip5, soda, godas, device, horizon, window, valid_split=0.1,
                 transfer=True, **kwargs):
        """
        n - length of time series (i.e. dataset size)
        m - number of nodes/grid cells (105 if using exactly the ONI region)
        :param file_name: Omitted if data is not None (e.g. if you use the datareader from ninolearn, as is enso_mtgnn.py)
        :param train: fraction to use for training
        :param valid: fraction to use for validation
        :param device: which device to run on (e.g. "cpu" or "cuda:0", ...)
        :param horizon: self.h - How many timesteps in advance to predict
        :param window: self.P - How many timesteps to use for prediction
        :param normalize: Valid: 0 (data is used as is), 1, 2,..,6, "EEMD" (will run node-wise EEMD, can be slow)
        """
        self.window = window
        self.h = horizon
        self.device = device
        self.T, self.channels, w, self.n_nodes = soda[0].shape  # T=#time series, m=#nodes
        assert w == window, f"Data shape {soda[0].shape} not consistent with argument window={window}"
        self.normalize = -1
        sodaX = np.array(soda[0]) if not isinstance(soda[0], np.ndarray) else soda[0]
        cmip5X = np.array(cmip5[0]) if not isinstance(cmip5[0], np.ndarray) else cmip5[0]
        godasX = np.array(godas[0]) if not isinstance(godas[0], np.ndarray) else godas[0]
        if transfer:
            self.pre_train = torch.tensor(cmip5X).float(), torch.tensor(cmip5[1]).float()

        first_val = int(valid_split * len(soda[0]))
        self.train = [torch.tensor(sodaX[:-first_val]).float(), torch.tensor(soda[1][:-first_val]).float()]
        self.valid = torch.tensor(sodaX[-first_val:]).float(), torch.tensor(soda[1][-first_val:]).float()
        self.test = torch.tensor(godasX).float(), torch.tensor(godas[1]).float()
        self.transfer = transfer
        if not transfer:  # instead of transfer, concat the cmip5 and soda data
            self.merge_transfer_and_train(cmip5X, cmip5[1])

    def __str__(self):
        string = f"Pre-training set of {self.pre_train[0].shape[0]} samples, " if self.transfer else ""
        string += f"Training, Validation, Test samples = {self.T}, {self.valid[0].shape[0]}, {self.test[0].shape[0]}, " \
                  f"#nodes = {self.n_nodes}, #channels = {self.channels}, " \
                  f"Predicting {self.h} time steps in advance using {self.window} time steps --- CNN DATA used"
        return string

    def merge_transfer_and_train(self, transfer_data, transfer_labels):
        transfer_data, transfer_labels = torch.tensor(transfer_data).float(), torch.tensor(transfer_labels).float()
        self.train[0] = torch.cat((transfer_data, self.train[0]), dim=0)
        self.train[1] = torch.cat((transfer_labels, self.train[1]), dim=0)

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


class IndexLoader:
    def __init__(self, args, start_date="1984-01", end_date="2020-08", data_dir="../data", transfer=True,
                 godas_data='GODAS.input.36mn.1980_2015.nc',  # Input of GODAS data set
                 godas_label='GODAS.label.12mn_3mv.1982_2017.nc',  # Label of gods set
                 verbose=False,
                 test_set="GODAS"
                 ):
        from enso.utils import read_ssta, get_index_mask, load_cnn_data, reformat_cnn_data
        self.device = args.device
        self.horizon = args.horizon
        self.window = args.window
        self.n_nodes = args.num_nodes
        try:
            cnn_mask = args.cnn_data_mask
            GODAS_X, GODAS_Y = reformat_cnn_data(lead_months=args.horizon, window=args.window,
                                                 use_heat_content=args.use_heat_content, lon_min=args.lon_min,
                                                 lon_max=args.lon_max, lat_min=args.lat_min, lat_max=args.lat_max,
                                                 data_dir=data_dir + "GODAS/", sample_file=godas_data,
                                                 label_file=godas_label,
                                                 get_valid_nodes_mask=False, get_valid_coordinates=False)
            GODAS = GODAS_X[:, :, :, cnn_mask], GODAS_Y
        except AttributeError:
            _, _, GODAS = load_cnn_data(window=args.window, lead_months=args.horizon, lon_min=args.lon_min,
                                        lon_max=args.lon_max, lat_min=args.lat_min, lat_max=args.lat_max,
                                        data_dir=data_dir, use_heat_content=args.use_heat_content,
                                        return_mask=False)
        if args.use_heat_content or test_set == "GODAS":
            self.dataset = "GODAS"
            if verbose:
                print("Testing on unseen GODAS data...")
            self.test = torch.tensor(np.array(GODAS[0])).float(), torch.tensor(GODAS[1]).float()
            self.semantic_time_steps = GODAS[0].attrs["time"]
        else:
            self.dataset = "ERSSTv5"
            if verbose:
                print("Testing on unseen ERSSTv5 data...")

            flattened_ssta = read_ssta(index=args.index, resolution=args.resolution, stack_lon_lat=True,
                                       start_date=start_date, end_date=end_date,  # end date can be anything for eval.
                                       lon_min=args.lon_min, lon_max=args.lon_max,
                                       lat_min=args.lat_min, lat_max=args.lat_max)
            self.semantic_time_steps = flattened_ssta.get_index("time")[self.window + self.horizon - 1:]

            if transfer:
                flattened_ssta = flattened_ssta[:, cnn_mask]
            _, self.mask = get_index_mask(flattened_ssta, args.index, flattened_too=True, is_data_flattened=True)
            self.test = self._batchify(np.array(flattened_ssta))

    def _batchify(self, data):
        Y_matrix = data[self.window + self.horizon - 1:]  # horizon = #time steps predicted in advance
        timesteps = Y_matrix.shape[0]

        X = torch.zeros((timesteps, 1, self.window, self.n_nodes))
        Y = torch.zeros((timesteps,))

        for start, Y_i in enumerate(Y_matrix):
            end = start + self.window
            X[start, 0, :, :] = torch.from_numpy(data[start:end, :])
            Y[start] = torch.tensor(np.mean(Y_i[self.mask]))
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size
