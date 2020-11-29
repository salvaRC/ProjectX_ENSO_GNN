import numpy as np
import torch
from torch.autograd import Variable
from utils import read_ssta, get_index_mask


class DataLoaderS(object):
    def __init__(self, args,
                 train_dates=("1871-01", "1972-12"), val_dates=("1973-01", "1983-12"),
                 test_dates=("1984-01", "2020-09")):
        """
        n - length of time series (i.e. dataset size)
        m - number of nodes/grid cells (105 if using exactly the ONI region)

        """
        self.window = args.window
        self.horizon = args.horizon
        self.device = args.device

        train = read_ssta(index=args.index, data_dir=args.data_dir, resolution=args.resolution, stack_lon_lat=True,
                          start_date=train_dates[0], end_date=train_dates[1],
                          lon_min=args.lon_min, lon_max=args.lon_max,
                          lat_min=args.lat_min, lat_max=args.lat_max)
        self.T, self.n_nodes = train.shape  # n=#time series, m=#nodes
        _, self.mask = get_index_mask(train, args.index, flattened_too=True, is_data_flattened=True)

        val = read_ssta(index=args.index, data_dir=args.data_dir, resolution=args.resolution, stack_lon_lat=True,
                        start_date=val_dates[0], end_date=val_dates[1],
                        lon_min=args.lon_min, lon_max=args.lon_max,
                        lat_min=args.lat_min, lat_max=args.lat_max)
        test = read_ssta(index=args.index, data_dir=args.data_dir, resolution=args.resolution, stack_lon_lat=True,
                         start_date=test_dates[0], end_date=test_dates[1],
                         lon_min=args.lon_min, lon_max=args.lon_max,
                         lat_min=args.lat_min, lat_max=args.lat_max)
        self.semantic_time_steps = (
            train.get_index("time")[self.window + self.horizon - 1:],
            val.get_index("time")[self.window + self.horizon - 1:],
            test.get_index("time")[self.window + self.horizon - 1:]

        )

        self.train = self._batchify(np.array(train))
        self.valid = self._batchify(np.array(val))
        self.test = self._batchify(np.array(test))

    def __str__(self):
        return f"Time series Length = {self.T}, Number of nodes = {self.n_nodes}," \
               f" Predict {self.horizon} time steps in advance using {self.window} time steps," \
               f" Training set size = {self.train[1].shape[0]} "

    def _batchify(self, data):
        Y_matrix = data[self.window + self.horizon - 1:, :]  # horizon = #time steps predicted in advance

        X = torch.zeros((self.T, self.window, self.n_nodes))
        Y = torch.zeros((self.T, self.n_nodes))

        for start, Y_i in enumerate(Y_matrix):
            end = start + self.window
            X[start, :, :] = torch.from_numpy(data[start:end, :])
            Y[start, :] = torch.tensor(Y_i)
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


class GODASLoader:
    def __init__(self, args, start_date="1984-01", end_date="2020-08", data_dir="../data", transfer=True,
                 godas_data='GODAS.input.36mn.1980_2015.nc',  # Input of GODAS data set
                 godas_label='GODAS.label.12mn_3mv.1982_2017.nc',  # Label of gods set
                 verbose=False,
                 test_set="GODAS",
                 as_index=True
                 ):
        from utils import read_ssta, get_index_mask, load_cnn_data, reformat_cnn_data
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
            transfer = False
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
            if as_index:
                self.test = self._batchify_index(np.array(flattened_ssta))
            else:
                self.test = self._batchify(np.array(flattened_ssta))

    def _batchify_index(self, data):
        Y_matrix = data[self.window + self.horizon - 1:]  # horizon = #time steps predicted in advance
        timesteps = Y_matrix.shape[0]

        X = torch.zeros((timesteps, 1, self.window, self.n_nodes))
        Y = torch.zeros((timesteps,))

        for start, Y_i in enumerate(Y_matrix):
            end = start + self.window
            X[start, 0, :, :] = torch.from_numpy(data[start:end, :])
            Y[start] = torch.tensor(np.mean(Y_i[self.mask]))
        return [X, Y]


    def _batchify(self, data):
        Y_matrix = data[self.window + self.horizon - 1:, :]  # horizon = #time steps predicted in advance
        timesteps = Y_matrix.shape[0]

        X = torch.zeros((timesteps, self.window, self.n_nodes))
        Y = torch.zeros((timesteps, self.n_nodes))

        for start, Y_i in enumerate(Y_matrix):
            end = start + self.window
            X[start, :, :] = torch.from_numpy(data[start:end, :])
            Y[start, :] = torch.tensor(Y_i)

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


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


if __name__ == "__main__":
    from utils import read_ssta, get_index_mask

    horizon = 3
    resolution = 5
    data_path = "../data/mtgnn/"
    start = '1871-01'
    end_date = '2019-12'
    lon_min, lon_max = 190, 240
    lat_min, lat_max = -5, 5
    flattened_ssta, _ = read_ssta(index="ONI", get_mask=True, resolution=resolution,
                                  start_date=start, end_date=end_date,
                                  lon_min=lon_min, lon_max=lon_max,
                                  lat_min=lat_min, lat_max=lat_max)
