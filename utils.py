import numpy as np
import xarray as xa
from ninolearn.IO.read_processed import data_reader
from sklearn.metrics import mean_squared_error

def rmse(y, preds):
    """
    The root-mean-squarred error (RMSE) for a given observation and prediction.

    :type y: array_like
    :param y: The true observation.

    :type pred: array_like
    :param pred: The prediction.

    :rtype: float
    :return: The RMSE value
    """
    return np.sqrt(mean_squared_error(y, preds))

def cord_mask(data: xa.DataArray, is_flattened=False, flattened_too=False, lat=(-5, 5), lon=(190, 240)):
    """
    :param data:
    :param dim:
    :return:
    """
    oni_mask = {'time': slice(None), 'lat': slice(lat[0], lat[1]), 'lon': slice(lon[0], lon[1])}
    if flattened_too:
        flattened = data.copy() if is_flattened else data.stack(cord=['lat', 'lon']).copy()
        flattened[:, :] = 0
        flattened.loc[oni_mask] = 1  # Masked (ONI) region has 1 as value
        flattened_mask = (flattened[0, :] == 1)
        # print(np.count_nonzero(flattened_mask), '<<<<<<<<<<<<<<<<<')
        # flattened.sel(oni_mask) == flattened.loc[:, flattened_mask]
        return oni_mask, flattened_mask
    return oni_mask


def get_index_mask(data, index, flattened_too=False, is_data_flattened=False):
    """
    Get a mask to mask out the region used for  the ONI/El Nino3.4 or ICEN index.
    :param data:
    :param index: ONI or Nino3.4 or ICEN
    :return:
    """
    lats, lons = get_index_region_bounds(index)
    return cord_mask(data, lat=lats, lon=lons, flattened_too=flattened_too, is_flattened=is_data_flattened)


def get_index_region_bounds(index):
    if index.lower() in ["nino3.4", "oni"]:
        return (-5, 5), (190, 240)  # 170W-120W
    elif index.lower() == "icen":
        return (-10, 0), (270, 280)  # 90W-80W
    else:
        raise ValueError("Unknown index")


def oni_correlation_skill(true, preds, axis=1):
    """
    If axis=0:
        Both true and preds should be N x T ndarrays, where
            - N is the number of nodes (in the ONI region)
            - T is the number of predictions made (for T different time steps)
    Conversely, if axis=1, T x N arrays.
    :param true: True SST anomalies over the ONI region
    :param preds: Predicted SST anomalies over the ONI region
    :return:
    """
    oni_Y = true.mean(axis=axis)  # mean over the ONI region
    oni_pred = preds.mean(axis=axis)
    oni_corr = np.corrcoef(oni_Y, oni_pred)[0, 1]
    return oni_corr


def get_adj(data, radius_lat=3, radius_lon=3, no_self_loop=True):
    n_nodes = data.shape[1]
    data0 = data[0, :]  # dont care about time axis here
    adj = np.zeros((data0.shape[0], data0.shape[0]))  # N x N
    tmp = xa.DataArray(adj, dims=("x1", "cord"), coords={"x1": range(n_nodes), "cord": data0.indexes['cord']})
    print(tmp.coords, "\n")
    for i in range(n_nodes):
        node = data.indexes["cord"][i]
        lat, lon = node[0], node[1]
        neighbors = {'lat': slice(lat - radius_lat, lat + radius_lat),
                     'lon': slice(lon - radius_lon, lon + radius_lon)}
        tmp.loc[i, neighbors] = 1

    assert np.count_nonzero(tmp.values != tmp.values.T) == 0  # symmetric adjacency matrix...
    matrix = tmp.values
    if no_self_loop:
        for i in range(n_nodes):
            matrix[i, i] = 0
    return matrix


def check_chosen_coordinates(index, lon_min=190, lon_max=240, lat_min=-5, lat_max=5, ):
    if index in ["Nino3.4", "ONI"]:
        assert lat_min <= -5 and lat_max >= 5
        assert lon_min <= 190 and lon_max >= 240  # 170W-120W
    elif index == "ICEN":
        assert lat_min <= -10 and lat_max >= 0
        assert lon_min <= 270 and lon_max >= 280  # 90W-80W
    elif index[-3:] == "mon":
        pass
    else:
        raise ValueError("Unknown index")


def read_ssta(index, data_dir, get_mask=False, stack_lon_lat=True, resolution=2.5, dataset="ERSSTv5", fill_nan=0,
              start_date='1871-01', end_date='2019-12',
              lon_min=190, lon_max=240,
              lat_min=-5, lat_max=5,
              reader=None):
    """

    :param index: choose target index (e.g. ONI, Nino3.4, ICEN)
    :param start_date:
    :param end_date:
    :param lon_min:
    :param lon_max:
    :param lat_min:
    :param lat_max:
    :param reader: If a data_reader is passed, {start,end}_date and {lat, lon}_{min, max} will be ignored.
    :return:
    """
    if index in ["Nino3.4", "ONI"]:
        k = 5 if index == "Nino3.4" else 3
    elif index == "ICEN":
        k = 3
    elif index[-3:] == "mon":
        k = int(index[-4])  # eg 1mon
    else:
        raise ValueError("Unknown index")

    if reader is None:
        reader = data_reader(data_dir=data_dir,
                             startdate=start_date, enddate=end_date,
                             lon_min=lon_min, lon_max=lon_max,
                             lat_min=lat_min, lat_max=lat_max)
        check_chosen_coordinates(index, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)

    resolution_suffix = f"{resolution}x{resolution}"
    ssta = reader.read_netcdf('sst', dataset=dataset, processed='anom', suffix=resolution_suffix)
    ssta = ssta.rolling(time=k).mean()[k - 1:]  # single months SSTAs --> rolling mean over k months SSTAs

    if stack_lon_lat:
        lats, lons = ssta.get_index('lat'), ssta.get_index('lon')
        ssta = ssta.stack(cord=['lat', 'lon'])
        ssta.attrs["Lons"] = lons
        ssta.attrs["Lats"] = lats
    if fill_nan is not None:
        if fill_nan == "trim":
            ssta_old_index = ssta.get_index('cord')
            ssta = ssta.dropna(dim='cord')
            print(f"Dropped {len(ssta_old_index) - len(ssta.get_index('cord'))} nodes.")
            # print("Dropped coordinates:", set(ssta_old_index).difference(set(ssta.get_index("cord"))))
            # print(flattened_ssta.loc["1970-01", (0, 290)]) --> will raise error
        else:
            ssta = ssta.fillna(fill_nan)

    if get_mask:
        index_mask, train_mask = get_index_mask(ssta, index=index, flattened_too=True, is_data_flattened=stack_lon_lat)
        train_mask = np.array(train_mask)
        return ssta, train_mask
    return ssta


def reformat_cnn_data(lead_months=3, window=3, use_heat_content=False,
                      lon_min=0, lon_max=360,
                      lat_min=-55, lat_max=60,
                      data_dir="data/",
                      sample_file='CMIP5.input.36mn.1861_2001.nc',  # Input of training set
                      label_file='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
                      sst_key="sst",
                      get_valid_nodes_mask=False,
                      get_valid_coordinates=False,
                      target_months="all"
                      ):
    """
    :param lon_min, lon_max, lat_min, lat_max: all inclusive
    :param target_months, if 'all' concatenate as timeseries, else take only the target month in [1, 12]
    """
    import pandas as pd
    lat_p1, lat_p2 = int((lat_min + 55) / 5), min(int((lat_max + 55) / 5), 23)
    lon_p1, lon_p2 = int(lon_min / 5), min(int(lon_max / 5), 71)
    data = xa.open_dataset(f'{data_dir}/{sample_file}')
    labels = xa.open_dataset(f'{data_dir}/{label_file}')
    # Shape T' x 36 x |lat| x |lon|, want : T x 12 x |lat| x |lon|
    lat_sz = lat_p2 - lat_p1 + 1
    lon_sz = lon_p2 - lon_p1 + 1
    features = 2 if use_heat_content else 1
    feature_names = ["sst", "heat_content"] if use_heat_content else ["sst"]

    filtered_region = data.sel(
        {'lat': slice(lat_min, lat_max), 'lon': slice(lon_min, lon_max)}
    )

    if target_months != "all":
        filtered_region = filtered_region.rename({"lev": "window"})  # rename coordinate name
        assert 1 <= target_months <= 12, f"Must be in [1 12], but got {target_months}"
        X_all_target_mons = np.empty((data.sizes["time"], features, window, lat_sz, lon_sz))
        Y_all_target_mons = np.empty((data.sizes["time"]))
        target_month = target_months - 1
        X_all_target_mons = xa.DataArray(X_all_target_mons, coords=[("time", data.get_index("time")),
                                                                    ("feature", feature_names),
                                                                    ("window", np.arange(1, window + 1)),
                                                                    ("lat", filtered_region.get_index("lat")),
                                                                    ("lon", filtered_region.get_index("lon"))
                                                                    ])
    else:  # if "all"
        filtered_region = filtered_region.rename({"lev": "window", "time": "year"})  # rename coordinate name
        X_all_target_mons = np.empty((data.sizes["time"], 12, features, window, lat_sz, lon_sz))
        Y_all_target_mons = np.empty((data.sizes["time"], 12))
        tg_mons = np.arange(0, 12)
        X_all_target_mons = xa.DataArray(X_all_target_mons, coords=[("year", data.get_index("time")),
                                                                    ("tg-mon", tg_mons),
                                                                    ("feature", feature_names),
                                                                    ("window", np.arange(1, window + 1)),
                                                                    ("lat", filtered_region.get_index("lat")),
                                                                    ("lon", filtered_region.get_index("lon"))
                                                                    ])
    if "CMIP5" not in label_file:
        X_all_target_mons.attrs["time"] = \
            [pd.Timestamp("1982-01-01") + pd.DateOffset(months=d_mon) for d_mon in range(len(data.get_index("time"))*12)]

    X_all_target_mons.attrs["Lons"] = filtered_region.get_index('lon')
    X_all_target_mons.attrs["Lats"] = filtered_region.get_index('lat')
    if target_months != "all":
        var_dict = {"ld_mn2": int(25 - lead_months + target_month) + 1,
                    "ld_mn1": int(25 - lead_months + target_month) + 1 - window}

        X_all_target_mons.loc[:, "sst", :, :, :] = \
            filtered_region.variables[sst_key][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]

        if use_heat_content:
            X_all_target_mons.loc[:, "heat_content", :, :, :] = \
                filtered_region.variables['t300'][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]

        Y_all_target_mons[:] = labels.variables['pr'][:, target_month, 0, 0]
        X_time_and_node_flattened = X_all_target_mons.stack(cord=["lat", "lon"])

    else:
        for target_month in range(0, 12):
            '''
            target months are indices [25, 36)
            possible predictor months (for lead months<=24) are indices [0, 24]
            '''
            var_dict = {"ld_mn2": int(25 - lead_months + target_month) + 1,
                        "ld_mn1": int(25 - lead_months + target_month) + 1 - window}

            X_all_target_mons.loc[:, target_month, "sst", :, :, :] = \
                filtered_region.variables[sst_key][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]

            if use_heat_content:
                X_all_target_mons.loc[:, target_month, "heat_content", :, :, :] = \
                    filtered_region.variables['t300'][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]

            Y_all_target_mons[:, target_month] = labels.variables['pr'][:, target_month, 0, 0]
        X_time_and_node_flattened = X_all_target_mons.stack(time=["year", "tg-mon"], cord=["lat", "lon"])
    X_time_and_node_flattened = X_time_and_node_flattened.transpose("time", "feature", "window", "cord")
    Y_time_flattened = Y_all_target_mons.reshape((-1,))

    if get_valid_nodes_mask:
        sea = (np.count_nonzero(X_time_and_node_flattened[:, 0, 0, :], axis=0) > 0)
        if get_valid_coordinates:
            return X_time_and_node_flattened, Y_time_flattened, sea, X_time_and_node_flattened.get_index("cord")
        return X_time_and_node_flattened, Y_time_flattened, sea

    return X_time_and_node_flattened, Y_time_flattened


def load_cnn_data(lead_months=3, window=3, use_heat_content=False,
                  lon_min=0, lon_max=359,
                  lat_min=-55, lat_max=60,
                  data_dir="data/",
                  cmip5_data='CMIP5.input.36mn.1861_2001.nc',  # Input of CMIP5 training set
                  cmip5_label='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
                  soda_data='SODA.input.36mn.1871_1970.nc',  # Input of SODA training set
                  soda_label='SODA.label.nino34.12mn_3mv.1873_1972.nc',  # Label of training set
                  godas_data='GODAS.input.36mn.1980_2015.nc',  # Input of GODAS training set
                  godas_label='GODAS.label.12mn_3mv.1982_2017.nc',  # Label of training set
                  truncate_GODAS=True,  # whether to truncate it to the 1984-2017 period the CNN paper used
                  return_new_coordinates=False,
                  return_mask=False,
                  target_months="all"
                  ):
    """

    :param lead_months:
    :param window:
    :param use_heat_content:
    :param lon_min:
    :param lon_max:
    :param lat_min:
    :param lat_max:
    :param data_dir:
    :param cmip5_data:
    :param cmip5_label:
    :param soda_data:
    :param soda_label:
    :param godas_data:
    :param godas_label:
    :param truncate_GODAS:
    :param return_new_coordinates:
    :param return_mask:
    :param target_months: if "all", the model will need to learn to give predictions for any target months,
                            if an int in [1, 12], it can focus on that specific target month/season,
                            where 1 translates to "JFM", ..., 12 to "DJF"
    :return:
    """
    cmip5, cmip5_Y, m1, cords = reformat_cnn_data(lead_months=lead_months, window=window,
                                                  use_heat_content=use_heat_content,
                                                  lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                                  data_dir=data_dir + "CMIP5_CNN/", sst_key="sst1",
                                                  sample_file=cmip5_data, label_file=cmip5_label,
                                                  get_valid_nodes_mask=True, get_valid_coordinates=True, target_months=target_months)
    SODA, SODA_Y, m2 = reformat_cnn_data(lead_months=lead_months, window=window, use_heat_content=use_heat_content,
                                         lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                         data_dir=data_dir + "SODA/", sample_file=soda_data, label_file=soda_label,
                                         get_valid_nodes_mask=True, get_valid_coordinates=False, target_months=target_months)
    GODAS, GODAS_Y, m3 = reformat_cnn_data(lead_months=lead_months, window=window, use_heat_content=use_heat_content,
                                           lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                           data_dir=data_dir + "GODAS/", sample_file=godas_data, label_file=godas_label,
                                           get_valid_nodes_mask=True, get_valid_coordinates=False, target_months=target_months)
    if truncate_GODAS:
        start_1984 = 24 if target_months == "all" else 2
        GODAS, GODAS_Y, GODAS.attrs["time"] = GODAS[start_1984:], GODAS_Y[start_1984:], GODAS.attrs["time"][start_1984:]
    # DUE to variations due to resolution = 5deg., there are some inconsistencies in which nodes are terrestrial
    final_mask = np.logical_and(m1, np.logical_and(m2, m3))
    cmip5, SODA, GODAS = cmip5[:, :, :, final_mask], SODA[:, :, :, final_mask], GODAS[:, :, :, final_mask]
    if return_new_coordinates:
        cords = cords[final_mask]
        if return_mask:
            return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), cords, final_mask
        return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), cords
    if return_mask:
        return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), final_mask
    return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y)


def get_cnn_data_oceanic_nodes_mask(
        lon_min=0, lon_max=359,
        lat_min=-55, lat_max=60,
        data_dir="data/",
        cmip5_data='CMIP5.input.36mn.1861_2001.nc',  # Input of CMIP5 training set
        cmip5_label='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
        soda_data='SODA.input.36mn.1871_1970.nc',  # Input of SODA training set
        soda_label='SODA.label.nino34.12mn_3mv.1873_1972.nc',  # Label of training set
        godas_data='GODAS.input.36mn.1980_2015.nc',  # Input of GODAS training set
        godas_label='GODAS.label.12mn_3mv.1982_2017.nc',  # Label of training set
):
    _, _, _, mask = load_cnn_data(
        lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
        data_dir=data_dir, cmip5_data=cmip5_data, cmip5_label=cmip5_label,
        soda_data=soda_data, soda_label=soda_label, godas_data=godas_data,
        godas_label=godas_label, return_mask=True
    )
    return mask


def read_index(index, dataset="ERSSTv5", start_date='1871-01', end_date='2019-12', resolution=2.5):
    ssta = read_ssta(index=index, dataset=dataset, start_date=start_date, end_date=end_date, resolution=resolution)
    index = np.mean(ssta, axis=1)
    return index


def east_to_west(longitude, to="East"):
    if to in ["E", "East", "east"]:
        return 360 - longitude
    if to in ["W", "West", "west"]:
        return longitude + 360


def get_filename(args, transfer=False):
    args.save += args.target_month
    args.save += f"{args.horizon}lead_{args.index}" \
                 f"_{args.lat_min}-{args.lat_max}lats" \
                 f"_{args.lon_min}-{args.lon_max}lons" \
                 f"_{args.window}w{args.layers}L{args.gcn_depth}gcnDepth{args.dilation_exponential}dil" \
                 f"_{args.batch_size}bs{args.dropout}d{args.normalize}normed"
    args.save += "_prelu" if args.prelu else ""
    args.save += "_withHC" if args.use_heat_content else ""
    args.save += "_CNN_DATA_TRANSFER.pt" if transfer else ".pt"
    return args.save
