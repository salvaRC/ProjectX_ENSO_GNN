import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xa
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from utils import read_ssta, get_index_region_bounds

def transform(x):
    if x > 180:
        x -= 360
    return x


def is_in_oni_region(lat, lon):
    if -5 <= lat <= 5:
        if 190 <= lon <= 240:
            return True
    return False


def heatmap_of_edges(file_path=None, model=None, args=None, index="ONI", resolution=5, min_weight=0.1, reader=None, data_dir=None,
                     save_to=None, plot_index_box=False, plot_heatmap=True, only_towards_oni_region=True):
    """
    The adaptively learnt edges by MTGNN are unidirectional!
    :param file_path: path to saved torch model
    :param reader: a pre-instantiated ninolearn reader
    :param index: the target index (ONI, Nino3.4 or ICEN)
    :param resolution: the climate dataset resolution (in degrees)
    :param plot_fraction:
    :param min_weight: threshold for the adjacency matrix edge weights to be plotted
    :param only_plot_index_region_edges: if True only edges (a, b) will be plotted, where a or b is within index region
    :param plot_index_box: if true plot a box around the index region
    :param save_to:
    :return:
    """
    import cartopy.crs as ccrs
    import torch
    assert file_path is not None or model is not None
    if model is None:
        model = torch.load(file_path, map_location='cpu')
    model.eval()

    if reader is not None:
        ssta = read_ssta(index, data_dir, get_mask=False, stack_lon_lat=True, resolution=resolution, reader=reader)
        coordinates = ssta.indexes["cord"]
        lats, lons = ssta.attrs["Lats"], ssta.attrs["Lons"]
    else:
        from utils import load_cnn_data
        data_dir = data_dir or args.data_dir
        _, _, GODAS, coordinates = load_cnn_data(window=args.window, lead_months=args.horizon, lon_min=args.lon_min,
                                                 lon_max=args.lon_max, lat_min=args.lat_min, lat_max=args.lat_max,
                                                 data_dir=data_dir, use_heat_content=args.use_heat_content,
                                                 return_new_coordinates=True)
        lats, lons = GODAS[0].attrs["Lats"], GODAS[0].attrs["Lons"]

    adj = model.adj_matrix.numpy()
    print("# Nonzero Edges:", np.count_nonzero(adj))

    lat_len, lon_len = len(lats), len(lons)

    incoming_edge_heat = xa.DataArray(np.zeros((lat_len, lon_len)), coords=[("lat", lats), ("lon", lons)])
    outgoing_edge_heat = xa.DataArray(np.zeros((lat_len, lon_len)), coords=[("lat", lats), ("lon", lons)])

    for i, neighbors in enumerate(adj):
        for j, weight in enumerate(neighbors):
            if weight < min_weight:
                continue

            a_lat = coordinates[i][0]
            a_lon = coordinates[i][1]
            b_lat = coordinates[j][0]
            b_lon = coordinates[j][1]

            if is_in_oni_region(a_lat, a_lon):
                incoming_edge_heat.loc[b_lat, b_lon] += weight  # edge a -> b, where a is in ONI region
            if is_in_oni_region(b_lat, b_lon):
                outgoing_edge_heat.loc[a_lat, a_lon] += weight  # edge a -> b, where b is in ONI region


    fig = plt.figure()
    gs = fig.add_gridspec(2, 1)
    cm = 180
    ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree(central_longitude=cm))

    minlon = -180 + cm
    maxlon = +179 + cm
    ax1.set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())

    ax1.set_title("Heatmap of summed edge weights that point towards ONI region")
    if only_towards_oni_region:
        loop_over = zip([ax1], [outgoing_edge_heat])
    else:
        ax2 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree(central_longitude=cm))
        ax2.set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())
        ax2.set_title("Heatmap of summed edge weights that point out of the ONI region")
        loop_over = zip([ax1, ax2], [outgoing_edge_heat, incoming_edge_heat])

    for ax, heat in loop_over:
        if plot_heatmap:
            cb = ax.pcolormesh(lons, lats, heat, cmap="Reds", transform=ccrs.PlateCarree())
        else:
            cb = ax.contourf(lons, lats, heat, transform=ccrs.PlateCarree(), alpha=0.85, cmap="Reds", levels=100)

        '''        map = Basemap(projection='cyl', llcrnrlat=-55, urcrnrlat=60, resolution='c', llcrnrlon=0, urcrnrlon=380, ax=ax)
        map.drawcoastlines(linewidth=0.2)
        map.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0], fontsize=6.5, color='grey', linewidth=0.2)
        map.drawmeridians(np.arange(0., 380., 60.), labels=[0, 0, 0, 1], fontsize=6.5, color='grey', linewidth=0.2)
        '''
        fig.colorbar(cb, ax=ax)

    ax1.coastlines()
    if not only_towards_oni_region:
        ax2.coastlines()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()


def create_filepath(prefix="", suffix="", **kwargs):
    if prefix is None:
        return None
    string = prefix
    for key, name in kwargs.items():
        string += f"_{name}{key}"
    return string + suffix


def plot_time_series(data, *args, labels=["timeseries"], time_steps=None, data_std=None, linewidth=2,
                     timeaxis="time", ylabel="Nino3.4 index", plot_months=False, show=True, save_to=None):
    if time_steps is not None:
        time = time_steps
    elif isinstance(data, xa.DataArray):
        time = data.get_index(timeaxis)
    else:
        time = np.arange(0, data.shape[0], 1)
    series = np.array(data)
    plt.figure()
    plt.plot(time, series, label=labels[0], linewidth=linewidth)
    if data_std is not None:
        plt.fill_between(time, series - data_std, series + data_std, alpha=0.25)
    minimum, maximum = np.min(data), np.max(data)
    for i, arg in enumerate(args, 1):
        minimum, maximum = min(minimum, np.min(arg)), max(maximum, np.max(np.max(arg)))
        try:
            plt.plot(time, arg, label=labels[i], linewidth=linewidth)
        except ValueError as e:
            raise ValueError("Please align the timeseries to the same time axis.", e)
        except IndexError:
            raise IndexError("You must pass as many entries in labels, as there are time series to plot")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.yticks(np.arange(np.round(minimum - 0.5, 0), np.round(maximum + 0.51, 0), 0.5))
    nth_month = 10
    if plot_months and isinstance(time[0], pd._libs.tslibs.Timestamp):
        xticks, year_mon = time[::nth_month][:-1], [f"{date.year}-{date.month}" for date in time[::nth_month][:-1]]
        xticks = xticks.append(pd.Index([time[-1]]))
        year_mon.append(f"{time[-1].year}-{time[-1].month}")  # add last month
        plt.xticks(ticks=xticks, labels=year_mon, rotation=20)
    plt.legend()
    plt.grid()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')

    if show:
        plt.show()
    return time
