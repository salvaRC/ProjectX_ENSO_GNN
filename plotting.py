import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xa
import os
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from enso.utils import get_index_mask, read_ssta, read_index, get_index_region_bounds
from ninolearn.IO.read_processed import data_reader


def plot_grid(data_grid, x=1):
    plt.figure()
    # plt.xlim([40, 80])
    plt.scatter([p[x] for p in data_grid], [p[(x + 1) % 2] for p in data_grid])
    plt.show()


def heatmap(data, t=None, year=None, seaborn=True, lat_mask=None, lon_mask=None, scale_mask_by=2):
    plt.figure()
    if year is None:
        if t is None:
            t = np.random.randint(0, data.shape[0])
        year = int(np.floor(1854 + t / 12))
        month = t % 12 + 1
        plt.title(f"Heatmap for {month}d month in {year}")
        plt.imshow(data[t, :, :], cmap="hot", interpolation='nearest')
    else:
        plt.title(f"Heatmap for {year}")
        year = pd.to_datetime(year)
        data2 = data.loc[year].copy()
        if lat_mask is not None and lon_mask is not None:
            data2.loc[lat_mask[0]:lat_mask[1], lon_mask[0]:lon_mask[1]] \
                = scale_mask_by * data2.loc[lat_mask[0]:lat_mask[1], lon_mask[0]:lon_mask[1]]

        nd = np.array(data2).reshape(data.shape[1:3])
        if seaborn:
            sns.heatmap(nd)
        else:
            # plt.xticks(np.arange(data.coords["lat"][0], data.coords["lat"][1], 10))
            # plt.yticks(data.coords["lon"][::5])
            plt.imshow(data.loc[year, :, :], cmap="hot", interpolation='nearest')
    plt.show()


def prediction_heatmap(preds, surrounding_data, args, seaborn=True, t=None, extent=None):
    """

    :param preds: model's anomaly predictions
    :param surrounding_data: data in which to embed the predictions, e.g. the globe
    :param args:
    :param seaborn:
    :param t:
    :param extent: a list/array of 4 values ~ [lon_min, lon_max, lat_min, lat_max]
    :return:
    """
    if t is None:
        t = -1  # take last prediction
    plot_data = surrounding_data[:, :, :].copy()
    lat_min, lat_max = args.lat_min, args.lat_max
    lon_min, lon_max = args.lon_min, args.lon_max
    to_overwrite = plot_data.loc[:, lat_min:lat_max, lon_min:lon_max]
    mask = get_index_mask(to_overwrite, index=args.index, flattened_too=False)
    to_overwrite = to_overwrite.loc[mask][t, :, :]
    heat_min, heat_max = np.min(preds[t, :]), np.max(preds[t, :])

    plot_data = plot_data.where(plot_data == 0, -10)  # set everything besides the continents to same color
    plot_data.loc[mask][t, :, :] = preds[t, :].reshape(to_overwrite.shape)
    plot_data = plot_data[t, :, :][::-1]  # ::-1, otherwise the world will be upside down in the plot

    plt.figure()
    # x_axis = plot_data.get_index("lon") # y_axis = plot_data.get_index("lat")
    if seaborn:
        plot = sns.heatmap(plot_data, vmin=heat_min, vmax=heat_max)  # , xticklabels=x_axis, yticklabels=y_axis)
        handle_xy_axes_seaborn(plot, extent)
    else:
        plt.imshow(plot_data, cmap="hot", interpolation='nearest', extent=extent, vmin=heat_min, vmax=heat_max)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def handle_xy_axes_seaborn(plot_ax, extent):
    x_axis = np.arange(extent[0], extent[1], (extent[1] - extent[0]) / len(plot_ax.get_xticklabels()))
    x_axis = [int(x) for x in x_axis]
    plot_ax.set_xticklabels(x_axis)

    y_axis = np.arange(extent[2], extent[3], (extent[3] - extent[2]) / len(plot_ax.get_yticklabels()))
    y_axis = [int(y) for y in y_axis][::-1]
    plot_ax.set_yticklabels(y_axis)


def transform(x):
    if x > 180:
        x -= 360
    return x


def is_in_oni_region(lat, lon):
    if -5 <= lat <= 5:
        if 190 <= lon <= 240:
            return True
    return False


def plot_learned_gnn_edgesOLD(file_path):
    import torch
    import conda
    conda_file_dir = conda.__file__
    conda_dir = conda_file_dir.split('lib')[0]
    proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
    os.environ["PROJ_LIB"] = proj_lib
    from mpl_toolkits.basemap import Basemap

    model = torch.load(file_path, map_location='cpu')
    model.eval()

    reader = data_reader(startdate='1871-01', enddate='2019-12',
                         lon_min=50, lon_max=300,
                         lat_min=-40, lat_max=40)
    ssta = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
    cords = ssta.stack(cord=['lat', 'lon']).indexes["cord"]
    adj = model.adj_matrix.numpy()

    plt.figure()
    m = Basemap(projection='mill', lat_ts=2.5,
                llcrnrlon=0, llcrnrlat=-30,
                urcrnrlon=360, urcrnrlat=30)  # for pacific in center, but bugs for plotting?
    m = Basemap(llcrnrlon=-180, llcrnrlat=-80,
                urcrnrlon=180, urcrnrlat=80)
    m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
    m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
    m.drawcoastlines(linewidth=0.1, color="white")

    # Add connections between edges:
    min = 0.99

    for i, neighbors in enumerate(adj):
        for j, weight in enumerate(neighbors[:i]):
            if weight < min:
                continue

            a_lat = cords[i][0]
            a_lon = cords[i][1]
            b_lat = cords[j][0]
            b_lon = cords[j][1]
            if not is_in_oni_region(a_lat, a_lon) and not is_in_oni_region(b_lat, b_lon):
                continue
            a_lon = transform(a_lon)
            b_lon = transform(b_lon)
            if np.random.randint(0, 100) < 95 or b_lon == 180 or a_lon == 180:
                continue
            # print(a_lon, a_lat, b_lon, b_lat)
            x1, y1 = m(a_lon, a_lat)
            x2, y2 = m(b_lon, b_lat)
            # m.plot()
            m.drawgreatcircle(a_lon, a_lat, b_lon, b_lat, linewidth=1, color='orange')

    plt.show()


def plot_learned_edges(file_path, args=None, index="ONI", resolution=5, reader=None, data_dir=None,
                       min_weight=0.99, plot_fraction=0.02,
                       only_plot_index_region_edges=True, plot_index_box=False, save_to=None, plot_arrows=False):
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
    model = torch.load(file_path, map_location='cpu')
    model.eval()

    if reader is not None:
        ssta = read_ssta(index, get_mask=False, stack_lon_lat=True, resolution=resolution, reader=reader)
        coordinates = ssta.indexes["cord"]
    else:
        from enso.utils import load_cnn_data
        data_dir = data_dir or args.data_dir
        _, _, GODAS, coordinates = load_cnn_data(window=args.window, lead_months=args.horizon, lon_min=args.lon_min,
                                                 lon_max=args.lon_max, lat_min=args.lat_min, lat_max=args.lat_max,
                                                 data_dir=data_dir, use_heat_content=args.use_heat_content,
                                                 return_new_coordinates=True)
    adj = model.adj_matrix.numpy()

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    title = f"{int(100 * plot_fraction)}% of the graph's edges, that have weight > {min_weight}"
    title = title + f" & start within the {index} region" if only_plot_index_region_edges else title
    ax.set_title(title)
    print("# Nonzero Edges:", np.count_nonzero(adj))
    plot_fraction = int(1000 * plot_fraction)
    n_edges_in_index_region, total_edges = 0, int(np.count_nonzero(adj > min_weight))
    print(np.count_nonzero((adj >= min_weight) == (adj.T >= min_weight)), "<--- bidirectional connections")
    for i, neighbors in enumerate(adj):
        for j, weight in enumerate(neighbors):
            if weight < min_weight:
                continue

            a_lat = coordinates[i][0]
            a_lon = coordinates[i][1]
            b_lat = coordinates[j][0]
            b_lon = coordinates[j][1]

            if is_in_oni_region(a_lat, a_lon) or is_in_oni_region(b_lat, b_lon):
                n_edges_in_index_region += 1
            elif only_plot_index_region_edges:
                continue

            if np.random.randint(0, 1000) > plot_fraction or b_lon == 180 or a_lon == 180:
                continue

            a_lon = transform(a_lon)
            b_lon = transform(b_lon)
            color = np.random.choice(['blue', 'brown', 'darkgreen', "darkorange", "purple"])
            if plot_arrows:
                plt.annotate('', xy=(b_lon, b_lat), xytext=(a_lon, a_lat),
                             size=10, xycoords='data',
                             arrowprops=dict(facecolor=color, ec='none', arrowstyle="fancy",
                                             connectionstyle="arc3,rad=-0.3"))
            else:
                plt.plot([a_lon, b_lon], [a_lat, b_lat],
                         color=color, linewidth=1, marker="o",
                         transform=ccrs.Geodetic()
                         )
                '''
                 a = plt.arrow(a_lon, a_lat, b_lon - a_lon, b_lat - a_lat,
                              fc=color, ec="none", linewidth=1.5, head_width=2, head_length=1.5,
                              transform=ccrs.Geodetic(),
                              )
                a.set_closed(False)
                '''
    '''
    plt.plot([50, -60], [-40, 40],
             color='red', linewidth=8,
             transform=ccrs.PlateCarree(),
             )
    plt.plot([-60, 50], [-40, 40],
             color='red', linewidth=8,
             transform=ccrs.PlateCarree(),
             )
    '''

    print(f"#Total edges with edge weight > {min_weight} =", total_edges)
    print(f"{n_edges_in_index_region} edges (a, b) with at least one node a,b within {index} region")

    if plot_index_box:
        lw, color, transf, line_style = 2, "red", ccrs.PlateCarree(), "--"
        (lat_min, lat_max), (lon_min, lon_max) = get_index_region_bounds(index)
        lon_min, lon_max = transform(lon_min), transform(lon_max)
        plt.plot([lon_min, lon_max], [lat_min, lat_min], line_style, color=color, linewidth=lw, transform=transf)
        plt.plot([lon_min, lon_max], [lat_max, lat_max], line_style, color=color, linewidth=lw, transform=transf)

        plt.plot([lon_min, lon_min], [lat_min, lat_max], line_style, color=color, linewidth=lw, transform=transf)
        plt.plot([lon_max, lon_max], [lat_min, lat_max], line_style, color=color, linewidth=lw, transform=transf)

        # TODO annotate bounding box nicely
        """
        plt.annotate(index, xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
        """
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()


def heatmap_of_edges(file_path, args=None, index="ONI", resolution=5, min_weight=0.1, reader=None, data_dir=None,
                     save_to=None, plot_index_box=False, plot_heatmap=True):
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
    model = torch.load(file_path, map_location='cpu')
    model.eval()

    if reader is not None:
        ssta = read_ssta(index, get_mask=False, stack_lon_lat=True, resolution=resolution, reader=reader)
        coordinates = ssta.indexes["cord"]
        lats, lons = ssta.attrs["Lats"], ssta.attrs["Lons"]
    else:
        from enso.utils import load_cnn_data
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
    ax2 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree(central_longitude=cm))

    minlon = -180 + cm
    maxlon = +179 + cm
    ax1.set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree()), ax2.set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())
    # ax2.stock_img(), ax3.stock_img()
    # ax1.patch.set_facecolor(color='black')#, ax2.patch.set_facecolor(color='black')

    ax1.set_title("Heatmap of summed edge weights that point towards ONI region")
    ax2.set_title("Heatmap of summed edge weights that point out of the ONI region")
    for ax, heat in zip([ax1, ax2], [outgoing_edge_heat, incoming_edge_heat]):
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

    ax1.coastlines(), ax2.coastlines()

    if plot_index_box:
        for ax in [ax1, ax2]:
            lw, color, transf, line_style = 2, "red", ccrs.PlateCarree(), "--"
            (lat_min, lat_max), (lon_min, lon_max) = get_index_region_bounds(index)
            lon_min, lon_max = transform(lon_min), transform(lon_max)
            ax.plot([lon_min, lon_max], [lat_min, lat_min], line_style, color=color, linewidth=lw, transform=transf)
            ax.plot([lon_min, lon_max], [lat_max, lat_max], line_style, color=color, linewidth=lw, transform=transf)

            ax.plot([lon_min, lon_min], [lat_min, lat_max], line_style, color=color, linewidth=lw, transform=transf)
            ax.plot([lon_max, lon_max], [lat_min, lat_max], line_style, color=color, linewidth=lw, transform=transf)

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()


def plot_edges_for_diff_weights(reader, save_plots_to, args, file_path_to_model, min_weights=None, plot_arrows=False, data_dir=None):
    if min_weights is None:
        min_weights = [0.75, 0.95, 0.99]
    save_to = None
    for w in min_weights:
        if save_plots_to is not None:
            save_to = create_filepath(prefix=save_plots_to, lead=args.horizon, i=args.index, w=args.window,
                                      normed=args.normalize, res=args.resolution, nodes=args.num_nodes,
                                      layers=args.layers, gcnDepth=args.gcn_depth, minW=w, suffix=".png")

        plot_learned_edges(file_path_to_model, args=args, reader=reader, plot_fraction=1.0, resolution=args.resolution,
                           index=args.index, plot_index_box=True, only_plot_index_region_edges=True, min_weight=w,
                           save_to=save_to, plot_arrows=plot_arrows, data_dir=data_dir or args.data_dir)
    if save_plots_to is not None:
        save_to = create_filepath(prefix=save_plots_to, lead=args.horizon, i=args.index, w=args.window,
                                  normed=args.normalize, res=args.resolution, nodes=args.num_nodes,
                                  layers=args.layers, gcnDepth=args.gcn_depth, minW=min_weights[-1],
                                  suffix="_ALL_edges.png")
    plot_learned_edges(file_path_to_model, args=args, reader=reader, plot_fraction=1.0, resolution=args.resolution,
                       index=args.index, plot_index_box=True, only_plot_index_region_edges=False, min_weight=min_weights[-1],
                       save_to=save_to, plot_arrows=plot_arrows, data_dir=data_dir or args.data_dir)


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


def plot_time_series_per_year(data, *args, years_per_subplot=3, labels=["timeseries"], time_steps=None, data_std=None,
                              linewidth=2, timeaxis="time", ylabel="Nino3.4 index", plot_months=False, show=True,
                              start_with_month=None):
    """

    :param data:
    :param args:
    :param years_per_subplot:
    :param labels:
    :param time_steps:
    :param data_std:
    :param linewidth:
    :param timeaxis:
    :param ylabel:
    :param plot_months:
    :param show:
    :param start_with_month: None or a number in {1,2,... 12}
                             If not None, the timeseries will be trimmed s.t. it starts with the provided month
    :return:
    """
    from pylab import subplot

    if time_steps is not None:
        time = time_steps
    elif isinstance(data, xa.DataArray):
        time = data.get_index(timeaxis)
    else:
        time = np.arange(0, data.shape[0], 1)
    if isinstance(time, pd.DatetimeIndex) and start_with_month is not None:
        first_month = int(time[:12].where(time[:12].month == start_with_month).notna().nonzero()[0])
        time, data = time[first_month:], data[first_month:]
        for arg in args:
            arg = arg[first_month:]
    n_years = len(time) / 12
    number_of_subplots = int((n_years + 1) / years_per_subplot)

    series = np.array(data)
    minimum, maximum = np.min(data), np.max(data)
    for other_series in args:
        minimum, maximum = min(minimum, np.min(other_series)), max(maximum, np.max(np.max(other_series)))

    i = 0
    for n_plot, v in enumerate(range(number_of_subplots)):
        v = v + 1
        ax = subplot(number_of_subplots, 1, v)
        ax.set_ylim([minimum - 0.1, maximum + 0.1])
        ax.set_ylabel(ylabel)
        # ax1.set_yticks(np.arange(np.round(minimum - 0.5, 0), np.round(maximum + 0.51, 0), 0.5))
        colors = [None for _ in range(len(args) + 1)]  # len(args) + 1 for main data series

        for j in range(i, i + years_per_subplot * 12, 12):
            t, year_series = time[j:j + 12], series[j:j + 12]
            p = ax.plot(t, year_series, label=labels[0], linewidth=linewidth, color=colors[0])
            colors[0] = p[-1].get_color()
            if data_std is not None:
                plt.fill_between(t, year_series - data_std[j:j + 12], year_series + data_std[j:j + 12],
                                 alpha=0.25, color=colors[0])
            for other_series_i, other_series in enumerate(args, 1):
                try:
                    p_i = ax.plot(t, other_series[j:j + 12], label=labels[other_series_i],
                                  linewidth=linewidth, color=colors[other_series_i])
                    colors[other_series_i] = p_i[-1].get_color()
                except ValueError as e:
                    raise ValueError("Please align the timeseries to the same time axis.", e)
                except IndexError:
                    raise IndexError("You must pass as many entries in labels, as there are time series to plot")
        if n_plot == number_of_subplots - 1:  # last subplot.. --> plot the whole x-axis
            ax.set_xlabel("Time")
            nth_month = 6
            if plot_months and isinstance(time[0], pd._libs.tslibs.Timestamp):
                xticks, year_mon = time[i::nth_month][:-1], [f"{date.year}-{date.month}" for date in
                                                             time[i::nth_month][:-1]]
                xticks = xticks.append(pd.Index([time[-1]]))
                year_mon.append(f"{time[-1].year}-{time[-1].month}")  # add last month
                plt.xticks(ticks=xticks, labels=year_mon, rotation=10)
        ax.grid()
        i = i + years_per_subplot * 12

    # plt.legend(loc=3)
    plt.savefig(r"C:\Users\salva\OneDrive\Documentos\Projects\ProjectX\Figure_enso.png", bbox_inches='tight')
    if show:
        plt.show()
    return time


def plot_running_means(runnings):
    """

    :param runnings: plot index running mean for each value in runnings
    :return:
    """
    plt.figure()
    plt.title("Index running mean over k months")
    for k in runnings:
        data = read_ssta(k=k)
        time = data.get_index("time")
        series = np.array(data)

        plt.plot(time, series, label=k)
    plt.xlabel("Time")
    plt.ylabel("SSTA")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # ++++++++++++++++++++++ Sample code for running the plots  ++++++++++++++++++
    # file = "../model/model-ENSO-1m-3333n.pt"
    # plot_learned_edges(file, plot_fraction=0.95)  # plot_learned_gnn_edges(file)
    from enso.utils import load_cnn_data
    from ENSO_GNN.data.reader import read_hadsst

    ersstv5_index = read_index(index="ONI")
    print(ersstv5_index)
    cmip5, SODA, GODAS, cords = load_cnn_data(window=3, lead_months=6, lon_min=0,
                                              lon_max=355, lat_min=-10, lat_max=10,
                                              data_dir=r'C:/Users/salva/OneDrive/Documentos/Projects/ProjectX/Data/',
                                              use_heat_content=True,
                                              return_new_coordinates=True,
                                              target_months=12)

    print(GODAS)

    plot_time_series(GODAS[1], labels=["GODAS"])
    plot_time_series(ersstv5_index.sel(time=slice("1984-01", "2017-12")), GODAS[1], labels=["ERSSTv5", "GODAS"])
    hadsst = read_hadsst("../data/single_timeseries/nino3.4_index_1mon")
    plot_time_series(ersstv5_index, hadsst, labels=["ERSSTv5", "HadSST"])
    # print(np.corrcoef(np.array(ersstv5_index), hadsst))
    # print(np.corrcoef(np.array(ersstv5_index)[1000:], hadsst[1000:]))

    plot_running_means(runnings=[1, 3])
