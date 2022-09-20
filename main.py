# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import morton
import datetime
import sys
import time
from rich.console import Console
import multiprocessing
import functools
import warnings


def plot_Values(df, dff, maneuver_list):
    fig, ax = plt.subplots(4, gridspec_kw={'height_ratios': [3, 3, 3, 1]})

    df.plot(x='lon', y='lat', ax=ax[0], color = 'blue')
    # dff.plot.scatter(x='lon', y='lat', ax=ax[0], color = 'red')

    cnt = 0
    for maneuver in maneuver_list:
        maneuver_df_temp = df.loc[(df.ts > maneuver[0]) & (df.ts < maneuver[1])]
        maneuver_df_temp.plot.scatter(x='lon', y='lat', ax=ax[0], color='red')
        # ax[0].annotate("Maneuver: " + str(cnt), maneuver_df_temp.iloc[-1]['lon'], maneuver_df_temp.iloc[-1]['lat'])
        cnt += 1

    df.plot(x='ts', y=['accel_lon', 'accel_trans', 'accel_down'], ax=ax[1])
    # Kästen mit Maneuvern plotten
    for maneuver in maneuver_list:
        ax[1].add_patch(
            Rectangle((maneuver[0], -4), maneuver[1] - maneuver[0], 8, fill=False, color='red', lw=1.5))
    # rote Punkte in 2. Graph
    dff.plot.scatter(x='ts', y='accel_lon', color='red', ax=ax[1])
    dff.plot.scatter(x='ts', y='accel_trans', color='red', ax=ax[1])

    df.plot(kind='scatter', x='accel_lon', y='accel_trans', color=df['ts'], ax=ax[2])
    # ax[2].add_patch(Rectangle((geofence[0][0], geofence[0][1]), geofence[1][0] - geofence[0][0],geofence[1][1] - geofence[0][1], fill=False, color='red', lw=2))
    dff.plot(kind='scatter', x='accel_lon', y='accel_trans', color='red', ax=ax[2])

    min = df['morton'].min()
    max = df['morton'].max()
    # min = 12000000000
    # max = 18000000000
    #min = 0
    #max = 30000000000
    bins = 400

    ax[3].hist(df['morton'], bins=bins, range=(min, max), color='blue')
    ax[3].hist(dff['morton'], bins=bins, range=(min, max), color='red')
    ax[3].set_xlim(min, max)
    ax[3].set_ylim(0, 1)

    fig.tight_layout()
    plt.show()

################################################################

def filter_Values(df, geofence, fence_x, fence_y):

    dff = df[((df[fence_x] > geofence[0][0]) & # linke Grenze
                (df[fence_x] < geofence[1][0]) & # rechte Grenze
                (df[fence_y] > geofence[0][1]) & # untere Grenze
                (df[fence_y] < geofence[1][1]))] # obere Grenze

    dff = dff.sort_values(by='ts').reset_index()
    dff['diff_to_prev'] = dff['ts'].diff()


    cnt_maneuver = len(dff[dff['diff_to_prev'] > 5000000].index) + 1

    date = dff['ts'].min() / 1000000
    dt = datetime.datetime.fromtimestamp(date, datetime.timezone(datetime.timedelta(hours=1)))

    print("First Maneuver detected at ", dt, " we detect ",cnt_maneuver, "maneuvers")

    return dff

################################################################

def filter_Morton(df, range_min, range_max):

    dff = df[((df['morton'] > range_min) &  # linke Grenze
              (df['morton'] < range_max))] # rechte Grenze


    dff = dff.sort_values(by='ts').reset_index()
    dff['diff_to_prev'] = dff['ts'].diff()

    cnt_maneuver = len(dff[dff['diff_to_prev'] > 5000000].index) + 1

    date = dff['ts'].min() / 1000000
    dt = datetime.datetime.fromtimestamp(date, datetime.timezone(datetime.timedelta(hours=1)))

    print("First Maneuver detected at ", dt, " we detect ", cnt_maneuver, "maneuvers")

    return dff

################################################################

def precondition_data(df, max_accel):
    length_prev = len(df)
    df = df.drop(df[(df.accel_lon < (max_accel * -1)) | (df.accel_lon > max_accel)].index)
    df = df.drop(df[(df.accel_trans < (max_accel * -1)) | (df.accel_trans > max_accel)].index)
    print("Ignore",length_prev - len(df), "out-of-range-values.")
    return df

################################################################

def calc_Morton(df, dimension, bits, offset, faktor_multiply):

    df['accel_lon_mult'] = df['accel_lon'].add(offset)
    df['accel_trans_mult'] = df['accel_trans'].add(offset)
    df['accel_down_mult'] = df['accel_down'].add(offset)


    df['accel_lon_mult'] = df.apply(lambda x: int(x['accel_lon_mult'] * faktor_multiply), axis=1)
    df['accel_trans_mult'] = df.apply(lambda x: int(x['accel_trans_mult'] * faktor_multiply), axis=1)
    df['accel_down_mult'] = df.apply(lambda x: int(x['accel_down_mult'] * faktor_multiply), axis=1)

    m = morton.Morton(dimensions=dimension, bits=bits)

    def set_value(row):
        return m.pack(int(row['accel_lon_mult']), int(row['accel_trans_mult']))

    df['morton'] = df.apply(set_value, axis=1)

    return df, m

################################################################

def generate_ts(df):

    df['ts'] = (df['sampleTimeStamp.seconds'] * 1000000) + df['sampleTimeStamp.microseconds']
    df = df.drop(columns=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds'], axis=1)

    return df

################################################################

def generate_Search_Mask(geofence_list, dim, bits, geofence_resolution, offset, faktor_multiply, store, m):
    time_filter_start = time.time()
    geofence_generate_parallel = []

    for geofence_temp in geofence_list:
        searchmask_name = str(dim) + '/' + str(bits) + '/' + str(geofence_resolution) + '/' + str(offset) + '/' + str(faktor_multiply) + '/SearchMask_' + str(
            geofence_temp[0][0]) + '_' + str(geofence_temp[0][1]) + '_' + str(geofence_temp[1][0]) + '_' + str(geofence_temp[1][1])

        if not searchmask_name in store:
            if generate_masks_multithread == True:
                geofence_generate_parallel.append(geofence_temp)
            else:
                print(searchmask_name + ' is not in store; Lets create it. This takes some time...')
                time_filter_start = time.time()
                search_mask_temp = transfer_Geofence_to_Morton(geofence_temp, m, bits, 1, offset, faktor_multiply)
                search_mask = pd.concat([search_mask, search_mask_temp], axis = 0)
                time_filter_end = time.time()
                print("Time to transfer geofence in morton", round(time_filter_end - time_filter_start, 5), "s; containing",
                  len(search_mask.index), "values.")
                store[searchmask_name] = search_mask
    # generate Masks Multithread
    if len(geofence_generate_parallel) > 0:
        print("Generate missing Searchmasks...")
        mp_handler(geofence_generate_parallel, dim, bits, geofence_resolution, m, offset, faktor_multiply)
    time_filter_end = time.time()


    search_mask = pd.DataFrame(columns=['morton'])
    print("Load Search Masks from Storage.")
    #load Masks
    for geofence_temp in geofence_list:
        searchmask_name = str(dim) + '/' + str(bits) + '/' + str(geofence_resolution) + '/' + str(offset) + '/' + str(faktor_multiply) + '/SearchMask_' + str(geofence_temp[0][0]) + '_' + str(geofence_temp[0][1]) + '_' + str(geofence_temp[1][0]) + '_' + str(geofence_temp[1][1])

        if searchmask_name in store:
            #print("Load SearchMask: ", str(searchmask_name), " from storage.")
            #time_filter_start = time.time()
            search_mask_temp = store.get(searchmask_name)
            search_mask = pd.concat([search_mask, search_mask_temp], axis=0)
            #time_filter_end = time.time()
            #print("Took", round(time_filter_end - time_filter_start, 5), "s; containing",
            #      len(search_mask.index), "values.")
        else:
            sys.exit("Error while loading SearchMask.")

    print("Done: SearchMask contains", len(search_mask.index), "values.")

    return search_mask

################################################################

def transfer_Geofence_to_Morton(geofence, m, resolution, resolution_search_space, offset, faktor_multiply):

    A = [0,0] #geofence[0]
    A[0] = int((geofence[0][0] + offset) * faktor_multiply)
    A[1] = int((geofence[0][1] + offset) * faktor_multiply)
    C = [0,0] #geofence[1]
    C[0] = int((geofence[1][0] + offset) * faktor_multiply)
    C[1] = int((geofence[1][1] + offset) * faktor_multiply)
    B = [A[0], C[1]]
    D = [A[1], C[0]]

    search_space = [m.pack(A[0], A[1]), m.pack(C[0], C[1])] # geofence in morton bereich; erste Wertebereich ermittlung

    #ax.add_patch(Rectangle((A[0]-0.25, A[1]-0.25), C[0]-A[0]+0.5, C[1]-A[1]+0.5, fill=False, color='red', lw = 2))

    #df_array[(df_array.morton >= search_space[0]) & (df_array.morton <= search_space[1])].sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax, label="SearchSpace")

    np_ar = np.arange(search_space[0], (search_space[1]+1), resolution_search_space)
    search_mask = pd.DataFrame(np_ar, columns = ['morton'])
    #print(len(search_mask))

    # search_mask = df_array[(df_array.morton >= search_space[0]) & (df_array.morton <= search_space[1])]

    min = 0
    max = (2**resolution)-1

    console = Console()
    with console.status("[bold green] Transform Geofence...") as status:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min, min, max, max, offset, faktor_multiply)

    #search_mask.sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax, label="SearchSpace")

    return search_mask

################################################################

def identifyNonRelvantAreas(m, geofence, search_mask, min_value_x, min_value_y, max_value_x, max_value_y, offset, faktor_multiply):

    # print("identification of non relevant areas", min_value_x, min_value_y, ";", max_value_x, max_value_y, "search_mask has", len(search_mask.index), "lines.")

    if (m.pack(max_value_x, max_value_y) - m.pack(min_value_x, min_value_y)) <=3:
        return search_mask

    A = [0, 0]  # geofence[0]
    A[0] = int((geofence[0][0] + offset) * faktor_multiply)
    A[1] = int((geofence[0][1] + offset) * faktor_multiply)
    C = [0, 0]  # geofence[1]
    C[0] = int((geofence[1][0] + offset) * faktor_multiply)
    C[1] = int((geofence[1][1] + offset) * faktor_multiply)

    #A = geofence[0]
    #C = geofence[1]

    half_value_x = int(((max_value_x - min_value_x) / 2) + 0.5 + min_value_x)
    half_value_y = int(((max_value_y - min_value_y) / 2) + 0.5 + min_value_y)

    # search_mask = df_array[['x', 'y', 'morton']]

    Q1 = False
    Q2 = False
    Q3 = False
    Q4 = False

    if (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
        # alle
        Q1 = True
        Q2 = True
        Q3 = True
        Q4 = True
    elif (A[0] < half_value_x) & (A[1] >= half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
        # oben beide
        Q3 = True
        Q4 = True
    elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] < half_value_y):
        # unten beide
        Q1 = True
        Q2 = True
    elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] < half_value_x) & (C[1] >= half_value_y):
        # links beide
        Q1 = True
        Q3 = True
    elif (A[0] >= half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
        # rechts beide
        Q2 = True
        Q4 = True
    elif (A[0] < half_value_x) & (A[1] >= half_value_y) & (C[0] < half_value_x) & (C[1] >= half_value_y):
        # oben links
        Q3 = True
    elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] < half_value_x) & (C[1] < half_value_y):
        # unten links
        Q1 = True
    elif (A[0] >= half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] < half_value_y):
        # unten rechts
        Q2 = True
    elif (A[0] >= half_value_x) & (A[1] >= half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
        # oben rechts
        Q4 = True
    else:
        #irgendwas stimmt mit der eingabe nicht
        sys.exit("Geofence is incorrect; please check!")

    Q1_range = (m.pack(min_value_x, min_value_y), m.pack((half_value_x-1), (half_value_y-1)))
    Q2_range = (m.pack(half_value_x, min_value_y), m.pack(max_value_x, (half_value_y - 1)))
    Q3_range = (m.pack(min_value_x, half_value_y), m.pack((half_value_x - 1), max_value_y))
    Q4_range = (m.pack(half_value_x, half_value_y), m.pack(max_value_x, max_value_y))


    if Q1 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < Q1_range[1]+1) & (search_mask.morton >= Q1_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=min_value_x, min_value_y=min_value_y, max_value_x=half_value_x-1, max_value_y=half_value_y-1, offset=offset, faktor_multiply=faktor_multiply)
    if Q2 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < Q2_range[1] + 1) & (search_mask.morton >= Q2_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=half_value_x, min_value_y=min_value_y,
                                            max_value_x=max_value_x, max_value_y=half_value_y - 1, offset=offset, faktor_multiply=faktor_multiply)
    if Q3 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < Q3_range[1] + 1) & (search_mask.morton >= Q3_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=min_value_x, min_value_y=half_value_y,
                                            max_value_x=half_value_x - 1, max_value_y=max_value_y, offset=offset, faktor_multiply=faktor_multiply)
    if Q4 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < Q4_range[1] + 1) & (search_mask.morton >= Q4_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=half_value_x, min_value_y=half_value_y,
                                            max_value_x=max_value_x, max_value_y=max_value_y, offset=offset, faktor_multiply=faktor_multiply)

    return search_mask


################################################################

def define_geofences(geofence_resolution, geofence):

    if not (geofence[0][0] < geofence[1][0]) & (geofence[0][1] < geofence[1][1]):
        sys.exit("Geofence is not correct, expect: A[0][0] < B[1][0] & A[0][1] < B[1][1]")

    if not ((geofence[0][0] % geofence_resolution) == 0) & ((geofence[0][1] % geofence_resolution) == 0) & ((geofence[1][0] % geofence_resolution) == 0) & ((geofence[1][1] % geofence_resolution) == 0):
        sys.exit("GeofenceResolution is not common multiple of geofence values.")

    geofence_list = []
    for i in np.arange(geofence[0][0],geofence[1][0],geofence_resolution): # accel_long
        for j in np.arange(geofence[0][1],geofence[1][1],geofence_resolution): # accel_trans
            geofence_list.append([[i, j], [(i+geofence_resolution), (j+geofence_resolution)]])

    # print(geofence_list)
    # print(len(geofence_list))

    return geofence_list


################################################################

def mp_worker(m, bits, offset, faktor_multiply, geofence_temp):
    print("Generate Searchmask ", geofence_temp)
    time_filter_start = time.time()
    search_mask = transfer_Geofence_to_Morton(geofence=geofence_temp, m=m, resolution=bits, resolution_search_space=1, offset=offset, faktor_multiply=faktor_multiply)
    time_filter_end = time.time()
    #print("Done: ",geofence_temp, "Duration:", str(datetime.timedelta(seconds=round(time_filter_end - time_filter_start, 5))))
    return search_mask, geofence_temp

def mp_handler(geofence_list, dim, bits, res_searchmask, m, offset, faktor_multiply):
    p = multiprocessing.Pool(int(multiprocessing.cpu_count()))
    partial_call = functools.partial(mp_worker, m, bits, offset, faktor_multiply)

    for search_mask, geofence_temp in p.map(partial_call, geofence_list):
        searchmask_name = str(dim) + '/' + str(bits) + '/' + str(geofence_resolution) + '/' + str(offset) + '/' + str(faktor_multiply) + '/SearchMask_' + str(geofence_temp[0][0]) + '_' + str(geofence_temp[0][1]) + '_' + str(geofence_temp[1][0]) + '_' + str(geofence_temp[1][1])

        store[searchmask_name] = search_mask


################################################################

def scene_filter(df, search_mask):

    filter = df["morton"].isin(search_mask['morton'])

    return df[filter]

################################################################

def detect_single_maneuver(df, search_mask, min_duration_maneuver):

    df_relevant_values = scene_filter(df, search_mask)

    df_relevant_values = df_relevant_values.sort_values(by='ts').reset_index()
    df_relevant_values['diff_to_prev'] = df_relevant_values['ts'].diff()

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000


    maneuver_start_list = list(df_relevant_values[df_relevant_values['diff_to_prev'] > min_time_between_two_independent_maneuvers].index)
    maneuver_start_list.append(df_relevant_values.index[-1])


    maneuver_start_idx = 0
    maneuver_start_ts = df_relevant_values.loc[maneuver_start_idx, 'ts']

    maneuver_list = []
    for maneuver_idx in maneuver_start_list:

        maneuver_end_ts = df_relevant_values.loc[maneuver_idx-1, 'ts']

        if (maneuver_end_ts - maneuver_start_ts) > min_duration_maneuver:

            maneuver_list.append([maneuver_start_ts, maneuver_end_ts])

            # start_date = datetime.datetime.fromtimestamp(maneuver_start_ts/1000000, datetime.timezone(datetime.timedelta(hours=1)))
            # end_date = datetime.datetime.fromtimestamp(maneuver_end_ts / 1000000,datetime.timezone(datetime.timedelta(hours=1)))
            # print("Maneuver detected! Start:",start_date , " End:", end_date)

        maneuver_start_idx = maneuver_idx
        maneuver_start_ts = df_relevant_values.loc[maneuver_start_idx, 'ts']

    return maneuver_list

################################################################

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("Välkommen!")

    generate_masks_multithread = True
    offset = 10
    faktor_multiply = 100
    geofence_resolution = 0.25
    max_accel = 10
    bits = 18
    dim = 2


    store = pd.HDFStore("searchMaskStorage.h5")

    # geofence
    geofence = [[-1.5,-4],[-0.25,-0.75]] # rechtskurve, beschleunigung
    #geofence = [[-1.5, 0.75], [1, 4]]  # linksskurve

    min_duration_maneuver = 500000 # 30000000 # 500000 # roughly 10 datapoints
    min_time_between_two_independent_maneuvers = 500000

    geofence_list = define_geofences(geofence_resolution=geofence_resolution, geofence=geofence)
    ################################################################

    print("Load_Database...")
    #df = pd.read_csv('../Data/Ausschnitte/opendlv.device.gps.pos.Grp1Data-0.csv', sep=';',
    #                 usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
    #                          'accel_lon', 'accel_trans', 'accel_down'])
    # df = pd.read_csv('../Data/Ausschnitte/Hard_Braking/braking_cut_8_brakes.csv', sep=';',
    #                  usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
    #                           'accel_lon', 'accel_trans', 'accel_down'])
    df = pd.read_csv('../Data/Ausschnitte/LaneChange/lanechange_single.csv', sep=';',
                      usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
                               'accel_lon', 'accel_trans', 'accel_down'])

    ################################################################

    df = generate_ts(df)
    df = precondition_data(df, max_accel)

    df, m = calc_Morton(df=df, dimension=2, bits=bits, offset=offset, faktor_multiply=faktor_multiply)

    ################################################################

    print("Define Search Mask.")
    search_mask = generate_Search_Mask(geofence_list, dim, bits, geofence_resolution, offset, faktor_multiply, store, m)

    ################################################################

    print("Filter Data.")
    time_filter_start = time.time()
    df_relevant_values = scene_filter(df, search_mask)
    time_filter_end = time.time()

    print("Done: Time to identify relevant values in database:", round(time_filter_end-time_filter_start, 10), "s; containing", len(df_relevant_values.index), "relevant values.")

    maneuver_list = detect_single_maneuver(df, search_mask, min_duration_maneuver)

    ################################################################
    plot_Values(df, df_relevant_values, maneuver_list)

    fig, ax = plt.subplots(1)
    df.plot(x='morton', y='ts', color='blue', ax=ax)
    df_relevant_values.plot.scatter(x='morton', y='ts', color='red', ax=ax)

    geofence = [[-1.5, 0.75], [-0.25, 4]]
    geofence_list = define_geofences(geofence_resolution=geofence_resolution, geofence=geofence)

    search_mask = generate_Search_Mask(geofence_list, dim, bits, geofence_resolution, offset, faktor_multiply, store, m)
    df_relevant_values = scene_filter(df, search_mask)

    df_relevant_values.plot.scatter(x='morton', y='ts', color='green', ax=ax)

    plt.show()

    store.close()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
