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


def plot_Values(df, dff, geofence):
    fig, ax = plt.subplots(4, gridspec_kw={'height_ratios': [3, 3, 3, 1]})

    df.plot(x='lon', y='lat', ax=ax[0], color = 'blue')
    dff.plot.scatter(x='lon', y='lat', ax=ax[0], color = 'red')

    df.plot(x='ts', y=['accel_lon', 'accel_trans', 'accel_down'], ax=ax[1])
    accelBoxes = False
    if accelBoxes == True:
        maneuver_start = dff['ts'][0]
        maneuver_end = 0
        maneuver_cnt = dff[((dff['diff_to_prev'] > 5000000) | (dff['diff_to_prev'].isna() == True))].index

        for i in maneuver_cnt:
            if i > 0:
                maneuver_end = int(dff["ts"][i - 1])
                ax[1].add_patch(Rectangle((maneuver_start, -4), maneuver_end - maneuver_start, 12, fill=False, color='red', lw=2))
                maneuver_start = dff['ts'][i]

            if i == maneuver_cnt[-1]:
                maneuver_end = int(dff["ts"][dff.index[-1]])
                ax[1].add_patch(Rectangle((maneuver_start, -4), maneuver_end - maneuver_start, 12, fill=False, color='red', lw=2))

    else:
        dff.plot.scatter(x='ts', y='accel_lon', color='red', ax=ax[1])
        dff.plot.scatter(x='ts', y='accel_trans', color='red', ax=ax[1])

    df.plot(kind='scatter', x='accel_lon', y='accel_trans', color=df['ts'], ax=ax[2])
    # ax[2].add_patch(Rectangle((geofence[0][0], geofence[0][1]), geofence[1][0] - geofence[0][0],geofence[1][1] - geofence[0][1], fill=False, color='red', lw=2))
    dff.plot(kind='scatter', x='accel_lon', y='accel_trans', color='red', ax=ax[2])

    min = df['morton'].min()
    max = df['morton'].max()
    # min = 12000000000
    # max = 18000000000
    min = 0
    max = 30000000000
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


################################################################

def calc_Morton(df, dimension, bits):

    offset = 10

    df['accel_lon_mult'] = df['accel_lon'].add(offset)
    df['accel_trans_mult'] = df['accel_trans'].add(offset)
    df['accel_down_mult'] = df['accel_down'].add(offset)

    faktor_multiply = 10000

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

def transfer_Geofence_to_Morton(geofence, m, resolution, resolution_search_space):
    offset = 10
    faktor_multiply = 10000

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

    np_ar = np.arange(search_space[0], search_space[1], resolution_search_space)
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
        search_mask = search_mask.drop(search_mask[(search_mask.morton < Q1_range[1]+1) & (search_mask.morton > Q1_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=min_value_x, min_value_y=min_value_y, max_value_x=half_value_x-1, max_value_y=half_value_y-1, offset=offset, faktor_multiply=faktor_multiply)
    if Q2 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < Q2_range[1] + 1) & (search_mask.morton > Q2_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=half_value_x, min_value_y=min_value_y,
                                            max_value_x=max_value_x, max_value_y=half_value_y - 1, offset=offset, faktor_multiply=faktor_multiply)
    if Q3 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < Q3_range[1] + 1) & (search_mask.morton > Q3_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=min_value_x, min_value_y=half_value_y,
                                            max_value_x=half_value_x - 1, max_value_y=max_value_y, offset=offset, faktor_multiply=faktor_multiply)
    if Q4 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < Q4_range[1] + 1) & (search_mask.morton > Q4_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=half_value_x, min_value_y=half_value_y,
                                            max_value_x=max_value_x, max_value_y=max_value_y, offset=offset, faktor_multiply=faktor_multiply)

    return search_mask


################################################################

def define_geofences(geofence_resolution):

    # geofence_list = [[[stark_beschl_min, links_min], [stark_beschl_max, links_max]], # 1, stark beschleunigen links
    #                  [[stark_beschl_min, gerade_min], [stark_beschl_max, gerade_max]], # 2, stark beschleunigen gerade
    #                  [[stark_beschl_min, rechts_min], [stark_beschl_max, rechts_max]],  # 3, stark beschleunigen rechts
    #                  [[leicht_beschl_min, links_min], [leicht_beschl_max, links_max]],  # 4, leicht beschleunigen links
    #                  [[leicht_beschl_min, gerade_min], [leicht_beschl_max, gerade_max]],  # 5, leicht beschleunigen gerade
    #                  [[leicht_beschl_min, rechts_min], [leicht_beschl_max, rechts_max]],  # 6, leicht beschleunigen rechts
    #                  [[no_beschl_min, links_min], [no_beschl_max, links_max]],  # 7, nicht beschleunigen links
    #                  [[no_beschl_min, gerade_min], [no_beschl_max, gerade_max]], # 8, nicht beschleunigen gerade
    #                  [[no_beschl_min, rechts_min], [no_beschl_max, rechts_max]], # 9, nicht beschleunigen rechts
    #                  [[leicht_brems_min, links_min], [leicht_brems_max, links_max]],  # 10, leicht bremsen links
    #                  [[leicht_brems_min, gerade_min], [leicht_brems_max, gerade_max]],  # 11, leicht bremsen gerade
    #                  [[leicht_brems_min, rechts_min], [leicht_brems_max, rechts_max]],  # 12, leicht bremsen rechts
    #                  [[stark_brems_min, links_min], [stark_brems_max, links_max]],  # 13, stark bremsen links
    #                  [[stark_brems_min, gerade_min], [stark_brems_max, gerade_max]],  # 14, stark bremsen gerade
    #                  [[stark_brems_min, rechts_min], [stark_brems_max, rechts_max]],  # 15, stark bremsen rechts
    #                  ]

    geofence_list = []
    for i in np.arange(-4,4,geofence_resolution):
        for j in np.arange(-2,2,geofence_resolution):
            geofence_list.append([[i, j], [(i+geofence_resolution), (j+geofence_resolution)]])

    # print(geofence_list)
    print(len(geofence_list))

    return geofence_list


################################################################

def mp_worker(m, bits, geofence_temp):
    print("Work on ", geofence_temp)
    time_filter_start = time.time()
    search_mask = transfer_Geofence_to_Morton(geofence=geofence_temp, m=m, resolution=bits, resolution_search_space=1)
    time_filter_end = time.time()
    print("Done: ",geofence_temp, "Duration:", str(datetime.timedelta(seconds=round(time_filter_end - time_filter_start, 5))))
    return search_mask, geofence_temp

def mp_handler(geofence_list, dim, bits, res_searchmask, m):
    p = multiprocessing.Pool(3) #int(multiprocessing.cpu_count()/8)
    partial_call = functools.partial(mp_worker, m, bits)

    for search_mask, geofence_temp in p.map(partial_call, geofence_list):
        searchmask_name = str(dim) + '/' + str(bits) + '/' + str(res_searchmask) + '/SearchMask_' + str(
            geofence_temp[0][0]) + '_' + str(geofence_temp[0][1]) + '_' + str(geofence_temp[1][0]) + '_' + str(
            geofence_temp[1][1])
        store[searchmask_name] = search_mask
    return search_mask


################################################################

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("Välkommen!")
    print("Load_Database...")

    generate_masks = True

#    df = pd.read_csv('C:/Users/LukasB/Documents/Chalmers/Data/Ausschnitte/Hard_Braking/braking_cut_8_brakes.csv', sep=';',
#                     usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
#                              'accel_lon', 'accel_trans', 'accel_down'])
    df = pd.read_csv('../Data/Ausschnitte/Hard_Braking/braking_cut_8_brakes.csv', sep=';',
                     usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
                              'accel_lon', 'accel_trans', 'accel_down'])
    # df.rename(columns = {'timestamp:10881:<lon>':'ts', 'accel_lon:10881:<double>':'accel_lon', 'accel_trans:10881:<double>':'accel_trans', 'accel_down:10881:<double>':'accel_down'}, inplace = True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    ################################################################


    ################################################################

    df = generate_ts(df)

    ################################################################
    bits = 18
    dim = 2
    res_searchmask = 1

    df, m = calc_Morton(df=df, dimension=2, bits=bits)

    ################################################################

    #time_filter_start = time.time()

    #geofence = [[0.5, -4], [2, -1]]
    #geofence = [[0.0, 0.0], [1, 1]]
    #geofence = [[0.1, 0.1], [0.5, 0.5]]

    #searchmask_name = str(dim) + '/' + str(bits) + '/' + str(res_searchmask) + '/SearchMask_' + str(
    #    geofence[0][0]) + '_' + str(geofence[0][1]) + '_' + str(geofence[1][0]) + '_' + str(
    #    geofence[1][1])

    #threshold, longitudinal
    stark_beschl_min = -8
    stark_beschl_max = -3
    leicht_beschl_min = -3
    leicht_beschl_max = -0.5
    no_beschl_min = -0.5
    no_beschl_max = 0.5
    leicht_brems_min = 0.5
    leicht_brems_max = 4
    stark_brems_min = 4
    stark_brems_max = 10

    # threshold, lateral
    links_max = 4
    links_min = 0.5
    gerade_max = 0.5
    gerade_min = -0.5
    rechts_max = -0.5
    rechts_min = -4

    # geofence_list = [geofence]
    geofence_list = define_geofences(geofence_resolution=1)

    fence_x = 'accel_lon'
    fence_y = 'accel_trans'

    store = pd.HDFStore("searchMaskStorage.h5")

    ################################################################

    print("Transfer fence in Morton-Space.")

    # dff = filter_Values(df, geofence, fence_x=fence_x, fence_y=fence_y)
    # dff = filter_Morton(df, 25000000000, 30000000000) # stark Bremsen
    # dff = filter_Morton(df, 10000000000, 13000000000) # starke Beschleunigung
    # dff = filter_Morton(df, 14000000000, 14800000000) # linkskurve Bremsen

    #time_filter_end = time.time()
    #print("Time to set geofence and filter with threshold value", len(df.index), "rows:", round(time_filter_end-time_filter_start, 5), "s")
    ################################################################

    #search_mask = pd.read_pickle("/SearchMask/" + geofence[0][0] + "_" + geofence[0][1] + "_" + geofence[1][0] + "_" + geofence[1][0] + "_" + dim + "_" + bits + "_" + res_searchmask)

    if generate_masks == True:
        time_filter_start = time.time()
        search_mask = mp_handler(geofence_list, dim, bits, res_searchmask, m)
        time_filter_end = time.time()

        print("Time parallel generation:", round(time_filter_end - time_filter_start, 5), "s")
    else:
        time_filter_start = time.time()
        for geofence_temp in geofence_list:
            searchmask_name = str(dim) + '/' + str(bits) + '/' + str(res_searchmask) + '/SearchMask_' + str(
                geofence_temp[0][0]) + '_' + str(geofence_temp[0][1]) + '_' + str(geofence_temp[1][0]) + '_' + str(geofence_temp[1][1])
            if searchmask_name in store:
                print("Load SearchMask from storage.")
                time_filter_start = time.time()
                search_mask = store.get(searchmask_name)
                time_filter_end = time.time()
                print("Took", round(time_filter_end - time_filter_start, 5), "s; containing",
                      len(search_mask.index), "values.")
            else:
                print(searchmask_name + ' is not in store; Lets create it. This takes some time...')

                time_filter_start = time.time()
                search_mask = transfer_Geofence_to_Morton(geofence_temp, m, bits, 1)
                time_filter_end = time.time()

                print("Time to transfer geofence in morton", round(time_filter_end - time_filter_start, 5), "s; containing",
                      len(search_mask.index), "values.")

                store[searchmask_name] = search_mask
        time_filter_end = time.time()
        print("Time sequentiell generation:", round(time_filter_end - time_filter_start, 5), "s")

    store.close()

    # df_relevant_values = df.drop(df[(df.morton < Q1_range[1]+1) & (search_mask.morton > Q1_range[0])].index)
    #df_relevant_values = df

    time_filter_start = time.time()

    filter = df["morton"].isin(search_mask['morton'])
    df_relevant_values = df[filter]

    time_filter_end = time.time()

    print("Time to identify relevant values in database:", round(time_filter_end-time_filter_start, 10), "s; containing", len(df_relevant_values.index), "values.")

    ################################################################
    plot_Values(df, df_relevant_values, geofence)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
