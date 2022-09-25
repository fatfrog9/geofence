# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

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
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import base64
import math
from tqdm import tqdm

class Setup:
    def __init__(self, generate_masks_multithread, offset, faktor_multiply, geofence_resolution, max_accel, bits, dim):
        self.generate_masks_multithread = generate_masks_multithread
        self.offset = offset
        self.faktor_multiply = faktor_multiply
        self.geofence_resolution = geofence_resolution
        self.max_accel = max_accel
        self.bits = bits
        self.dim = dim

    def __str__(self):
        return 'generate_masks_multithread ' + str(self.generate_masks_multithread) + os.linesep\
               + 'offset:' + str(self.offset) + os.linesep \
               + 'faktor_multiply:' + str(self.faktor_multiply) + os.linesep \
               + 'geofence_resolution:' + str(self.geofence_resolution) + os.linesep \
               + 'max_accel:' + str(self.max_accel) + os.linesep \
               + 'bits:' + str(self.bits) + os.linesep \
               + 'dim:' + str(self.dim)



class DrivingStatus:
    def __init__(self, name, fence, min_time, max_time, min_gap, max_gap, setup, color):
        self.name = name
        self.fence = fence
        self.min_time = min_time
        self.max_time = max_time
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.setup = setup
        self.color = color
        self.geofence_list = define_geofences(geofence_resolution=self.setup.geofence_resolution, geofence=self.fence)
        self.search_mask = generate_Search_Mask(self.geofence_list, self.setup, store)
        self.relevant_values_list = []
        self.maneuver_list = []

    def __str__(self):
        return 'DrivingStatus ' + str(self.name) + os.linesep\
               + 'Duration: ' + str(self.min_time) + ' / ' + str(self.max_time) + os.linesep\
               + 'Gap (after): ' + str(self.min_gap) + ' / ' + str(self.max_gap) + os.linesep\
               + 'Color: ' + str(self.color) + os.linesep\
               + 'Setup: ' + os.linesep + str(self.setup) + os.linesep\
               + 'Geofence_List:' + os.linesep + str(self.fence)


class Maneuver:
    def __init__(self, name, drivingStatus_list):
        self.name = name
        self.drivingStatus_list = drivingStatus_list
        self.start_time = 0
        self.end_time = 0
        self.start_lat = 0
        self.end_lat = 0
        self.start_lon = 0
        self.end_long = 0

    def __str__(self):
        return 'Maneuver ' + str(self.name) + os.linesep\
               + 'detected from ' + self.start_time + ' to ' + self.end_time


def plot_Values(df, df_relevant_values, driving_status_list, label):
    fig, ax = plt.subplots(4, gridspec_kw={'height_ratios': [3, 3, 3, 1]})

    # df.plot(x='lon', y='lat', ax=ax[0], color = 'blue')
    # # dff.plot.scatter(x='lon', y='lat', ax=ax[0], color = 'red')
    #
    # cnt = 0
    # for driving_status_temp in driving_status_list:
    #     color = 'red'
    #     for status in driving_status_temp:
    #         maneuver_df_temp = df.loc[(df.ts > status[0]) & (df.ts < status[1])]
    #         maneuver_df_temp.plot.scatter(x='lon', y='lat', ax=ax[0], color=color)
    #         # ax[0].annotate("Maneuver: " + str(cnt), maneuver_df_temp.iloc[-1]['lon'], maneuver_df_temp.iloc[-1]['lat'])
    #         cnt += 1

    df.plot(x='ts', y=['accel_lon', 'accel_trans'], ax=ax[1])
    # Kästen mit Maneuvern plotten

    for driving_status_temp in driving_status_list:
        color = 'red'
        for status in driving_status_temp:
            ax[1].add_patch(
                Rectangle((status[0], -3), status[1] - status[0], 6, fill=False, color='red', lw=1.5))

    df.plot(kind='scatter', x='accel_lon', y='accel_trans', color=df['ts'], ax=ax[2])
    ax[2].set_xlim(-10, 10)
    ax[2].set_ylim(-4, 4)

    min = df['morton'].min()
    max = df['morton'].max()
    # min = 12000000000
    # max = 18000000000
    #min = 0
    #max = 30000000000
    bins = 400

    # Morton space statisch
    ax[3].hist(df['morton'], bins=bins, range=(min, max), color='blue')
    ax[3].set_xlim(min, max)
    ax[3].set_ylim(0, 1)

    #morton space zeitabhängig
    # df.plot.scatter(x='ts', y='morton', color='blue', ax=ax[3])

    for idx, row in label.iterrows():
        ax[1].annotate(row['label'], (row['ts'], 2.5), rotation=60)

    # add rote Punkte
    for relevant_temp in df_relevant_values:
        # zweiter Gtraph
        #relevant_temp.plot.scatter(x='ts', y='accel_lon', color='red', ax=ax[1])
        #relevant_temp.plot.scatter(x='ts', y='accel_trans', color='red', ax=ax[1])
        # dritter Graph
        relevant_temp.plot(kind='scatter', x='accel_lon', y='accel_trans', color='red', ax=ax[2])
        # vierter Graph
        # morton statisch
        ax[3].hist(relevant_temp['morton'], bins=bins, range=(min, max), color='red')
        # morton zeitabhängig
        #relevant_temp.plot.scatter(x='ts', y='morton', color='red', ax=ax[3])

    #fig.tight_layout()
    plt.show()

################################################################

def filter_Values(df, geofence, fence_x, fence_y):

    dff = df[((df[fence_x] > geofence[0][0]) & # linke Grenze
                (df[fence_x] < geofence[1][0]) & # rechte Grenze
                (df[fence_y] > geofence[0][1]) & # untere Grenze
                (df[fence_y] < geofence[1][1]))] # obere Grenze

    dff = dff.sort_values(by='ts').reset_index()
    dff['diff_to_prev'] = dff['ts'].diff()


    cnt_maneuver = len(dff[dff['diff_to_prev'] > 80000].index) + 1

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

    cnt_maneuver = len(dff[dff['diff_to_prev'] > 80000].index) + 1

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

def calc_Morton(df, setup):

    df['accel_lon_mult'] = df['accel_lon'].add(setup.offset)
    df['accel_trans_mult'] = df['accel_trans'].add(setup.offset)
    df['accel_down_mult'] = df['accel_down'].add(setup.offset)


    df['accel_lon_mult'] = df.apply(lambda x: int(x['accel_lon_mult'] * setup.faktor_multiply), axis=1)
    df['accel_trans_mult'] = df.apply(lambda x: int(x['accel_trans_mult'] * setup.faktor_multiply), axis=1)
    df['accel_down_mult'] = df.apply(lambda x: int(x['accel_down_mult'] * setup.faktor_multiply), axis=1)

    m = morton.Morton(dimensions=setup.dim, bits=setup.bits)

    def set_value(row):
        return m.pack(int(row['accel_lon_mult']), int(row['accel_trans_mult'])) # normal
        #return m.pack(int(row['accel_trans_mult']), int(row['accel_lon_mult']))

    df['morton'] = df.apply(set_value, axis=1)

    return df

################################################################

def generate_ts(df):

    df['ts'] = (df['sampleTimeStamp.seconds'] * 1000000) + df['sampleTimeStamp.microseconds']
    df = df.drop(columns=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds'], axis=1)

    return df

################################################################

def remove_background_noise(df, setup, store, geofence):

    # vierte Reihe von unten
    #df = df.drop(df[(df.morton > 3140000) & (df.morton < 3170000)].index)
    # dritte Reihe von unten
    #df = df.drop(df[(df.morton > 2410000) & (df.morton < 2460000)].index)
    # zweite Reihe von unten
    #df = df.drop(df[(df.morton > 1720000) & (df.morton < 1770000)].index)
    # unterste Zeile
    #df = df.drop(df[(df.morton > 1025000) & (df.morton < 1070000)].index)

    #geofence_list = define_geofences(setup.geofence_resolution, [[-0.3, -0.25], [0.4, 0.25]])
    geofence_list = define_geofences(setup.geofence_resolution, geofence)

    search_mask = generate_Search_Mask(geofence_list, setup, store)

    df = scene_filter(df, search_mask)

    return df

################################################################

def maskMortonSpace(dff, min_M, max_M, duration_min, duration_max, setup):
    # fig, ax = plt.subplots(1)
    #
    # df_temp = dff.reset_index()
    # for i in range(0, len(df_temp)):
    #     point = Point(int(df_temp.at[i, 'ts']), int(df_temp.at[i, 'morton']))
    #     polygon = Polygon([(df_temp['ts'].min(), min_M), (df_temp['ts'].min(), max_M), (df_temp['ts'].max(), max_M), (df_temp['ts'].max(), min_M)])
    #     if polygon.contains(point):
    #         ax.scatter(point.x, point.y, color = 'red')
    # #   print("Point:" + str(polygon.contains(point)))
    #
    #
    # df_temp.plot.scatter(x='ts', y='morton', ax=ax, s=5)
    # ax.plot(*polygon.exterior.xy)
    #
    # plt.show()

    df = dff[(dff['morton'] > min_M) & (dff['morton'] < max_M)]
    df = remove_background_noise(df, setup)
    df = df.sort_values(by='ts').reset_index()
    df['diff_to_prev'] = df['ts'].diff()

    print(df)

    if len(df) > 0:
        start_ts = df['ts'][0]
        end_ts = 0

        maneuver_start_list = df[df['diff_to_prev'] > 80000].index #ein sample darf fehlen
        if not len(maneuver_start_list) > 0:
            maneuver_start_list.append(df.index[-1])

        for man_start in maneuver_start_list:
            end_ts = df.at['ts', man_start]
            if ((end_ts - start_ts) > min_M) & ((end_ts - start_ts) < max_M):
                print("linkskurve, dauer:", end_ts-start_ts)
            start_ts



################################################################

def generate_Search_Mask(geofence_list, setup, store):
    m = morton.Morton(dimensions=setup.dim, bits=setup.bits)
    time_filter_start = time.time()
    geofence_generate_parallel = []
    geofence_generate_parallel = []

    for geofence_temp in geofence_list:
        searchmask_name = str(setup.dim) + '/' + str(setup.bits) + '/' + str(setup.geofence_resolution) + '/' + str(setup.offset) + '/' + str(setup.faktor_multiply) + '/SearchMask_' + str(
            geofence_temp[0][0]) + '_' + str(geofence_temp[0][1]) + '_' + str(geofence_temp[1][0]) + '_' + str(geofence_temp[1][1])

        if not searchmask_name in store:
            if setup.generate_masks_multithread == True:
                geofence_generate_parallel.append(geofence_temp)
            else:
                print(searchmask_name + ' is not in store; Lets create it. This takes some time...')
                time_filter_start = time.time()
                search_mask_temp = transfer_Geofence_to_Morton(geofence_temp, m, setup.bits, 1, setup.offset, setup.faktor_multiply)
                search_mask = pd.concat([search_mask, search_mask_temp], axis = 0)
                time_filter_end = time.time()
                print("Time to transfer geofence in morton", round(time_filter_end - time_filter_start, 5), "s; containing",
                  len(search_mask.index), "values.")
                store[searchmask_name] = search_mask
    # generate Masks Multithread
    if len(geofence_generate_parallel) > 0:
        print("Generate missing Searchmasks...")
        mp_handler(geofence_generate_parallel, setup.dim, setup.bits, setup.geofence_resolution, m, setup.offset, setup.faktor_multiply)
    time_filter_end = time.time()


    search_mask = pd.DataFrame(columns=['morton'])
    print("Load Search Masks from Storage.")
    #load Masks
    for geofence_temp in geofence_list:
        searchmask_name = str(setup.dim) + '/' + str(setup.bits) + '/' + str(setup.geofence_resolution) + '/' + str(setup.offset) + '/' + str(setup.faktor_multiply) + '/SearchMask_' + str(geofence_temp[0][0]) + '_' + str(geofence_temp[0][1]) + '_' + str(geofence_temp[1][0]) + '_' + str(geofence_temp[1][1])

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


    if not math.isclose((geofence[0][0] % geofence_resolution), 0.0, abs_tol=0.001) &\
            math.isclose((geofence[0][1] % geofence_resolution), 0.0, abs_tol=0.001) &\
            math.isclose((geofence[1][0] % geofence_resolution), 0.0, abs_tol=0.001) &\
            math.isclose((geofence[1][1] % geofence_resolution), 0.0, abs_tol=0.001):
        sys.exit("GeofenceResolution is not common multiple of geofence values." + os.linesep +
                 str(geofence[0][0]) +": " + str((geofence[0][0] % geofence_resolution)) + os.linesep +
                 str(geofence[0][1]) +": " + str((geofence[0][1] % geofence_resolution)) + os.linesep +
                 str(geofence[1][0]) + ": " + str((geofence[1][0] % geofence_resolution)) + os.linesep +
                 str(geofence[1][1]) +": " + str((geofence[1][1] % geofence_resolution)))

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

def mp_handler(geofence_list, dim, bits, geofence_resolution, m, offset, faktor_multiply):
    p = multiprocessing.Pool(int(multiprocessing.cpu_count()))
    partial_call = functools.partial(mp_worker, m, bits, offset, faktor_multiply)

    #TODO: die Liste geofence_list teilen, sodass immer mal wieder zwischengespeichert wird
    for search_mask, geofence_temp in p.map(partial_call, geofence_list):
        searchmask_name = str(dim) + '/' + str(bits) + '/' + str(geofence_resolution) + '/' + str(offset) + '/' + str(faktor_multiply) + '/SearchMask_' + str(geofence_temp[0][0]) + '_' + str(geofence_temp[0][1]) + '_' + str(geofence_temp[1][0]) + '_' + str(geofence_temp[1][1])

        store[searchmask_name] = search_mask


################################################################

def scene_filter(df, search_mask):

    filter = df["morton"].isin(search_mask['morton'])

    return df[filter]

################################################################

def detect_single_maneuver_bf(df, driving_status, min_time_between_same_driving_status, setup):

    df_relevant_values = df

    df_relevant_values = df_relevant_values.drop(df_relevant_values[df_relevant_values.accel_lon < int((driving_status.fence[0][0] + setup.offset) * setup.faktor_multiply)].index)
    df_relevant_values = df_relevant_values.drop(df_relevant_values[df_relevant_values.accel_lon > int((driving_status.fence[1][0] + setup.offset) * setup.faktor_multiply)].index)
    df_relevant_values = df_relevant_values.drop(
        df_relevant_values[df_relevant_values.accel_trans < int((driving_status.fence[0][1] + setup.offset) * setup.faktor_multiply)].index)
    df_relevant_values = df_relevant_values.drop(
        df_relevant_values[df_relevant_values.accel_trans > int((driving_status.fence[1][1] + setup.offset) * setup.faktor_multiply)].index)

    # df_relevant_values = scene_filter(df, driving_status.search_mask)

    df_relevant_values = df_relevant_values.sort_values(by='ts').reset_index()
    df_relevant_values['diff_to_prev'] = df_relevant_values['ts'].diff()

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000


    maneuver_start_list = list(df_relevant_values[df_relevant_values['diff_to_prev'] > min_time_between_same_driving_status].index)
    maneuver_list = []

    if len(df_relevant_values) > 0:
        maneuver_start_list.append(df_relevant_values.index[-1])

        maneuver_start_idx = 0
        maneuver_start_ts = df_relevant_values.loc[maneuver_start_idx, 'ts']


        for maneuver_idx in maneuver_start_list:

            maneuver_end_ts = df_relevant_values.loc[maneuver_idx-1, 'ts']

            if ((maneuver_end_ts - maneuver_start_ts) > driving_status.min_time) & ((maneuver_end_ts - maneuver_start_ts) < driving_status.max_time):

                maneuver_list.append([maneuver_start_ts, maneuver_end_ts])

                # start_date = datetime.datetime.fromtimestamp(maneuver_start_ts/1000000, datetime.timezone(datetime.timedelta(hours=1)))
                # end_date = datetime.datetime.fromtimestamp(maneuver_end_ts / 1000000,datetime.timezone(datetime.timedelta(hours=1)))
                # print("Maneuver detected! Start:",start_date , " End:", end_date)

            maneuver_start_idx = maneuver_idx
            maneuver_start_ts = df_relevant_values.loc[maneuver_start_idx, 'ts']

    return maneuver_list

################################################################

def detect_single_maneuver(df, driving_status, min_time_between_same_driving_status):

    df_relevant_values = scene_filter(df, driving_status.search_mask)

    df_relevant_values = df_relevant_values.sort_values(by='ts').reset_index()
    df_relevant_values['diff_to_prev'] = df_relevant_values['ts'].diff()

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000


    maneuver_start_list = list(df_relevant_values[df_relevant_values['diff_to_prev'] > min_time_between_same_driving_status].index)
    maneuver_list = []

    if len(df_relevant_values) > 0:
        maneuver_start_list.append(df_relevant_values.index[-1])

        maneuver_start_idx = 0
        maneuver_start_ts = df_relevant_values.loc[maneuver_start_idx, 'ts']


        for maneuver_idx in maneuver_start_list:

            maneuver_end_ts = df_relevant_values.loc[maneuver_idx-1, 'ts']

            if ((maneuver_end_ts - maneuver_start_ts) > driving_status.min_time) & ((maneuver_end_ts - maneuver_start_ts) < driving_status.max_time):

                maneuver_list.append([maneuver_start_ts, maneuver_end_ts])

                # start_date = datetime.datetime.fromtimestamp(maneuver_start_ts/1000000, datetime.timezone(datetime.timedelta(hours=1)))
                # end_date = datetime.datetime.fromtimestamp(maneuver_end_ts / 1000000,datetime.timezone(datetime.timedelta(hours=1)))
                # print("Maneuver detected! Start:",start_date , " End:", end_date)

            maneuver_start_idx = maneuver_idx
            maneuver_start_ts = df_relevant_values.loc[maneuver_start_idx, 'ts']

    return maneuver_list

################################################################

def detect_maneuver_combination(df, maneuver_obj, min_time_between_same_driving_status, morton, setup):

    for status_obj in maneuver_obj.drivingStatus_list:
        status_obj.relevant_values_list = scene_filter(df, status_obj.search_mask)
        if morton == True:
            status_obj.maneuver_list = detect_single_maneuver(df, status_obj, min_time_between_same_driving_status)
        else:
            status_obj.maneuver_list = detect_single_maneuver_bf(df, status_obj, min_time_between_same_driving_status, setup)
        #print("relevant value list", status_obj.relevant_values_list)
        #print("driving_status_list", status_obj.maneuver_list)

    driving_maneuver_list = []

    status_prev_obj = maneuver_obj.drivingStatus_list[0]

    cnt_obj = len(maneuver_obj.drivingStatus_list)

    # for status_obj in maneuver_obj_list:
    #     if not status_obj == status_prev_obj:
    #         for status_prev in status_prev_obj.maneuver_list:
    #             for status_curr in status_obj.maneuver_list:
    #                 gap = status_curr[0] - status_prev[1]
    #                 if (gap > status_prev_obj.min_gap) & (gap < status_prev_obj.max_gap):
    #                     print("Maneuver detected: ", status_prev_obj.name, "-", status_obj.name
    #                     , " :", status_prev[0], "bis", status_curr[1])
    #                     driving_maneuver_list.append([[status_prev[0], status_curr[1]]])
    #         status_prev_obj = status_obj

    relevant_values_list = []

    for i in range(0,len(maneuver_obj.drivingStatus_list[0].maneuver_list)-1):
        time_end, end = recursive_Maneuver_Search(0,i,maneuver_obj.drivingStatus_list)
        if end == True:
            time_start = maneuver_obj.drivingStatus_list[0].maneuver_list[i][0]
            #print(maneuver_obj.name, "detected from", time_start, "to", time_end, "; Duration:", time_end-time_start/1000000, "s")
            driving_maneuver_list.append([[time_start, time_end]])
            relevant_values_list.append(df[(df['ts'] > time_start) & (df['ts'] < time_end)])

    #driving_maneuver_list = [maneuver_obj.drivingStatus_list[0].maneuver_list]

    driving_status_list = []

    for cur_obj in maneuver_obj.drivingStatus_list:
        driving_status_list.append(cur_obj.maneuver_list)
        #relevant_values_list.append(cur_obj.relevant_values_list)


    # status_prev_obj = maneuver_obj_list[0]
    # for status_cur_obj in maneuver_obj_list:
    #     if not status_cur_obj == status_prev_obj:
    #         recursive_Maneuver_Search(status_prev_obj, status_cur_obj)

    return driving_maneuver_list, driving_status_list, relevant_values_list

################################################################

def recursive_Maneuver_Search(man_idx, status_idx, maneuver_obj_list):

    if man_idx >= len(maneuver_obj_list):
        return 0, False

    # if not (status_idx < len(cur_man_obj.maneuver_list)):
    #    return 0, False

    cur_man_obj = maneuver_obj_list[man_idx]

    # TODO: maneuverobj lists mit nur einem element
    if len(maneuver_obj_list) == 1:
        return cur_man_obj.maneuver_list[status_idx], True

    next_man_obj = maneuver_obj_list[man_idx + 1]

    cur_status = cur_man_obj.maneuver_list[status_idx]

    for next_status in next_man_obj.maneuver_list:

        gap = next_status[0] - cur_status[1]

        if gap > cur_man_obj.max_gap:
            break

        if (gap > cur_man_obj.min_gap) & (gap < cur_man_obj.max_gap):

            if man_idx == (len(maneuver_obj_list) - 2):
                return next_status[1], True

            else:
                next_status_idx = next_man_obj.maneuver_list.index(next_status)

                return recursive_Maneuver_Search(man_idx + 1, next_status_idx, maneuver_obj_list)

    return 0, False


################################################################

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("Välkommen!")

    setup = Setup(True, 10, 100, 0.25, 10, 18, 2) # resolution: 0.0625
    min_time_between_same_driving_status = 160000

    store = pd.HDFStore("searchMaskStorage.h5")

    ################################################################

    print("Load_Database...")
    #df = pd.read_csv('../Data/Ausschnitte/opendlv.device.gps.pos.Grp1Data-0.csv', sep=';',
    #                  usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
    #                           'accel_lon', 'accel_trans', 'accel_down'])
    # df = pd.read_csv('../Data/Ausschnitte/Hard_Braking/braking_cut_8_brakes.csv', sep=';',
    #                  usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
    #                           'accel_lon', 'accel_trans', 'accel_down'])
    #df = pd.read_csv('../Data/Ausschnitte/LaneChange/lanechange_20220921_fast_right.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    #df = pd.read_csv('../Data/Ausschnitte/LaneChange/LC_in_Kurve.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    #df = pd.read_csv('../Data/Ausschnitte/Noise/noise_complete.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    # df = pd.read_csv('../Data/Messfahrten/Lindholmen_2/opendlv.device.gps.pos.Grp1Data-0.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    #df = pd.read_csv('../Data/Messfahrten/20220921/CSV/Motorway_Roundabout/opendlv.device.gps.pos.Grp1Data-0.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    #df = pd.read_csv('../Data/Ausschnitte/Kreisfahrt/Kreis_rechts_im_Uhrzeigen_const_slow.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    #df = pd.read_csv('../Data/Ausschnitte/Kreisfahrt/Kreis_links_gegen_Uhrzeigen.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    #df = pd.read_csv('../Data/Ausschnitte/S_Kurve/S_Kurve_aus_rechtsskurve_kommend.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])

    # Datacollection 2022 09 23
    # df = pd.read_csv('../Data/Messfahrten/20220923/CSV/20220923_to_Boras/opendlv.device.gps.pos.Grp1Data-0_motorway_to_boras.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    # label = pd.read_csv('../Data/Messfahrten/20220923/CSV/20220923_to_Boras/opendlv.system.LogMessage-999.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'description'])
    df = pd.read_csv('../Data/Messfahrten/20220923/CSV/20220923_back_to_gothenburg/opendlv.device.gps.pos.Grp1Data-0.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])
    label = pd.read_csv('../Data/Messfahrten/20220923/CSV/20220923_back_to_gothenburg/opendlv.system.LogMessage-999.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'description'])

    #noise
    #df = pd.read_csv('../Data/Ausschnitte/Noise/noise_straight_complete.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed', 'accel_lon', 'accel_trans', 'accel_down'])

    #Voyager
    #df = pd.read_csv('C:/Users/Lukas Birkemeyer/Documents/Confidential_Promotion/Voyager/17T114115Z/opendlv.proxy.AccelerationReading-2.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'accelerationX', 'accelerationY', 'accelerationZ'])
    #df.rename(columns={'accelerationX': 'accel_lon', 'accelerationY': 'accel_trans', 'accelerationZ': 'accel_down'}, inplace=True)


    #label = pd.read_csv('../Data/Messfahrten/20220921/CSV/Testgelaende/Testgelaende_noise_LC/opendlv.system.LogMessage-999.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'description'])
    #label = pd.read_csv('../Data/Messfahrten/20220921/CSV/Testgelaende/Kreisfahrt/opendlv.system.LogMessage-999.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'description'])

    #label = pd.read_csv('../Data/Messfahrten/20220921/CSV/Motorway_Roundabout/opendlv.system.LogMessage-999.csv', sep=';', usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds','description'])
    label = generate_ts(label)
    label['label'] = label['description'].apply(lambda b: str(base64.b64decode(b)))
    label = label[label['label'].str.contains("right") == True]
    label = label.drop(columns='description')

    print("The length of data is:", len(df))
    ################################################################

    df = generate_ts(df)

    df = precondition_data(df, setup.max_accel)

    df = calc_Morton(df=df, setup=setup)

    #df = remove_background_noise(df, setup, store, [[-0.25, -0.25], [0.375, 0.25]])
    #dff_temp = maskMortonSpace(dff, 2470000, 2580000, 2000000, 5000000, setup)

    ################################################################
    # Maneuver definition
    # LaneChange on straight road
    rechtsKurve_LC_gerade = DrivingStatus(name='Kurve rechts', fence=[[-1.5, -5], [0.75, -0.75]], min_time=100000, max_time=30000000,
                                min_gap=-200000, max_gap=2000000, setup=setup, color='red')
    linksKurve_LC_gerade = DrivingStatus(name='Kurve links', fence=[[-1.5, 0.75], [0.75, 5]], min_time=100000, max_time=30000000,
                                min_gap=-200000, max_gap=2000000, setup=setup, color='red')

    maneuver_LC_to_left_straight_list = []
    maneuver_LC_to_left_straight_list.append(linksKurve_LC_gerade)
    maneuver_LC_to_left_straight_list.append(rechtsKurve_LC_gerade)
    maneuver_LC_to_left_straight = Maneuver('LC_to_left_straight', maneuver_LC_to_left_straight_list)

    maneuver_LC_to_right_straight_list = []
    maneuver_LC_to_right_straight_list.append(rechtsKurve_LC_gerade)
    maneuver_LC_to_right_straight_list.append(linksKurve_LC_gerade)
    maneuver_LC_to_right_straight = Maneuver('LC_to_right_straight', maneuver_LC_to_right_straight_list)

    # ################################################################
    # # Maneuver definition
    # # LaneChange in left curve road
    #
    # linksKurve_vor_LC_to_left_linkskurve = DrivingStatus(name='Kurve links', fence=[[-1, 0.5], [1, 1.5]], min_time=2000000,
    #                                       max_time=300000000,
    #                                       min_gap=-200000, max_gap=800000, setup=setup, color='red')
    # ausscheren_LC_to_left_linkskurve = DrivingStatus(name='Ausscheren links', fence=[[-1, 1.5], [1, 4]], min_time=100000,
    #                                       max_time=5000000,
    #                                       min_gap=-200000, max_gap=800000, setup=setup, color='red')
    # abfangen_LC_to_left_linkskurve = DrivingStatus(name='Abfangen rechts', fence=[[-1, -1.5], [1, 0.5]], min_time=100000,
    #                                      max_time=5000000,
    #                                      min_gap=-200000, max_gap=800000, setup=setup, color='red')
    # #startBremsen = DrivingStatus(name='Kurve links', fence=[[3, -4], [10, 4]], min_time=500000, max_time=30000000,
    # #                           min_gap=0, max_gap=0, setup=setup, color='green')
    #
    # maneuver_LC_to_left_left_curve_list = []
    # maneuver_LC_to_left_left_curve_list.append(linksKurve_vor_LC_to_left_linkskurve)
    # maneuver_LC_to_left_left_curve_list.append(ausscheren_LC_to_left_linkskurve)
    # maneuver_LC_to_left_left_curve_list.append(abfangen_LC_to_left_linkskurve)
    # maneuver_LC_to_left_left_curve = Maneuver('LC_to_left_left_curve', maneuver_LC_to_left_left_curve_list)
    #
    # ################################################################
    # # Maneuver definition
    # # LaneChange in left curve road
    #
    # rechtsKurve_vor_LC_to_right_linkskurve = DrivingStatus(name='Kurve links', fence=[[-1, -1.5], [1, -0.5]],
    #                                                      min_time=2000000,
    #                                                      max_time=300000000,
    #                                                      min_gap=-200000, max_gap=800000, setup=setup, color='red')
    # ausscheren_LC_to_right_rechtskurve = DrivingStatus(name='Ausscheren links', fence=[[-1, -4], [1, -1.5]],
    #                                                  min_time=100000,
    #                                                  max_time=5000000,
    #                                                  min_gap=-200000, max_gap=800000, setup=setup, color='red')
    # abfangen_LC_to_right_rechtskurve = DrivingStatus(name='Abfangen rechts', fence=[[-1, -0.5], [1, 1.5]],
    #                                                min_time=100000,
    #                                                max_time=5000000,
    #                                                min_gap=-200000, max_gap=800000, setup=setup, color='green')
    # # startBremsen = DrivingStatus(name='Kurve links', fence=[[3, -4], [10, 4]], min_time=500000, max_time=30000000,
    # #                           min_gap=0, max_gap=0, setup=setup, color='green')
    #
    # maneuver_LC_to_right_right_curve_list = []
    # maneuver_LC_to_right_right_curve_list.append(rechtsKurve_vor_LC_to_right_linkskurve)
    # maneuver_LC_to_right_right_curve_list.append(ausscheren_LC_to_right_rechtskurve)
    # maneuver_LC_to_right_right_curve_list.append(abfangen_LC_to_right_rechtskurve)
    # maneuver_LC_to_right_right_curve = Maneuver('LC_to_right_right_curve', maneuver_LC_to_right_right_curve_list)

    # ################################################################
    # # Maneuver definition
    # # AEB
    # Stop_1 = DrivingStatus(name='AEB1', fence=[[4.5, -4], [10, 4]], min_time=100000,
    #                                       max_time=30000000,
    #                                       min_gap=-30000000, max_gap=2000000, setup=setup, color='red')
    # Stop_2 = DrivingStatus(name='AEB2', fence=[[4.5, -4], [10, 4]], min_time=100000,
    #                                      max_time=30000000,
    #                                      min_gap=-30000000, max_gap=2000000, setup=setup, color='red')
    #
    # maneuver_AEB_list = []
    # maneuver_AEB_list.append(Stop_1)
    # maneuver_AEB_list.append(Stop_2)
    # maneuver_AEB = Maneuver('AEB', maneuver_AEB_list)

    #print("Define Search Mask.")
    #search_mask = generate_Search_Mask(geofence_list, setup, store)

    ################################################################


    #driving_status_list.append(detect_single_maneuver(df, startBremsen))
    # print(driving_status_list)

    number_itteration = 100
    print("Maneuver Detection Vektor, Morton.")
    time_filter_start = time.time()
    for i in tqdm(range(number_itteration)):
        driving_maneuver_list, driving_status_list, relevant_values_list = detect_maneuver_combination(df, maneuver_LC_to_left_straight, min_time_between_same_driving_status, True, setup)
    time_filter_end = time.time()
    print("Done: Time to detect",number_itteration,"maneuvers with morton index:", round(time_filter_end-time_filter_start, 10), "s")

    print("Maneuver Detection Vektor, ts.")
    time_filter_start = time.time()
    for i in tqdm(range(number_itteration)):
        driving_maneuver_list, driving_status_list, relevant_values_list = detect_maneuver_combination(df, maneuver_LC_to_left_straight, min_time_between_same_driving_status, False, setup)
    time_filter_end = time.time()
    print("Done: Time to detect",number_itteration,"maneuvers with ts index:", round(time_filter_end - time_filter_start, 10), "s")


    ################################################################
    plot_Values(df, relevant_values_list, driving_maneuver_list, label)

    # print(driving_maneuver_list)

    fig, ax = plt.subplots(2, gridspec_kw = {'height_ratios': [1, 7]})

    df.plot(x='ts', y=['accel_lon', 'accel_trans'], ax=ax[0])

    for driving_status_temp in driving_maneuver_list:
        color = 'red'
        for status in driving_status_temp:
            ax[0].add_patch(
                Rectangle((status[0], -4), status[1] - status[0], 8, fill=False, color='red', lw=1.5))

    for idx, row in label.iterrows():
        ax[0].annotate(row['label'], (row['ts'], 2.5), rotation=60)

    df.plot.scatter(x='ts', y='morton', color='blue', ax=ax[1], s=15)
    #dff.plot.scatter(x='ts', y='morton', color='blue', ax=ax[1], s=11)
    for relevant in relevant_values_list:
        relevant.plot.scatter(x='ts', y='morton', color='red', ax=ax[1], s=5)

    plt.show()

    store.close()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
