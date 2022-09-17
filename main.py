# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import morton
import datetime
import sys
import time
from rich.console import Console

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

def search_Morton(geofence, df_array, curve, m, resolution):
    offset = 10
    faktor_multiply = 10000

    A = geofence[0]
    A[0] = int((A[0] + offset) * faktor_multiply)
    A[1] = int((A[1] + offset) * faktor_multiply)
    C = geofence[1]
    C[0] = int((C[0] + offset) * faktor_multiply)
    C[1] = int((C[1] + offset) * faktor_multiply)
    B = [A[0], C[1]]
    D = [A[1], C[0]]

    search_space = [m.pack(A[0], A[1]), m.pack(C[0], C[1])]

    #ax.add_patch(Rectangle((A[0]-0.25, A[1]-0.25), C[0]-A[0]+0.5, C[1]-A[1]+0.5, fill=False, color='red', lw = 2))

    #df_array[(df_array.morton >= search_space[0]) & (df_array.morton <= search_space[1])].sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax, label="SearchSpace")

    search_df = df_array[(df_array.morton >= search_space[0]) & (df_array.morton <= search_space[1])]

    min = 0
    max = (2**resolution)-1

    console = Console()
    with console.status("[bold green] Transform Geofence...") as status:
        search_df = identifyNonRelvantAreas(m, geofence, search_df, min, min, max, max)

    #search_df.sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax, label="SearchSpace")

    geofence_area = (C[0] - A[0] + 1) * (C[1] - A[1] + 1)
    search_area = len(search_df.axes[0])
    precision = geofence_area / (geofence_area + (search_area - geofence_area))

    print("Search space for geofence:", geofence, "requires search between", search_space[0], "and", search_space[1], "requires", search_area, "queries to search ", geofence_area, "entries." )
    #print("Precision of", round(precision, 3))

    return search_df

################################################################

def identifyNonRelvantAreas(m, geofence, search_df, min_value_x, min_value_y, max_value_x, max_value_y):

    # print("identification of non relevant areas", min_value_x, min_value_y, ";", max_value_x, max_value_y, "Search_df has", len(search_df.index), "lines.")

    if (m.pack(max_value_x, max_value_y) - m.pack(min_value_x, min_value_y)) <=3:
        return search_df

    A = geofence[0]
    C = geofence[1]

    half_value_x = int(((max_value_x - min_value_x) / 2) + 0.5 + min_value_x)
    half_value_y = int(((max_value_y - min_value_y) / 2) + 0.5 + min_value_y)

    # search_df = df_array[['x', 'y', 'morton']]

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
        search_df = search_df.drop(search_df[(search_df.morton < Q1_range[1]+1) & (search_df.morton > Q1_range[0])].index)
    else:
        search_df = identifyNonRelvantAreas(m, geofence, search_df, min_value_x=min_value_x, min_value_y=min_value_y, max_value_x=half_value_x-1, max_value_y=half_value_y-1)
    if Q2 == False:
        search_df = search_df.drop(search_df[(search_df.morton < Q2_range[1] + 1) & (search_df.morton > Q2_range[0])].index)
    else:
        search_df = identifyNonRelvantAreas(m, geofence, search_df, min_value_x=half_value_x, min_value_y=min_value_y,
                                            max_value_x=max_value_x, max_value_y=half_value_y - 1)
    if Q3 == False:
        search_df = search_df.drop(search_df[(search_df.morton < Q3_range[1] + 1) & (search_df.morton > Q3_range[0])].index)
    else:
        search_df = identifyNonRelvantAreas(m, geofence, search_df, min_value_x=min_value_x, min_value_y=half_value_y,
                                            max_value_x=half_value_x - 1, max_value_y=max_value_y)
    if Q4 == False:
        search_df = search_df.drop(search_df[(search_df.morton < Q4_range[1] + 1) & (search_df.morton > Q4_range[0])].index)
    else:
        search_df = identifyNonRelvantAreas(m, geofence, search_df, min_value_x=half_value_x, min_value_y=half_value_y,
                                            max_value_x=max_value_x, max_value_y=max_value_y)

    return search_df

################################################################

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

#    df = pd.read_csv('../Data/Ausschnitte/opendlv.device.gps.pos.Grp1Data-0.csv', sep=';',
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
    df, m = calc_Morton(df=df, dimension=2, bits=bits)

    ################################################################

    time_filter_start = time.time()

    #geofence = [[0.5, -4], [2, -1]]
    geofence = [[0.0, 0.0], [0.1, 0.1]]
    #geofence = [[0.1, -1], [0.2, 0.1]]
    fence_x = 'accel_lon'
    fence_y = 'accel_trans'

    # dff = filter_Values(df, geofence, fence_x=fence_x, fence_y=fence_y)
    dff = filter_Morton(df, 25000000000, 30000000000) # stark Bremsen
    # dff = filter_Morton(df, 10000000000, 13000000000) # starke Beschleunigung
    # dff = filter_Morton(df, 14000000000, 14800000000) # linkskurve Bremsen

    time_filter_end = time.time()
    print("Time to set geofence and filter with threshold value", len(df.index), "rows:", round(time_filter_end-time_filter_start, 5), "s")
    ################################################################

    time_filter_start = time.time()
    search_df = search_Morton(geofence, df, 'morton', m, bits)
    time_filter_end = time.time()

    print("Time to calculate search_df from geofence", round(time_filter_end-time_filter_start, 5), "s; containing", len(search_df.index), "values.")

    # df_relevant_values = df.drop(df[(df.morton < Q1_range[1]+1) & (search_df.morton > Q1_range[0])].index)

    ################################################################
    plot_Values(df, search_df, geofence)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
