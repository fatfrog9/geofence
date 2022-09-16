# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import morton
import datetime
import sys


def plot_Values(df, dff, geofence):
    fig, ax = plt.subplots(4, gridspec_kw={'height_ratios': [3, 3, 3, 1]})

    df.plot(x='lon', y='lat', ax=ax[0], color = 'blue')
    dff.plot.scatter(x='lon', y='lat', ax=ax[0], color = 'red')

    df.plot(x='ts', y=['accel_lon', 'accel_trans', 'accel_down'], ax=ax[1])
    accelBoxes = True
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

    df.plot(kind='scatter', x='accel_lon', y='accel_trans', color=df['ts'], ax=ax[2])
    ax[2].add_patch(Rectangle((geofence[0][0], geofence[0][1]), geofence[1][0] - geofence[0][0],
                              geofence[1][1] - geofence[0][1], fill=False, color='red', lw=2))

    min = df['morton'].min()
    max = df['morton'].max()
    # min = 12000000000
    # max = 18000000000
    min = 0
    max = 30000000000

    ax[3].hist(df['morton'], bins=400, color='blue')
    ax[3].hist(dff['morton'], bins=400, color='red')
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('../Data/Ausschnitte/Hard_Braking/braking_cut_8_brakes.csv', sep=';',
                     usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
                              'accel_lon', 'accel_trans', 'accel_down'])
    # df.rename(columns = {'timestamp:10881:<lon>':'ts', 'accel_lon:10881:<double>':'accel_lon', 'accel_trans:10881:<double>':'accel_trans', 'accel_down:10881:<double>':'accel_down'}, inplace = True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    ################################################################

    geofence = [[4.2, -3], [10, 3]]
    fence_x = 'accel_lon'
    fence_y = 'accel_trans'

    ################################################################

    df = generate_ts(df)

    ################################################################

    calc_Morton(df=df, dimension=2, bits=18)

    ################################################################

    dff = filter_Values(df, geofence, fence_x=fence_x, fence_y=fence_y)

    ################################################################
    plot_Values(df, dff, geofence)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
