# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import morton
import sys


def plot_Values(dff, geofence):
    fig, ax = plt.subplots(4, gridspec_kw={'height_ratios': [3, 3, 3, 1]})

    dff.plot(x='lon', y='lat', ax=ax[0])
    dff.plot(x='ts', y=['accel_lon', 'accel_trans', 'accel_down'], ax=ax[1])

    dff.plot(kind='scatter', x='accel_lon', y='accel_trans', color=dff['ts'], ax=ax[2])
    ax[2].add_patch(Rectangle((geofence[0][0], geofence[0][1]), geofence[1][0]-geofence[0][0],
                              geofence[1][1] - geofence[0][1], fill=False, color='red', lw=2))

    min = dff['morton'].min()
    max = dff['morton'].max()
    # min = 12000000000
    # max = 18000000000
    min = 0
    max = 30000000000

    ax[3].hist(dff['morton'], bins=400)
    ax[3].set_xlim(min, max)
    ax[3].set_ylim(0, 1)

    # fig.tight_layout()
    plt.show()
################################################################

def filter_Values(dff):

    ts = 1646666599000000
    off = 204
    ts = ts + (off * 1000000)
    # dff = df[(df['ts'] > ts) & (df['ts'] < ts + 4500000)]  # 10000000
    # dff = df[(df['ts'] > 1646666563800000) & (df['ts'] < 1646666564800000)]

    # dff = df[(df['accel_lon'] > 140000)]
    # dff = df[(df['morton'] > 26776019010)]
    dff = df

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
    df = pd.read_csv('../Data/Ausschnitte/Hard_Braking/braking_cut.csv', sep=';',
                     usecols=['sampleTimeStamp.seconds', 'sampleTimeStamp.microseconds', 'lat', 'lon', 'speed',
                              'accel_lon', 'accel_trans', 'accel_down'])
    # df.rename(columns = {'timestamp:10881:<lon>':'ts', 'accel_lon:10881:<double>':'accel_lon', 'accel_trans:10881:<double>':'accel_trans', 'accel_down:10881:<double>':'accel_down'}, inplace = True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    ################################################################

    geofence = [[4.2, -1], [5.8, 1]]

    ################################################################

    df = generate_ts(df)

    ################################################################

    calc_Morton(df=df, dimension=2, bits=18)

    ################################################################

    dff = filter_Values(df)

    ################################################################
    plot_Values(dff, geofence)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
