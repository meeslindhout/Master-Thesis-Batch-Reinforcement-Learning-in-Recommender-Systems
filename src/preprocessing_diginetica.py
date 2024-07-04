'''
Code has been sourced from the following repository:
Malte. (2024). Rn5l/session-rec [Python]. https://github.com/rn5l/session-rec (Original work published 2019)
(https://github.com/rn5l/session-rec/blob/master/preprocessing/session_based/preprocess_diginetica.py)
'''

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# data config (all methods)
DATA_PATH = '../data/diginetica/raw/'
DATA_PATH_PROCESSED = '../data/diginetica/prepared_test/'
# DATA_FILE = 'yoochoose-clicks-10M'
# DATA_FILE = 'train-clicks'
# MAP_FILE = 'train-queries'
# MAP_FILE2 = 'train-item-views'
DATA_FILE = 'train-item-views'

# COLS=[0,1,2]
COLS = [0, 2, 3, 4]

# filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# min date config
MIN_DATE = '2016-05-07'

# days test default config
DAYS_TEST = 7

# slicing default config
NUM_SLICES = 5
DAYS_OFFSET = 45
DAYS_SHIFT = 18
DAYS_TRAIN = 25
DAYS_TEST = 7

# retraining default config
DAYS_RETRAIN = 1


def process_data(data):
    data['Time'] = data.Time.fillna(0).astype(np.int64)
    # convert time string to timestamp and remove the original column
    # start = datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
    data['Date'] = data.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data['Datestamp'] = data['Date'].apply(lambda x: x.timestamp())
    data['Time'] = (data['Time'] / 1000)
    data['Time'] = data['Time'] + data['Datestamp']
    data['TimeO'] = data.Time.apply(lambda x: datetime.fromtimestamp(x, timezone.utc))
    
    # drop 

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data = data.groupby('SessionId').apply(lambda x: x.sort_values('Time'))     # data = data.sort_values(['SessionId'],['Time'])
    data.index = data.index.get_level_values(1)
    return data
	
def filter_data(data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH):
    # filter session length
    session_lengths = data.groupby('SessionId').size()
    session_lengths = session_lengths[ session_lengths >= min_session_length ]
    data = data[np.in1d(data.SessionId, session_lengths.index)]

    # filter item support
    data['ItemSupport'] = data.groupby('ItemId')['ItemId'].transform('count')
    data = data[data.ItemSupport >= min_item_support]

    # filter session length again, after filtering items
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    
    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set default \n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data

def split_data(data, days_test=DAYS_TEST):
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)

    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <= test_from.timestamp()].index
    session_test = session_max_times[session_max_times > test_from.timestamp()].index
    train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    
    return train, test