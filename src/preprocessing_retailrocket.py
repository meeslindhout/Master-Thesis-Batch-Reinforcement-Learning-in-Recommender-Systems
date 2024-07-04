'''
Code has been sourced from the following repository:
Malte. (2024). Rn5l/session-rec [Python]. https://github.com/rn5l/session-rec (Original work published 2019)
(https://github.com/rn5l/session-rec/blob/master/preprocessing/session_based/preprocess_retailrocket.py)
'''
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

#data config (all methods)
DATA_PATH = '../data/retailrocket/raw/'
DATA_PATH_PROCESSED = '../data/retailrocket/prepared/'
DATA_FILE = 'events'
SESSION_LENGTH = 30 * 60 #30 minutes

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#min date config
MIN_DATE = '2014-04-01'

#days test default config 
DAYS_TEST = 1

#slicing default config
NUM_SLICES = 10
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 9
DAYS_TEST = 1


def load_data( file ) : 
    
    #load csv
    data = pd.read_csv( file+'.csv', sep=',', header=0, usecols=[0,1,2,3], dtype={0:np.int64, 1:np.int32, 2:str, 3:np.int32})
    #specify header names
    data.columns = ['Time','UserId','Type','ItemId']
    data['Time'] = (data.Time / 1000).astype( int )
    
    #sessionize
    data.sort_values(by=['UserId', 'Time'], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data['Time'].values)
    # check which of them are bigger then session_th
    split_session = tdiff > SESSION_LENGTH
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data['UserId'].values[1:] != data['UserId'].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data['SessionId'] = session_ids
    
    data.sort_values( ['SessionId','Time'], ascending=True, inplace=True )
    
    cart = data[data.Type == 'addtocart']
    data = data[data.Type == 'view']
    del data['Type']
    
    print(data)
    
    #output
    
    print( data.Time.min() )
    print( data.Time.max() )
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    del data['TimeTmp']
    
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data, cart


def split_data( data, output_file, days_test ) :
    
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    test_from = data_end - timedelta( days_test )
    
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[ session_max_times < test_from.timestamp() ].index
    session_test = session_max_times[ session_max_times >= test_from.timestamp() ].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    