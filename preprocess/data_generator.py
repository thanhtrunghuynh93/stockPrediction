import pandas as pd
import numpy as np
import tensorflow as tf

from utils.date_util import get_previous_trade_day, get_most_recent_trade_day


# Extracting trading data from the existing data source dataframe
def generate_trading_data(source_df, feature_cols, target_col, history_window, start_date, end_date, setting = "many_to_many"):

    if start_date not in source_df.index:
        raise Exception("Error: start date not in source dataframe: {} vs {}".format(source_df.index[0], start_date))
    idx = source_df.index.get_loc(start_date)
    if idx < history_window:
        raise Exception("Error: not sufficient history window in source dataframe: {} vs {}".format(source_df.index[0], start_date))
    needed_start_date = source_df.index[idx - history_window]

    df_X = source_df.copy()
    df_X = df_X[df_X.index >= needed_start_date]
    df_X = df_X[df_X.index <= end_date]
    df_y = df_X[target_col]
    train_X_data = df_X.filter((feature_cols)).values
    train_y_data = df_y.values


    X_train = np.array([train_X_data[i : i + history_window] for i in range(len(train_X_data) - history_window)])

    if setting == "many_to_many":
        y_train = np.array([train_y_data[i + 1: i + history_window + 1] for i in range(len(train_y_data) - history_window)])
    else:
        y_train = np.array([train_y_data[i + history_window] for i in range(len(train_y_data) - history_window)])

    df = source_df.copy()
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]
    
    return X_train, y_train, df

# Extracting trading data from the existing data source dataframe
def generate_trading_data_by_date(df, feature_col, history_window = 20, start_date = "2021-01-01"):

    id = df[df.index >= start_date].iloc[0].name
    idx = df.index.get_loc(id)
    
    if idx < history_window:
        print("Error: the history is insufficient for getting the window")
        return None

    start_idx = idx - history_window

    df_X = df.filter((feature_col))    
    df = df[id:].copy()

    train_X_data = df_X[start_idx:].values
    
    # Split the data into x_train and y_train data sets
    X_train = []
    X_train = np.array([train_X_data[i : i + history_window] for i in range(len(train_X_data) - history_window)])
        
    return X_train, df

#Split train test set by split date
def split_train_test_by_split_date(df, feature_col, target_col, history_window = 20, split_date = "2021-01-01"):

    id = df[df.index >= split_date].iloc[0].name
    idx = df.index.get_loc(id)

    start_idx = idx - history_window

    df_X = df.filter((feature_col))
    df_y = df[target_col]

    train_X_data = df_X[:idx].values
    test_X_data = df_X[start_idx:].values

    train_y_data = df_y[:idx].values
    test_y_data = df_y[start_idx:].values

    # Split the data into x_train and y_train data sets
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    
    X_train = np.array([train_X_data[i : i + history_window] for i in range(len(train_X_data) - history_window)])
    y_train = np.array([train_y_data[i + history_window] for i in range(len(train_y_data) - history_window)])
    X_test = np.array([test_X_data[i : i + history_window] for i in range(len(test_X_data) - history_window)])
    y_test = np.array([test_y_data[i + history_window] for i in range(len(test_y_data) - history_window)])
    
    y_train_vec = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_test_vec = tf.keras.utils.to_categorical(y_test, num_classes=3)

    
    return X_train, y_train, y_train_vec, X_test, y_test, y_test_vec


def split_train_test_by_ratio(df, feature_col, target_col, history_window = 20, train_ratio = 0.9):

    df_X = df.filter((feature_col))
    df_y = df[target_col]
    
    # Create the training data set 
    training_data_len = int(np.ceil(len(df) * train_ratio))


    train_X_data = df_X[:training_data_len].values
    test_X_data = df_X[training_data_len:].values

    train_y_data = df_y[:training_data_len].values
    test_y_data = df_y[training_data_len:].values
    
    
    # Split the data into x_train and y_train data sets
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    X_train = np.array([train_X_data[i : i + history_window] for i in range(len(train_X_data) - history_window)])
    y_train = np.array([train_y_data[i + history_window] for i in range(len(train_y_data) - history_window)])
    X_test = np.array([test_X_data[i : i + history_window] for i in range(len(test_X_data) - history_window)])
    y_test = np.array([test_y_data[i + history_window] for i in range(len(test_y_data) - history_window)])
    
    y_train_vec = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_test_vec = tf.keras.utils.to_categorical(y_test, num_classes=3)
   
    return X_train, y_train, y_train_vec, X_test, y_test, y_test_vec


