import os
import pandas as pd
import numpy as np

def load_data(ticker, data_src, interval, start_data_date, end_data_date):
    
    df = pd.read_csv("{}/{}_{}.csv".format(data_src, ticker, interval), sep=',', index_col = 'date') 
    df.index = pd.DatetimeIndex(df.index)
    df = df[df.index >= start_data_date]
    df = df[df.index <= end_data_date]
    if len(df) > 0:
        df["ticker"] = df.iloc[0]["symbol"].split(":")[1]

    return df

def clean_data(stock_df_storage, tickers):

    max_daily_len = 0
    max_trade_index = None
    for ticker in tickers:
        source_df = stock_df_storage[ticker]
        if len(source_df) > max_daily_len:
            max_daily_len = len(source_df)
            max_trade_index = source_df.index
            
    for ticker in tickers:    
        missing_points = np.setdiff1d(max_trade_index, stock_df_storage[ticker].index)
        if len(missing_points) > 0:
            print("{} is missing at: {}".format(ticker, missing_points))
            for ind in missing_points:
                stock_df_storage[ticker].loc[ind] = np.nan
            stock_df_storage[ticker] = stock_df_storage[ticker].sort_index(axis = 0, ascending = True)
            stock_df_storage[ticker]["open"] = stock_df_storage[ticker]["open"].interpolate()
            stock_df_storage[ticker]["high"] = stock_df_storage[ticker]["high"].interpolate()
            stock_df_storage[ticker]["low"] = stock_df_storage[ticker]["low"].interpolate()
            stock_df_storage[ticker]["close"] = stock_df_storage[ticker]["close"].interpolate()
            stock_df_storage[ticker]["volume"] = stock_df_storage[ticker]["volume"].interpolate()
            stock_df_storage[ticker]["symbol"] = stock_df_storage[ticker].iloc[0]["symbol"]
            stock_df_storage[ticker]["ticker"] = stock_df_storage[ticker].iloc[0]["ticker"]

    return stock_df_storage

def prepare_env_df(df_storage):

    env_df = pd.DataFrame()
    for df in df_storage:    
        # df = df.filter(needed_columns)
        df = df.reset_index()
        env_df = pd.concat([env_df, df], ignore_index=True)

    return env_df

def prepare_env_df_datetime_indexing(env_df, indices):
    env_df_indexing = {}    
    for idx in indices:
        env_df_indexing[idx] = env_df[env_df["date"] == idx]
    
    return env_df_indexing





# # Load data from src and return list of stock within 
# def load_data(stock_list, data_src, interval, start_data_date, end_data_date):
#     stocks = []
#     sources = []

#     index_df = pd.read_csv(data_src + "VNINDEX" + "_1day.csv", sep=',', index_col = 'datetime') 
#     index_df.index = pd.DatetimeIndex(index_df.index)
#     index_df = index_df[index_df.index >= start_data_date]
#     index_df = index_df[index_df.index <= end_data_date]

#     for stock in stock_list:    
        
#         source_df = pd.read_csv(data_src + stock + "_" + interval + ".csv", sep=',', index_col = 'datetime')      
#         source_df.index = pd.DatetimeIndex(source_df.index)
#         source_df = source_df[source_df.index >= start_data_date]
#         source_df = source_df[source_df.index <= end_data_date]
        
#         if start_data_date != source_df.index[0]:
#             print("Remove {} because of missing data".format(stock))
#         else:
#             stocks.append(stock)
#             sources.append(source_df)

#     if len(stocks) == 0:
#         print("Error: the start data date is invalid, please check if it is a weekend or holiday")

#     return stocks, sources


            
    