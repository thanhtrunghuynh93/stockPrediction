from stockstats import StockDataFrame as Sdf
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import IchimokuIndicator
import pandas as pd
import numpy as np
from indicators.PSAR import PSAR

periods = [10, 20]

def get_ratio(x, y):
    epsilon = 0.0000001
    # return (x - y) / abs(y + epsilon)
    return x / abs(y + epsilon) - 1

def supertrend(df, features):

    muls = [3, 2]
    windows = [12, 11]

    for i in range(len(windows)):
        Sdf.SUPERTREND_MUL = muls[i]
        Sdf.SUPERTREND_WINDOW = windows[i]
        stockstat = Sdf.retype(df.copy())
        df['supertrend_{}'.format(windows[i])] = stockstat['supertrend'] 
        df['supertrend_{}_r'.format(windows[i])] = df['supertrend_{}'.format(windows[i])] / df['close']
        features.extend(['supertrend_{}_r'.format(windows[i])])

    return df, features

def close_ratio(df, features, periods):
    stockstat = Sdf.retype(df.copy())    

    for period in periods:         
        df["close_{}_max_ratio".format(period)] = get_ratio(stockstat["close"], stockstat["close_-{}~0_max".format(period)])
        df["close_{}_min_ratio".format(period)] = get_ratio(stockstat["close"], stockstat["close_-{}~0_min".format(period)])
        features.extend(["close_{}_max_ratio".format(period), "close_{}_min_ratio".format(period)])

    return df, features
    
def volume_ratio(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["volume_{}_max_ratio".format(period)] = get_ratio(stockstat["volume"], stockstat["volume_-{}~0_max".format(period)])
        df["volume_{}_min_ratio".format(period)] = get_ratio(stockstat["volume"], stockstat["volume_-{}~0_min".format(period)])
        features.extend(["volume_{}_max_ratio".format(period), "volume_{}_min_ratio".format(period)])
    
    return df, features

def close_sma(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:
        df["close_sma_{}".format(period)] = stockstat["close_{}_sma".format(period)]
        df["close_sma_{}_ratio".format(period)] = get_ratio(df["close"], stockstat["close_{}_sma".format(period)])
        features.extend(["close_sma_{}_ratio".format(period)])

    df["close_sma_50"] = stockstat["close_50_sma"]
    df["close_sma_5"] = stockstat["close_5_sma"]
    
    return df, features
    

def volume_sma(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["volume_sma_{}".format(period)] = stockstat["volume_{}_sma".format(period)]
        df["volume_sma_{}_ratio".format(period)] = get_ratio(df["volume"], stockstat["volume_{}_sma".format(period)])
        features.extend(["volume_sma_{}_ratio".format(period)])
    
    return df, features

def close_ema(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["close_ema_{}_ratio".format(period)] = get_ratio(df["close"], stockstat["close_{}_ema".format(period)])
        features.extend(["close_ema_{}_ratio".format(period)])
    
    return df, features
    

def volume_ema(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["volume_ema_{}".format(period)] = stockstat["volume_{}_ema".format(period)]
        df["volume_ema_{}_ratio".format(period)] = get_ratio(df["volume"], stockstat["volume_{}_ema".format(period)])
        features.extend(["volume_ema_{}_ratio".format(period)])
    
    return df, features


def atr(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["atr_{}_ratio".format(period)] = get_ratio(stockstat["atr_{}".format(period)], df["close"])
        features.extend(["atr_{}_ratio".format(period)])
    
    return df, features    

def adx(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["adx_{}_ratio".format(period)] = stockstat["dx_{}_ema".format(period)] / 25 - 1        
        features.extend(["adx_{}_ratio".format(period)])

    return df, features

def kdj(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["kdjk_{}".format(period)] = stockstat["kdjk_{}".format(period)] / 50 - 1    
        df["kdjd_{}".format(period)] = stockstat["kdjd_{}".format(period)] / 50 - 1    
        df["kdj_{}_ratio".format(period)] = get_ratio(stockstat["kdjk_{}".format(period)], stockstat["kdjd_{}".format(period)])
        features.extend(["kdjk_{}".format(period), "kdjd_{}".format(period), "kdj_{}_ratio".format(period)])
        
    return df, features

def rsi(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    period = 14
    df["rsi_{}".format(period)] = stockstat["rsi_{}".format(period)] / 50 - 1 
    df["rsi_{}_change".format(period)] = (df["rsi_{}".format(period)] - df["rsi_{}".format(period)].shift(2))
    features.extend(["rsi_{}".format(period), "rsi_{}_change".format(period)])
    return df, features

def macd(df, features, medium_period = 10, slow_period = 20):
    
    stockstat = Sdf.retype(df.copy())    

    ema_short = 'close_12_ema'
    ema_long = 'close_26_ema'
    ema_signal = 'macd_9_ema'
    fast = stockstat[ema_short]
    slow = stockstat[ema_long]
    stockstat['macd'] = fast - slow
    stockstat['macds'] = stockstat[ema_signal]    
    stockstat['macdh'] = (stockstat['macd'] - stockstat['macds'])
    df['macdh_normed'] = stockstat["macdh"] / abs(stockstat['macds'])    
    df['macdh_returned'] = (df['macdh_normed'] - df['macdh_normed'].shift(2))

    features.extend(["macdh_normed", "macdh_returned"])
    return df, features


def mfi(df, features, periods):

    period = 14
    mfi_generator = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=period)    
    df['mfi_{}'.format(period)] = mfi_generator.money_flow_index() / 50 - 1
    df['mfi_{}_change'.format(period)] = (df['mfi_{}'.format(period)] - df['mfi_{}'.format(period)].shift(2))
    features.extend(["mfi_{}".format(period), 'mfi_{}_change'.format(period)])

    # for period in periods:               
    #     mfi_generator = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=period)        
    #     df['mfi_{}'.format(period)] = mfi_generator.money_flow_index() / 50 - 1           
    #     features.extend(["mfi_{}".format(period)])
    return df, features

def bb(df, features):
    stockstat = Sdf.retype(df.copy())
    df['boll_lb'] = stockstat['boll_lb']
    df['boll_ub'] = stockstat['boll_ub']
    df['boll_width'] = df['boll_ub'] / df['boll_lb'] - 1
    df['boll_width_change'] = df['boll_width'] - df['boll_width'].shift(2)

    features.extend(["boll_width", "boll_width_change"])
    return df, features

def psar(df, features):
    indic = PSAR()
    indic.calcPSAR(df['high'].values, df['low'].values)
    
    df['PSAR'] = indic.psar_list
    df['PSAR_r'] = df['PSAR'] / df['close'] - 1
    df['PSAR_trend'] = indic.trend_list
    df['PSAR_reverse'] = df['PSAR_trend'] - df['PSAR_trend'].shift(1)

    features.extend(["PSAR_r", "PSAR_trend", 'PSAR_reverse'])
    return df, features

def trend_return(df, features):
    df['daily_return'] = df['close'].pct_change()
    df['trend_return'] = df['close'].pct_change(periods=5) # 5 is a week
    df['trend_return'] = df['trend_return'].shift(-5) # 5 is a week

    features.extend(["trend_return"])
    return df, features

def trend(df, features, trend_up_threshold, trend_down_threshold):
    df["trend"] = 0
    df.loc[(df['trend_return'] > trend_up_threshold), 'trend'] = 1
    df.loc[(df['trend_return'] < -trend_down_threshold), 'trend'] = 2

    features.extend(["trend"])
    return df, features

def arithmetic_returns(df, features):
    df['open_r'] = df['open'] / df['close'] - 1 # Create arithmetic returns column
    df['high_r'] = df['high'] / df['close'] - 1# Create arithmetic returns column
    df['low_r'] = df['low'] / df['close']  - 1 # Create arithmetic returns column
    df['close_r'] = df['close'].pct_change()  # Create arithmetic returns column
    df['volume_r'] = df['volume'].pct_change()

    features.extend(["open_r", "high_r", "low_r", "close_r", "volume_r"])
    return df, features

def obv(df, features, periods):
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()
    period = 20
    df['obv_{}'.format(period)] = df['obv'].pct_change()    
    features.extend(['obv_{}'.format(period)])
    return df, features

def ichimoku(df, features, fast_period = 9, medium_period = 26, slow_period = 52):
    ichimoku = IchimokuIndicator(df['high'], df['low'], fast_period, medium_period, slow_period)
    df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()

    features.extend(['ichimoku_conversion_line', 'ichimoku_base_line', 'ichimoku_a', 'ichimoku_b'])
    return df, features

def add_trend(df, trend_ahead, trend_up_threshold, trend_down_threshold):
    df['daily_return'] = df['close'].pct_change()
    df['stability'] = df['daily_return'].shift(1).rolling(10).std()
    df['trend_return'] = df['close'].pct_change(periods=trend_ahead)
    df['trend_return'] = df['trend_return'].shift(-trend_ahead)    
    df['trend_log_return'] = np.log(df['trend_return'] + 1)
    
    #trend    
    df["trend"] = 0
    df.loc[(df['trend_return'] > trend_up_threshold), 'trend'] = 1
    df.loc[(df['trend_return'] < -trend_down_threshold), 'trend'] = 2

    return df

# Stock Trend Prediction Using Candlestick Charting and Ensemble Machine Learning Techniques with a Novelty Feature Engineering Scheme
def k_line(df, features):
    df['k_line'] = 0
    msk_0 = (df['open'] == df['close']) & (df['open'] == df['low']) & (df['open'] == df['high'])
    msk_1 = (df['open'] == df['close']) & (df['open'] == df['high']) & (df['open'] == df['low'])
    msk_2 = (df['open'] == df['low']) & (df['close'] == df['high'])
    msk_3 = (df['open'] == df['high']) & (df['close'] == df['low'])
    msk_4 = (df['open'] == df['close']) & (df['open'] == df['high']) & (df['low'] < df['close'])
    msk_5 = (df['open'] == df['close']) & (df['open'] == df['low']) & (df['high'] > df['high'])
    msk_6 = (df['open'] == df['close']) & (df['low'] < df['close']) & (df['high'] > df['close'])
    msk_7 = (df['open'] > df['low']) & (df['close'] > df['open']) & (df['close'] == df['high'])
    msk_8 = (df['close'] > df['low']) & (df['open'] > df['close']) & (df['open'] == df['high'])
    msk_9 = (df['open'] == df['low']) & (df['close'] > df['open']) & (df['high'] > df['close'])
    msk_10 = (df['close'] == df['low']) & (df['open'] > df['close']) & (df['high'] > df['open'])
    msk_11 = (df['open'] < df['close']) & (df['low'] < df['open']) & (df['high'] > df['close'])
    msk_12 = (df['open'] > df['close']) & (df['low'] < df['close']) & (df['high'] > df['open'])
    
    df.loc[msk_0, 'k_line'] = 0
    df.loc[msk_1, 'k_line'] = 1
    df.loc[msk_2, 'k_line'] = 2
    df.loc[msk_3, 'k_line'] = 3
    df.loc[msk_4, 'k_line'] = 4
    df.loc[msk_5, 'k_line'] = 5
    df.loc[msk_6, 'k_line'] = 6
    df.loc[msk_7, 'k_line'] = 7
    df.loc[msk_8, 'k_line'] = 8
    df.loc[msk_9, 'k_line'] = 9
    df.loc[msk_10, 'k_line'] = 10
    df.loc[msk_11, 'k_line'] = 11
    df.loc[msk_12, 'k_line'] = 12
    features.extend(['k_line'])
    return df, features

def eight_trigrams(df, features):
    df['high_pre'] = df['high'].shift(1)
    df['low_pre'] = df['low'].shift(1)
    df['close_pre'] = df['close'].shift(1)
    df['open_pre'] = df['open'].shift(1)
    df['eight_trigrams'] = 0
    # bear high
    msk_0 = (df['high'] > df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] > df['low_pre'])
    msk_1 = (df['high'] < df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] < df['low_pre'])
    msk_2 = (df['high'] < df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] > df['low_pre'])
    msk_3 = (df['high'] > df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] > df['low_pre'])
    msk_4 = (df['high'] < df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] < df['low_pre'])
    msk_5 = (df['high'] > df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] < df['low_pre'])
    msk_6 = (df['high'] < df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] > df['low_pre'])
    msk_7 = (df['high'] > df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] < df['low_pre'])
    df.loc[msk_0, 'eight_trigrams'] = 0
    df.loc[msk_1, 'eight_trigrams'] = 1
    df.loc[msk_2, 'eight_trigrams'] = 2
    df.loc[msk_3, 'eight_trigrams'] = 3
    df.loc[msk_4, 'eight_trigrams'] = 4
    df.loc[msk_5, 'eight_trigrams'] = 5
    df.loc[msk_6, 'eight_trigrams'] = 6
    df.loc[msk_7, 'eight_trigrams'] = 7
    features.extend(['eight_trigrams'])

    return df, features

def remove_outliers(df, features, threshold = 1000):
    for feat in features:
        df[feat].fillna(threshold, inplace = True)
        df.loc[df[feat] > threshold, feat] = threshold
        df.loc[df[feat] < -threshold, feat] = -threshold
    return df

def add_indicators_all(source_data, indicators, periods = [10, 20], trend_ahead = 5, trend_up_threshold = 0.03, trend_down_threshold = 0.03, outlier_threshold = 1000):

    data_storages = {}
    feature_storages = {}
    for ticker in list(source_data.keys()):
        result_df, features = add_indicators(source_data[ticker], indicators, periods, trend_ahead, trend_up_threshold, trend_down_threshold, outlier_threshold)
        data_storages[ticker] = result_df
        feature_storages[ticker] = features

    if "rs" in indicators:

        milestones = [10, 20, 50, 100]
        weight = [0.1, 0.15, 0.25, 0.5]
        stocks = list(source_data.keys())

        for ticker in stocks:
            
            changes = [data_storages[ticker]["close"].pct_change(milestones[i]).values * weight[i] for i in range(len(milestones))]
            aggregated_changes = np.array(changes)
            aggregated_changes = np.sum(aggregated_changes, axis = 0)
            data_storages[ticker]["agg_changes"] = aggregated_changes   

        df = data_storages[stocks[0]]
        for index in df.index:
            temp = []
            for ticker in stocks:
                temp.append(data_storages[ticker].loc[index]["agg_changes"])
                rank = np.argsort(temp)
            
            for i in range(len(stocks)):    
                ticker_rs = int(list(rank).index(i) / len(stocks) * 100)
                if index in data_storages[stocks[i]].index:
                    data_storages[stocks[i]].loc[index, "rs"] = ticker_rs / 100

        for ticker in stocks:
            data_storages[ticker]["rs_change"] = data_storages[ticker]["rs"] - data_storages[ticker]["rs"].shift(1)        
            if "rs" in indicators:    
                feature_storages[ticker].extend(["rs", "rs_change"])
        
    return data_storages, feature_storages

def add_indicators(source_df, indicators, periods = [10, 20], trend_ahead = 5, trend_up_threshold = 0.03, trend_down_threshold = 0.03, outlier_threshold = 1000):
    
    df = source_df.copy()    
    features = []

    if "close_ratio" in indicators:
        df, features = close_ratio(df, features, periods)

    if "volume_ratio" in indicators:
        df, features = volume_ratio(df, features, periods)

    if "close_sma" in indicators:
        df, features = close_sma(df, features, periods)

    if "volume_sma" in indicators:
        df, features = volume_sma(df, features, periods)
    
    if "close_ema" in indicators:
        df, features = close_ema(df, features, periods)

    if "volume_ema" in indicators:
        df, features = volume_ema(df, features, periods)

    if "atr" in indicators:
        df, features = atr(df, features, periods)
    
    #Adx, greater than 25 is a strong trend    
    if "adx" in indicators:
        df, features = adx(df, features, periods)
    
    #Stochastic Oscillators k
    if "kdj" in indicators:    
        df, features = kdj(df, features, periods)
    
    if "rsi" in indicators:            
        df, features = rsi(df, features, periods)

    if "MACD" in indicators:            
        df, features = macd(df, features)
    
    if "boll" in indicators:                
        df, features = bb(df, features)

    if "mfi" in indicators:                
        df, features = mfi(df, features, periods)

    if "obv" in indicators:
        df, features = obv(df, features, periods)
    
    if "k_line" in indicators:
        df, features = k_line(df, features)
    
    if "eight_trigrams" in indicators:
        df, features = eight_trigrams(df, features)

    if "psar" in indicators:
        df, features = psar(df, features)

    if "supertrend" in indicators:
        df, features = supertrend(df, features)


    df, features = arithmetic_returns(df, features)
    df = add_trend(df, trend_ahead, trend_up_threshold, trend_down_threshold)
    df = remove_outliers(df, features, threshold = outlier_threshold)

    # df = df[max(periods):] #Remove first incomplete rows
    # df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

    return df, features


def __add_indicators(source_df, indicators, short_trend = 10, long_trend = 20, trend_ahead = 5, trend_up_threshold = 0.03, trend_down_threshold = 0.03):
    
    df = source_df.copy()    
    stockstat = Sdf.retype(df.copy())    
    features = []

    if "close_ratio" in indicators:
        df["close_short_max".format(short_trend)] = stockstat["close_-{}~0_max".format(short_trend)]
        df["close_short_max_ratio".format(short_trend)] = stockstat["close"] / stockstat["close_-{}~0_max".format(short_trend)] - 1
        df["close_long_max".format(long_trend)] = stockstat["close_-{}~0_max".format(long_trend)]
        df["close_long_max_ratio".format(short_trend)] = stockstat["close"] / stockstat["close_-{}~0_max".format(long_trend)] - 1
        df["close_short_min".format(short_trend)] = stockstat["close_-{}~0_min".format(short_trend)]
        df["close_short_min_ratio".format(short_trend)] = stockstat["close"] / stockstat["close_-{}~0_min".format(short_trend)] - 1
        df["close_long_min".format(long_trend)] = stockstat["close_-{}~0_min".format(long_trend)]
        df["close_long_min_ratio".format(short_trend)] = stockstat["close"] / stockstat["close_-{}~0_min".format(long_trend)] - 1

        features.extend(["close_short_max_ratio", "close_long_max_ratio", "close_short_min_ratio", "close_long_min_ratio"])

    if "volume_ratio" in indicators:
        df["volume_short_max".format(short_trend)] = stockstat["volume_-{}~0_max".format(short_trend)]
        df["volume_short_max_ratio".format(short_trend)] = stockstat["volume"] / stockstat["volume_-{}~0_max".format(short_trend)] - 1
        df["volume_long_max".format(long_trend)] = stockstat["volume_-{}~0_max".format(long_trend)]
        df["volume_long_max_ratio".format(short_trend)] = stockstat["volume"] / stockstat["volume_-{}~0_max".format(long_trend)] - 1
        df["volume_short_min".format(short_trend)] = stockstat["volume_-{}~0_min".format(short_trend)]
        df["volume_short_min_ratio".format(short_trend)] = stockstat["volume"] / stockstat["volume_-{}~0_min".format(short_trend)] - 1
        df["volume_long_min".format(long_trend)] = stockstat["volume_-{}~0_min".format(long_trend)]
        df["volume_long_min_ratio".format(short_trend)] = stockstat["volume"] / stockstat["volume_-{}~0_min".format(long_trend)] - 1

        features.extend(["volume_short_max_ratio", "volume_long_max_ratio", "volume_short_min_ratio", "volume_long_min_ratio"])

    if "close_sma" in indicators:
        df["close_short_sma"] = stockstat["close_{}_sma".format(short_trend)]
        df["close_long_sma"] = stockstat["close_{}_sma".format(long_trend)]
        # df["close_sma_ratio"] = df["close_{}_sma".format(long_trend)] / df["close_{}_sma".format(short_trend)] - 1
        df["short_long_close_sma_ratio"] = df["close_short_sma"] / df["close_long_sma"] - 1
        df["long_close_sma_ratio"] = df["close"] / df["close_long_sma"] - 1
        df["short_close_sma_ratio"] = df["close"] / df["close_short_sma"] - 1

        features.extend(["short_long_close_sma_ratio", "long_close_sma_ratio", "short_close_sma_ratio"])


    if "volume_sma" in indicators:
        df["volume_short_sma"] = stockstat["volume_{}_sma".format(short_trend)]
        df["volume_long_sma"] = stockstat["volume_{}_sma".format(long_trend)]
        # df["volume_sma_ratio"] = df["volume_{}_sma".format(long_trend)] / df["volume_{}_sma".format(short_trend)] - 1
        df["short_long_volume_sma_ratio"] = df["volume_short_sma"] / df["volume_long_sma"] - 1
        df["long_volume_sma_ratio"] = df["volume"] / df["volume_long_sma"] - 1
        df["short_volume_sma_ratio"] = df["volume"] / df["volume_short_sma"] - 1
        features.extend(["short_long_volume_sma_ratio", "long_volume_sma_ratio", "short_volume_sma_ratio"])

    if "atr" in indicators:
        df["atr_short"] = stockstat["atr_{}".format(short_trend)]
        df["atr_long"] = stockstat["atr_{}".format(long_trend)]
        # df["atr_ratio"] = stockstat["atr_{}".format(short_trend)] / stockstat["atr_{}".format(long_trend)] - 1
        df["atr_ratio"] = df["atr_long"] / df["atr_short"] - 1
        features.extend(["atr_ratio"])

    
    #Adx, greater than 25 is a strong trend    
    if "adx" in indicators:
        df["adx_short".format(short_trend)] = stockstat["dx_{}_ema".format(short_trend)] / 25 - 1        
        df["adx_long".format(long_trend)] = stockstat["dx_{}_ema".format(long_trend)] / 25 - 1        
        # df["adx_{}".format(long_trend)] = stockstat["dx_{}_ema".format(long_trend)] / 25
        # df["adx_ratio"] = df["adx_{}".format(short_trend)] / df["adx_{}".format(long_trend)]
        features.extend(["adx_short", "adx_long"])
    
    #Stochastic Oscillators k
    if "kdj" in indicators:    
        df["kdj_short"] = stockstat["kdjk_{}".format(short_trend)] / 50 - 1
        df["kdj_short_ratio"] = stockstat["kdjk_{}".format(short_trend)] / stockstat["kdjd_{}".format(short_trend)] - 1
        df["kdj_long"] = stockstat["kdjk_{}".format(long_trend)] / 50 - 1
        df["kdj_long_ratio"] = stockstat["kdjk_{}".format(long_trend)] / stockstat["kdjd_{}".format(long_trend)] - 1
        # df["kdjk_ratio"] = df["kdjk_long"] / df["kdjk_short"] - 0.5
        features.extend(["kdj_short", "kdj_short_ratio", "kdj_long", "kdj_long_ratio"])

    # #Stochastic Oscillators d
    # df["kdjd_short"] = stockstat["kdjd_{}".format(short_trend)]
    # df["kdjd_long"] = stockstat["kdjd_{}".format(long_trend)]
    # df["kdjd_ratio"] = df["kdjd_long"] / df["kdjd_short"] - 1

    # #Stochastic Oscillators j
    # df["kdjj_short"] = stockstat["kdjj_{}".format(short_trend)]
    # df["kdjj_long"] = stockstat["kdjj_{}".format(long_trend)]
    # df["kdjj_ratio"] = df["kdjj_long"] / df["kdjj_short"] - 1

    # # #MFI
    # # df["mfi".format(short_trend)] = get_mfi(stockstat, 5)["mfi_{}".format(short_trend)]

    if "rsi" in indicators:            
        #rsi
        df["rsi_short"] = stockstat["rsi_{}".format(short_trend)] / 50 - 1
        df["rsi_long"] = stockstat["rsi_{}".format(long_trend)] / 50 - 1
        df["rsi_ratio"] = df["rsi_long"] / df["rsi_short"] - 1
        features.extend(["rsi_short", "rsi_long", "rsi_ratio"])

    if "MACD" in indicators:            
        df['MACD_ratio'] = (stockstat['close_{}_ema'.format(long_trend)] / stockstat['close_{}_ema'.format(short_trend)]) - 1
        features.extend(["MACD_ratio"])
    
    if "boll" in indicators:                
        #bollinger band / close price
        # df['boll'] = stockstat['boll'] / df['close']
        df['boll_lb'] = stockstat['boll_lb'] / df['close'] - 1
        df['boll_ub'] = stockstat['boll_ub'] / df['close'] - 1

        features.extend(["boll_lb", "boll_ub"])

    df['daily_return'] = df['close'].pct_change()
    df['trend_return'] = df['close'].pct_change(periods=trend_ahead)
    df['trend_return'] = df['trend_return'].shift(-trend_ahead)    
    
    #trend    
    df["trend"] = 0
    df.loc[(df['trend_return'] > trend_up_threshold), 'trend'] = 1
    df.loc[(df['trend_return'] < -trend_down_threshold), 'trend'] = 2

    # features.extend(["trend"])

    df['open_r'] = df['open'] / df['close'] - 1 # Create arithmetic returns column
    df['high_r'] = df['high'] / df['close'] - 1# Create arithmetic returns column
    df['low_r'] = df['low'] / df['close']  - 1 # Create arithmetic returns column
    df['close_r'] = df['close'].pct_change()  # Create arithmetic returns column
    df['volume_r'] = df['volume'].pct_change()  
    
    features.extend(["open_r", "high_r", "low_r", "close_r", "volume_r"])
    df = df[long_trend:] #Remove first incomplete rows
    # df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

    return df, features

if __name__ == '__main__':
    stock = "HPG"
    data_src = "data/Vietnam/HOSE/"
    source_df = pd.read_csv(data_src + stock + "_1day.csv", sep=',', index_col = 'datetime')        
    indicators = ["close_ratio", "close_sma", "volume_ratio", "volume_sma", "atr", "adx", "rsi", "MACD", "boll"]
    df, features = add_indicators(source_df, indicators)

    print(df.tail(10)["close"])
    print(df.tail(10)["trend"])
    print(df.tail(10)["trend_return"])
    print(df.tail(10)["close"])
