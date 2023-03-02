from pandas import DataFrame
from stockstats import StockDataFrame as Sdf
import pandas as pd
import numpy as np
from analysis.TA_analysis import TA_analysis
from analysis.FA_analysis import FA_analysis
from utils.date_util import get_current_report_quarter, generate_quarters

available_filters = ["sma", "ema", "rsi", "CANSLIM", "TA_scoring", "intrinsic_value"]

class FilterFactory():
    def __init__(self, filters, env_df, stock_infos, source = "data/Vietnam", ta_score_threshold = 4, intrinsic_mos = 0.4):

        self.filters = filters
        self.env_df = env_df.set_index(["date", "ticker"])
        
        self.trade_sessions = env_df["date"].unique()
        self.ticker_symbols = env_df["symbol"].unique()

        start_quarter = get_current_report_quarter(self.trade_sessions[0])
        end_quarter = get_current_report_quarter(self.trade_sessions[-1])

        #Get the corresponding trading quarters from the start date to the end date
        self.trade_quarters = generate_quarters(start_quarter[0], start_quarter[1], end_quarter[0], end_quarter[1])
        self.trade_quarters_days = {}
        for quarter in self.trade_quarters:
            self.trade_quarters_days[quarter] = []                

        for index in self.trade_sessions:
            current_quarter = get_current_report_quarter(index, string_format = True)
            self.trade_quarters_days[current_quarter].append(index)
        
        self.tickers = []

        for ticker in self.ticker_symbols:
            self.tickers.append(ticker.split(":")[1])

        #Initialize necessary variable for the filters        
        if "CANSLIM" in filters:
            self.env_df["CANSLIM_shortlist"] = False
            for quarter in self.trade_quarters:
                canslim_candidates = np.load("{}/CANSLIM/{}_shortlist.npy".format(source, quarter.replace("/", "_")), allow_pickle = True)                
                days_in_quarter = self.trade_quarters_days[quarter]
                self.env_df.loc[(days_in_quarter, canslim_candidates), ['CANSLIM_shortlist']] = True

        if "intrinsic_value" in filters:
            self.price_estimator = pd.read_csv(source + "/estimate_price.csv", index_col = "ticker")
            self.env_df["max_intrinsic_value"] = 0
            self.env_df["min_intrinsic_value"] = 0
            self.intrinsic_mos = intrinsic_mos
            
            print("Loading intrinsic values for backtesting")
            for stock in self.tickers:
                print(stock)
                price_estimation_insight = self.price_estimator.loc[stock]
                for quarter in self.trade_quarters:     
                    max_price = float(price_estimation_insight[quarter][1])        
                    min_price = float(price_estimation_insight[quarter][0])        
                    days_in_quarter = self.trade_quarters_days[quarter]
                    self.env_df.loc[(days_in_quarter, stock), ['max_intrinsic_value']] = max_price
                    self.env_df.loc[(days_in_quarter, stock), ['min_intrinsic_value']] = min_price

            self.env_df['intrinsic_mos'] = self.env_df["close"] < self.env_df["max_intrinsic_value"] * (1 - intrinsic_mos)
            
        if "TA_scoring" in filters:
            self.score_threshold = ta_score_threshold            
            self.TA_analysis = TA_analysis(stock_infos, self.tickers)
            self.env_df = self.TA_analysis.calculate_score(self.env_df)
            
        self.cache = []
        self.env_df = self.env_df.reset_index()
        
        for idx in self.trade_sessions:
            candidates = np.array(range(len(self.tickers)))
            current_market_df = self.env_df[self.env_df["date"] == idx]
            candidates = self.filter(current_market_df, candidates)

            #Only buy if there is enough candidates
            if len(candidates) < 2:
                candidates = []

            self.cache.append(candidates)

    def TA_scoring_filter(self, current_market_df, candidates, score_threshold):
        index = current_market_df.iloc[0]["date"]
        current_market_df = self.env_df[self.env_df["date"] == index]
        candidate_by_score = current_market_df["TA_score"].to_numpy()
        candidate_by_score = np.where(candidate_by_score >= score_threshold)[0]
        candidates = np.intersect1d(candidates, candidate_by_score) 

        return candidates       

    def sma_filter(self, current_market_df, candidates, period = 20):

        candidate_by_long_sma_filter = current_market_df["close_sma_{}_ratio".format(period)].to_numpy()
        candidate_by_long_sma_filter = np.where(candidate_by_long_sma_filter >= 0)[0]
        candidates = np.intersect1d(candidates, candidate_by_long_sma_filter)
        
        return candidates

    def ema_filter(self, current_market_df, candidates):

        candidate_by_short_long_ema_filter = current_market_df["short_long_close_ema_ratio"].to_numpy()
        candidate_by_short_long_ema_filter = np.where(candidate_by_short_long_ema_filter >= 0)[0]
        candidate_by_short_ema_filter = current_market_df["short_close_ema_ratio"].to_numpy()
        candidate_by_short_ema_filter = np.where(candidate_by_short_ema_filter >= 0)[0]
        candidates = np.intersect1d(candidates, candidate_by_short_ema_filter)
        candidates = np.intersect1d(candidates, candidate_by_short_long_ema_filter)

        return candidates

    def rsi_filter(self, current_market_df, candidates, period = 20):
        candidate_by_long_rsi_filter = current_market_df["rsi_{}".format(period)].to_numpy()
        candidate_by_long_rsi_filter_max = np.where(candidate_by_long_rsi_filter <= 0.4)[0]
        candidate_by_long_rsi_filter_min = np.where(candidate_by_long_rsi_filter >= 0)[0]
        candidates = np.intersect1d(candidates, candidate_by_long_rsi_filter_max)
        candidates = np.intersect1d(candidates, candidate_by_long_rsi_filter_min)

        return candidates

    def CANSLIM_filter(self, current_market_df, candidates):
        candidate_by_CANSLIM_filter = current_market_df["CANSLIM_shortlist"].to_numpy()
        candidate_by_CANSLIM_filter = np.where(candidate_by_CANSLIM_filter == True)[0]
        candidates = np.intersect1d(candidates, candidate_by_CANSLIM_filter)

        return candidates

    def intrinsic_filter(self, current_market_df, candidates):

        candidate_by_intrinsic_filter = current_market_df["intrinsic_mos"].to_numpy()
        candidate_by_intrinsic_filter = np.where(candidate_by_intrinsic_filter == True)[0]
        candidates = np.intersect1d(candidates, candidate_by_intrinsic_filter)

        return candidates       
            

    def filter(self, current_market_df, candidates):
        if "sma" in self.filters:
            candidates = self.sma_filter(current_market_df, candidates)
        
        if "ema" in self.filters:
            candidates = self.ema_filter(current_market_df, candidates)
        
        if "rsi" in self.filters:
            candidates = self.rsi_filter(current_market_df, candidates)

        if "CANSLIM" in self.filters:
            candidates = self.CANSLIM_filter(current_market_df, candidates)

        if "TA_scoring" in self.filters:
            candidates = self.TA_scoring_filter(current_market_df, candidates, self.score_threshold)

        if "intrinsic_value" in self.filters:
            candidates = self.intrinsic_filter(current_market_df, candidates)

        return candidates


    

