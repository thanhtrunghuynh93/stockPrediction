import numpy as np

criteria = ["ta_candle_shape", "ta_vol", "ta_ma5", "ta_rs", "ta_rs_change", "ta_MACD", "ta_rsi", "ta_stability", "ta_boll_width", "ta_psar", "ta_supertrend"]
criteria_coeff = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]

class TA_analysis:

    def __init__(self, stock_infos, tickers, criteria = criteria, criteria_coeff = criteria_coeff):
        
        self.criteria = criteria
        self.criteria_coeff = criteria_coeff
        self.tickers = tickers
        self.stock_infos = stock_infos
        self.categories = stock_infos.loc[self.tickers]["category"].unique()

    def calculate_score(self, df):
        
        df["ta_candle_shape"] = (df["daily_return"] > 0.02) & ((df["close"] / df["open"]) > 1.02) & ((df["high"] / df["close"]) < 1.03)        
        df["ta_vol"] = ((df["volume"] / df["volume_sma_20"]) > 1.2) & ((df["volume"] / df["volume_sma_5"]) > 1.5) & (df["mfi_14_change"] > 0)
        df["ta_ma5"] = df["close_sma_5"] > df["close_sma_50"] 
        df["ta_rs"] = (df["rs"] * 100) > 40        
        df["ta_rs_change"] = (df["rs_change"] * 100) > 2
        df["ta_MACD"] = (df['macdh_returned'] > 0) & (df['macdh_normed'] > 0)        
        df["ta_rsi"] = (df["rsi_14_change"] > 0) & (df["rsi_14"] < 0.5)            
        df["ta_stability"] = df["stability"] < 0.05
        df["ta_boll_width"] = df["boll_width_change"] < 0
        df["ta_psar"] = df["PSAR_trend"] == 1
        df["ta_supertrend"] = (df["supertrend_11"] < 1) & (df["supertrend_12"] < 1)

        df["TA_score"] = 0

        for i in range(len(criteria)):
            df["TA_score"] = df["TA_score"] + df[self.criteria[i]].astype(int) * self.criteria_coeff[i]

        return df
        
    # def calculate_score(self, current_market_df):
        
    #     scores = []
        
    #     # category_rs = {}
    #     # category_rs_change = {}

    #     # for cat in self.categories:
    #     #     category_rs[cat] = []
    #     #     category_rs_change[cat] = []

    #     # for ticker in self.tickers:
    #     #     category = self.stock_infos.loc[ticker]["datx_category"]
    #     #     category_rs_change[category].append(current_market_df[current_market_df["ticker"] == ticker]["rs_change"].values[0])
    #     #     category_rs[category].append(current_market_df[current_market_df["ticker"] == ticker]["rs"].values[0])
                
    #     # for cat in self.categories:
            
    #     #     category_rs[cat] = np.mean(category_rs[cat])
    #     #     category_rs_change[cat] = np.mean(category_rs_change[cat])

    #     for ticker in self.tickers:

    #         insight = current_market_df[current_market_df["ticker"] == ticker]  
    #         category = self.stock_infos.loc[ticker]["datx_category"]

    #         criteria_check = {}
            
    #         criteria_check["candle_shape"] = (insight["daily_return"] > 0.02) & ((insight["close"] / insight["open"]) > 1.02) & ((insight["high"] / insight["close"]) < 1.03)        
    #         criteria_check["vol"] = ((insight["volume"] / insight["volume_sma_20"]) > 1) & ((insight["volume"] / insight["volume_sma_5"]) > 1.5) & (insight["mfi_14_change"] > 0)
    #         criteria_check["ma5"] = insight["close_sma_5"] > insight["close_sma_20"] 
    #         criteria_check["rs"] = (insight["rs"] * 100) > 40        
    #         criteria_check["rs_change"] = (insight["rs_change"] * 100) > 2
    #         criteria_check["MACD"] = (insight['macdh_returned'] > 0) & (insight['macdh_normed'] > 0)        
    #         criteria_check["rsi"] = (insight["rsi_14_change"] > 0) & (insight["rsi_14"] < 0.5)
    #         # criteria_check["category_rs"] = (category_rs_change[category] > 0) & ((category_rs[category] * 100) > 40)
    #         criteria_check["stability"] = insight["stability"] < 0.03
    #         criteria_check["boll_width"] = insight["boll_width_change"] < 0
    #         criteria_check["psar"] = insight["PSAR_trend"] == 1
            
    #         score = 0

    #         for i in range(len(criteria)):
    #             score += int(criteria_check[criteria[i]]) * criteria_coeff[i]               

    #         scores.append(score)

    #     return scores

