from utils.date_util import generate_quarters, get_last_quarters
import numpy as np
import math
import pandas as pd

# quarters = ['Q1/2019', 'Q2/2019', 'Q3/2019', 'Q4/2019', 'Q1/2020', 'Q2/2020', 'Q3/2020', 'Q4/2020', 'Q1/2021', 'Q2/2021', 'Q3/2021', 'Q4/2021', 'Q1/2022', 'Q2/2022', 'Q3/2022', 'Q4/2022']
# years = ['2017', '2018', '2019', '2020', '2021', '2022']
columns = ["ticker", "full_ticker", "freeFloat", "ROE", "RGCQ", "RGPQ", "RGQA", "PGCQ", "PGPQ", "PGQA", "RGCY", "RGPY", "RGYA", "PGCY", "PGPY", "PGYA"]


#stock_list: list of all stocks
#stock_infos: panda dataframe, storing the eternal fundamental infos of the stocks (i.e. markets, category)
#basicFA: current FA characteristics of the stocks


class CANSLIM_analysis:

    def __init__(self, stock_list, stock_infos, basicFA, trailing_quarter, trailing_year, insight_quarter = "Q2/2022"):

        #Get previous quarters and years for investigation
        quarters, years = get_last_quarters(insight_quarter)   

        # if "Q1" not in insight_quarter: #Only if the current quarter is Q1, then the year of previous quarter is available
        #     years = years[:-2]

        self.insight_quarter = insight_quarter
        self.stock_list = stock_list
        self.processed_fa_df = processFA(stock_list, stock_infos, basicFA, trailing_quarter, trailing_year, quarters, years)
        self.processed_fa_df = self.processed_fa_df.set_index("ticker")

    def calculate_performance(self, stock_data, start_test_date, end_test_date):
        
        changes = []
        drawdowns = []

        for ticker in self.stock_list:
            
            df = stock_data[ticker]
            
            test_df = df[df.index >= start_test_date]
            test_df = test_df[test_df.index <= end_test_date]
            test_prices = test_df[test_df.index <= end_test_date]["close"].values
            
            changes.append((test_prices[-1] / test_prices[0]) - 1)
            drawdowns.append(df["close"][-1] / np.max(df["close"].values) - 1)     
            
        self.processed_fa_df["changes"] = changes
        self.processed_fa_df["drawdowns"] = drawdowns

    score_cols = ["RGCQ", "RGPQ", "RGQA", "PGCQ", "PGPQ", "PGQA", "RGCY", "RGPY", "RGYA", "PGCY", "PGPY", "PGYA"]
    score_conds = [0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15]
    score_factors = [3, 1, 2, 4, 2, 3, 2, 1, 1, 3, 1, 2]

    hard_cols = ["ROE", "institutionalOwnership"]
    # hard_cols = ["ROE"]

    def calculate_score(self, ticker, score_cols = score_cols, score_conds = score_conds, score_factors = score_factors):
        return cal_score(self.processed_fa_df.loc[ticker], score_cols, score_conds, score_factors)
    
    def filter(self, hard_cols = hard_cols, score_cols = score_cols, score_conds = score_conds, score_factors = score_factors, threshold = 20, verbose = 1):

        filtered_results = self.processed_fa_df.copy()
        if "ROE" in hard_cols:
            filtered_results = filtered_results[filtered_results["ROE"] >= 9]
            filtered_results = filtered_results[filtered_results["ROE_average"] >= 9]
        if "institutionalOwnership" in hard_cols:
            filtered_results = filtered_results[filtered_results["institutionalOwnership"] > 0.45]
        
        candidates = filtered_results.index.values

        scores = []
        for stock in candidates:    
            score = cal_score(self.processed_fa_df.loc[stock], score_cols, score_conds, score_factors)
            scores.append(score)
            
        filtered_results["scores"] = scores

        filtered_results = filtered_results.sort_values(by=['scores'], ascending = False)
        filtered_results = filtered_results[filtered_results["scores"] >= threshold]

        # if verbose == 1:
        #     print("Report for {}".format(self.insight_quarter))

        #     print(filtered_results.index.values)

        #     print("Average changes of all stocks: {:.2f}% ({} stocks)".format(self.processed_fa_df["changes"].mean() * 100, len(self.processed_fa_df)))

        #     print("Average changes of CANSLIM+FA stocks without MA200 control: {:.2f}% ({} stocks)".format(filtered_results["changes"].mean() * 100, len(filtered_results)))

        
        return filtered_results.index.values

    def analysis_threshold(self, hard_cols = hard_cols, score_cols = score_cols, score_conds = score_conds, score_factors = score_factors, verbose = 1):

        print("Average changes of all stocks: {:.2f}% ({} stocks)".format(self.processed_fa_df["changes"].mean() * 100, len(self.processed_fa_df)))

        for threshold in range(10, 50):
            print("Score threshold: {}".format(threshold))
            self.filter(threshold = threshold, verbose = 1)
        
        # np.save("{}/{}_shortlist.npy".format(data_src, quarters[-1].replace("/", "_")), filtered_results.index.values)

# Supporting functions

def calculate_score(value, threshold):
    if math.isnan(value):
        return 0
    score = int(value / threshold)
    score = max(score, 0)
    score = min(score, 2)
    return score

def cal_score(insight, score_cols, score_conds, score_factors):
    
    score = 0

    for i in range(len(score_cols)):
        score += calculate_score(insight[score_cols[i]], score_conds[i]) * score_factors[i]
    
    return score

def processFA(stock_list, stock_infos, basicFA, trailing_quarter, trailing_year, quarters, years):
    dats = []
    columns = ["ticker", "full_ticker", "institutionalOwnership", "ROE", "ROE_average", "ROE_growth", "RGCQ", "RGPQ", "RGQA", "PGCQ", "PGPQ", "PGQA", "RGCY", "RGPY", "RGYA", "PGCY", "PGPY", "PGYA"]

    trailing_quarter = trailing_quarter.reset_index()
    trailing_quarter["quarter_year"] = "Q" + trailing_quarter["quarter"].astype(str) + "/" + trailing_quarter["year"].astype(str)
    trailing_quarter = trailing_quarter.set_index(["ticker", "quarter_year"])

    trailing_year = trailing_year.reset_index()
    trailing_year = trailing_year.set_index(["ticker", "year"])

    for stock in stock_list:
        
#         print("Processing {}".format(stock))

        res = []
        res.append(stock)
        res.append(stock_infos.loc[stock].exchange + ":" + stock_infos.loc[stock].name)

        institutionalOwnership = basicFA.loc[stock]["institutionalOwnership"]
        res.append(institutionalOwnership)

        current_quarter = quarters[-1]
        last_current_quarter = quarters[-5]
        previous_quarter = quarters[-2]
        last_previous_quarter = quarters[-6]

        p_previous_quarter = quarters[-3]
        pp_previous_quarter = quarters[-4]
        
        try:

            current_ROE = trailing_quarter.loc[(stock, current_quarter)]["netProfit"] / abs(trailing_quarter.loc[(stock, current_quarter)]["ownerEquity"]) * 100 * 4
            previous_ROE = trailing_quarter.loc[(stock, previous_quarter)]["netProfit"] / abs(trailing_quarter.loc[(stock, previous_quarter)]["ownerEquity"]) * 100 * 4
            p_previous_ROE = trailing_quarter.loc[(stock, p_previous_quarter)]["netProfit"] / abs(trailing_quarter.loc[(stock, p_previous_quarter)]["ownerEquity"]) * 100 * 4
            pp_previous_ROE = trailing_quarter.loc[(stock, pp_previous_quarter)]["netProfit"] / abs(trailing_quarter.loc[(stock, pp_previous_quarter)]["ownerEquity"]) * 100 * 4

            average_ROE = (current_ROE * 4 + previous_ROE * 3 + p_previous_ROE * 2 + pp_previous_ROE) / 10

            res.append(current_ROE)
            res.append(average_ROE)
            res.append(current_ROE / previous_ROE - 1)

            revenue_growth_current_quarter = trailing_quarter.loc[(stock, current_quarter)]["sales"] / trailing_quarter.loc[(stock, last_current_quarter)]["sales"] - 1
            revenue_growth_previous_quarter = trailing_quarter.loc[(stock, previous_quarter)]["sales"] / trailing_quarter.loc[(stock, last_previous_quarter)]["sales"] - 1
            revenue_growth_quarter_acceleration = (revenue_growth_current_quarter - revenue_growth_previous_quarter) / abs(revenue_growth_previous_quarter)

            profit_growth_current_quarter = trailing_quarter.loc[(stock, current_quarter)]["netProfit"] / trailing_quarter.loc[(stock, last_current_quarter)]["netProfit"] - 1
            profit_growth_previous_quarter = trailing_quarter.loc[(stock, previous_quarter)]["netProfit"] / trailing_quarter.loc[(stock, last_previous_quarter)]["netProfit"] - 1
            profit_growth_quarter_acceleration = (profit_growth_current_quarter - profit_growth_previous_quarter) / abs(profit_growth_previous_quarter)
            
#             print("Revenue growth current quarter: {:.3f}%".format(revenue_growth_current_quarter * 100))
#             print("Revenue growth previous quarter: {:.3f}%".format(revenue_growth_previous_quarter * 100))
#             print("Revenue growth quarter acceleration: {:.3f}%".format(revenue_growth_quarter_acceleration * 100))

#             print("Profit growth current quarter: {:.3f}%".format(profit_growth_current_quarter * 100))
#             print("Profit growth previous quarter: {:.3f}%".format(profit_growth_previous_quarter * 100))
#             print("Profit growth quarter acceleration: {:.3f}%".format(profit_growth_quarter_acceleration * 100))

            res = res + [revenue_growth_current_quarter, revenue_growth_previous_quarter, revenue_growth_quarter_acceleration, profit_growth_current_quarter, profit_growth_previous_quarter, profit_growth_quarter_acceleration]

            current_year = int(years[-1])
            prev_year = int(years[-2])
            prev_two_year = int(years[-3])

            revenue_growth_current_year = trailing_year.loc[(stock, current_year)]["sales"] / trailing_year.loc[(stock, prev_year)]["sales"] - 1    
            revenue_growth_previous_year = trailing_year.loc[(stock, prev_year)]["sales"] / trailing_year.loc[(stock, prev_two_year)]["sales"] - 1
            revenue_growth_year_acceleration = (revenue_growth_current_year - revenue_growth_previous_year) / abs(revenue_growth_previous_year)

            profit_growth_current_year = trailing_year.loc[(stock, current_year)]["netProfit"] / trailing_year.loc[(stock, prev_year)]["netProfit"] - 1    
            profit_growth_previous_year = trailing_year.loc[(stock, prev_year)]["netProfit"] / trailing_year.loc[(stock, prev_two_year)]["netProfit"] - 1
            profit_growth_year_acceleration = (profit_growth_current_year - profit_growth_previous_year) / abs(profit_growth_previous_year)
            
#             print("Revenue growth current year: {:.3f}%".format(revenue_growth_current_year * 100))
#             print("Revenue growth previous year: {:.3f}%".format(revenue_growth_previous_year * 100))
#             print("Revenue growth year acceleration: {:.3f}%".format(revenue_growth_year_acceleration * 100))

#             print("Profit growth current year: {:.3f}%".format(profit_growth_current_year * 100))
#             print("Profit growth previous year: {:.3f}%".format(profit_growth_previous_year * 100))
#             print("Profit growth year acceleration: {:.3f}%".format(profit_growth_year_acceleration * 100))

            res = res + [revenue_growth_current_year, revenue_growth_previous_year, revenue_growth_year_acceleration, profit_growth_current_year, profit_growth_previous_year, profit_growth_year_acceleration]

            assert(len(res) == len(columns))
            dats.append(res)
        
        except:
            print("Error of missing data for {} in {}".format(stock, current_quarter))
            continue


    df = pd.DataFrame(dats, columns = columns)    
    return df


