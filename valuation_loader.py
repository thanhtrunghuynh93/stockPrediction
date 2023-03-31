import os
import pandas as pd
import numpy as np
import requests
import yaml 
import os, time
import argparse
from datetime import datetime
from pytz import timezone
from pathlib import Path
from utils.date_util import *

stream = open("config.yaml", 'r')
config = yaml.safe_load(stream)
host = config["host"]

def get_valuation(ticker, start_date, end_date):
    
    params = {
        "ticker": ticker,        
        "start_date": start_date,        
        "end_date": end_date
    }
    
    res = requests.get(host + "/valuation/stock_valuation", params=params)

    if res.status_code != 200:
        print("Error loading price estimation for {} with status code {}".format(ticker, res.status_code))
        return None

    res = res.json()

    if "data" not in res:
        print("Error loading quarter report for {} with wrong return format".format(ticker))
        print(res)
        return None

    res = res["data"]
    df = pd.DataFrame(res)

    return df


def get_valuations(tickers, quarters):

    start_date = get_date_by_report_quarter(quarters[0])[0].strftime("%Y-%m-%d")
    end_date = get_date_by_report_quarter(quarters[-1])[1].strftime("%Y-%m-%d")

    columns = ["ticker", "name"] + quarters
    insight_cols = ["bvps", "estimate_eps", "estimated_price_high", "estimated_price_low", "high_estimate_PB", "high_estimate_PE", "low_estimate_PB", "low_estimate_PE"]

    res = []
    for ticker in tickers:            

        print("Loading intrinsic valuation for {}".format(ticker))

        df = get_valuation(ticker, start_date, end_date)

        if df is None:
            continue

        df["quarter_year"] = "Q" + df["quarter"].astype(int).astype(str) + "/" + df["year"].astype(int).astype(str)
        df = df.set_index("quarter_year")

        for insight in insight_cols:
            row = [ticker, insight]
            
            for quarter in quarters: 
                if quarter not in df.index:
                    row.append(None)
                else:
                    row.append(df.loc[quarter][insight]) 
            
            res.append(row)
        
    df = pd.DataFrame(res, columns = columns)
    return df

# def get_pe_valuation(ticker, start_date, end_date):
    
#     params = {
#         "ticker": ticker,        
#         "start_date": start_date,        
#         "end_date": end_date
#     }
    
#     res = requests.get(host + "/valuation/pe_valuation", params=params)

#     if res.status_code != 200:
#         print("Error loading price estimation for {} with status code {}".format(ticker, res.status_code))
#         return None

#     res = res.json()

#     if "data" not in res:
#         print("Error loading quarter report for {} with wrong return format".format(ticker))
#         print(res)
#         return None

#     res = res["data"]
#     df = pd.DataFrame(res)

#     return df

# def get_pb_valuation(ticker, start_date, end_date):
    
#     params = {
#         "ticker": ticker,        
#         "start_date": start_date,        
#         "end_date": end_date
#     }
    
#     res = requests.get(host + "/valuation/pb_valuation", params=params)

#     if res.status_code != 200:
#         print("Error loading price estimation for {} with status code {}".format(ticker, res.status_code))
#         return None

#     res = res.json()

#     if "data" not in res:
#         print("Error loading quarter report for {} with wrong return format".format(ticker))
#         print(res)
#         return None

#     res = res["data"]
#     df = pd.DataFrame(res)

#     return df

# def extract_quarterly_info(input_df, quarters, mode = "PE"):

#     res = {}

#     for quarter in quarters:

#         start_date, end_date = get_date_by_report_quarter(quarter)
#         start_date = start_date.strftime("%Y-%m-%d")
#         end_date = end_date.strftime("%Y-%m-%d")

#         df = input_df
#         df = df[df["date"] > start_date]
#         df = df[df["date"] < end_date]

#         if len(df) == 0:
#             res[quarter] = None            
#         else:
#             res[quarter] = df.iloc[0]
        
#     return res

# def get_pe_valuations(tickers, quarters):

#     start_date = get_date_by_report_quarter(quarters[0])[0].strftime("%Y-%m-%d")
#     end_date = get_date_by_report_quarter(quarters[-1])[1].strftime("%Y-%m-%d")

#     columns = ["ticker", "quarter", "low_estimate_price", "high_estimate_price", "low_estimate_PE", "high_estimate_PE", "eps"] 
                    
#     res = []
#     for ticker in tickers:            

#         print("Loading pe valuation for {}".format(ticker))

#         df = get_pe_valuation(ticker, start_date, end_date)
        
#         if df is None:
#             continue

#         pe_by_quarters = extract_quarterly_info(df, quarters) 
        
#         for quarter in quarters: 
#             insight = pe_by_quarters[quarter]
#             if insight is None: 
#                 res.append([ticker, quarter, None, None, None, None, None])            
#             else:
#                 low_estimate_price = insight["low_estimate_price"]
#                 high_estimate_price = insight["high_estimate_price"]
#                 low_estimate_PE = insight["low_estimate_PE"]
#                 high_estimate_PE = insight["high_estimate_PE"]
#                 eps = insight["eps"]
#                 res.append([ticker, quarter, low_estimate_price, high_estimate_price, low_estimate_PE, high_estimate_PE, eps])
        
#     df = pd.DataFrame(res, columns = columns)
#     return df

# def get_pb_valuations(tickers, quarters):

#     start_date = get_date_by_report_quarter(quarters[0])[0].strftime("%Y-%m-%d")
#     end_date = get_date_by_report_quarter(quarters[-1])[1].strftime("%Y-%m-%d")

#     columns = ["ticker", "quarter", "low_estimate_price", "high_estimate_price", "low_estimate_PB", "high_estimate_PB", "bvps"] 

#     res = []
#     for ticker in tickers:            

#         print("Loading pb valuation for {}".format(ticker))

#         df = get_pb_valuation(ticker, start_date, end_date)
        
#         if df is None:
#             continue

#         pb_by_quarters = extract_quarterly_info(df, quarters)        

#         for quarter in quarters: 
#             insight = pb_by_quarters[quarter]
#             if insight is None: 
#                 res.append([ticker, quarter, None, None, None, None, None])            
#             else:
#                 low_estimate_price = insight["low_estimate_price"]
#                 high_estimate_price = insight["high_estimate_price"]
#                 low_estimate_PE = insight["low_estimate_PB"]
#                 high_estimate_PE = insight["high_estimate_PB"]
#                 eps = insight["bvps"]
#                 res.append([ticker, quarter, low_estimate_price, high_estimate_price, low_estimate_PE, high_estimate_PE, eps])
    
#     df = pd.DataFrame(res, columns = columns)
#     return df

# def estimate_price(stock_list, stock_infos, estimate_price_pe, estimate_price_pb):

#     results = []    
#     quarters = estimate_price_pe["quarter"].unique()
    
#     estimate_price_pe = estimate_price_pe.set_index("ticker")
#     estimate_price_pb = estimate_price_pb.set_index("ticker")

#     for stock in stock_list:

#         if stock not in estimate_price_pe.index:
#             continue

#         res_lows = [stock, "estimate_price_low"]
#         res_highs = [stock, "estimate_price_high"]
#         insight_pe = estimate_price_pe.loc[stock]
#         insight_pb = estimate_price_pb.loc[stock]
#         stock_category = stock_infos.loc[stock]["category"]
                
#         for quarter in quarters:
            
#             estimate_PE_low = insight_pe[insight_pe["quarter"] == quarter]["low_estimate_price"][0]
#             estimate_PE_high = insight_pe[insight_pe["quarter"] == quarter]["high_estimate_price"][0]
            
#             estimate_PB_low = insight_pb[insight_pb["quarter"] == quarter]["low_estimate_price"][0]
#             estimate_PB_high = insight_pb[insight_pb["quarter"] == quarter]["high_estimate_price"][0]
            
#             if math.isnan(estimate_PE_low) or math.isnan(estimate_PE_high):
#                 res_lows.append(None)
#                 res_highs.append(None)            
#             else:       
#                 if stock_category in ["Banks", "Financial_Services", "Insurance"]:    
#                     pb_weight = 0.62
#                     pe_weight = 0.38
#                 else:
#                     pb_weight = 0.38
#                     pe_weight = 0.62
              
#                 if estimate_PE_low > 0:
#                     estimated_price_low = estimate_PE_low * pe_weight + estimate_PB_low * pb_weight
#                 else:
#                     estimated_price_low = estimate_PB_low

#                 if estimate_PE_high > 0:
#                     estimated_price_high = estimate_PE_high * pe_weight + estimate_PB_high * pb_weight
#                 else:
#                     estimated_price_high = estimate_PB_high

#                 res_lows.append("{:.0f}".format(estimated_price_low))
#                 res_highs.append("{:.0f}".format(estimated_price_high))

#         results.append(res_lows)
#         results.append(res_highs)
        
#     df = pd.DataFrame(results, columns = ["ticker", "name"] + list(quarters))
#     return df

def parse_args():
    parser = argparse.ArgumentParser(description="crawler")
    parser.add_argument('--stock_list', default="data/Vietnam/list_stocks.npy")
    parser.add_argument('--info_file', default="data/Vietnam/stock_infos.csv")
    parser.add_argument('--target_folder', default="data/Vietnam/")    
    parser.add_argument('--last_quarter_length', default=26)

    return parser.parse_args()

if __name__ == '__main__':

    os.environ['TZ'] = 'Asia/Saigon'
    time.tzset()
    
    args = parse_args()

    stock_list = np.load(args.stock_list, allow_pickle = True)
    stock_infos = pd.read_csv(args.info_file, index_col = "ticker")

    curr_quarter = get_current_report_quarter(string_format = True)
    quarters, years = get_last_quarters(get_next_quarter(curr_quarter), num_periods = args.last_quarter_length)

    start_date = get_date_by_report_quarter(quarters[0])[0].strftime("%Y-%m-%d")
    end_date = get_date_by_report_quarter(quarters[-1])[1].strftime("%Y-%m-%d")

    # estimate_price_pe = get_pe_valuations(stock_list, quarters)
    # estimate_price_pe.to_csv(args.target_folder + "estimate_price_pe.csv")

    # estimate_price_pb = get_pb_valuations(stock_list, quarters)
    # estimate_price_pb.to_csv(args.target_folder + "estimate_price_pb.csv")

    # price_estimator = estimate_price(stock_list, stock_infos, estimate_price_pe, estimate_price_pb)
    # price_estimator.to_csv(args.target_folder + "estimate_price.csv")

    estimate_price = get_valuations(stock_list, quarters)
    estimate_price.to_csv(args.target_folder + "estimate_price.csv")



