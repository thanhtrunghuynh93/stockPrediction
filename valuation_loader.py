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

def get_pe_valuation(ticker, start_date, end_date):
    
    params = {
        "ticker": ticker,        
        "start_date": start_date,        
        "end_date": end_date
    }
    
    res = requests.get(host + "/valuation/pe_valuation", params=params)

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

def get_pb_valuation(ticker, start_date, end_date):
    
    params = {
        "ticker": ticker,        
        "start_date": start_date,        
        "end_date": end_date
    }
    
    res = requests.get(host + "/valuation/pb_valuation", params=params)

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

def extract_quarterly_info(input_df, quarters):

    res = {}

    for quarter in quarters:

        start_date, end_date = get_date_by_report_quarter(quarter)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        df = input_df
        df = df[df["date"] > start_date]
        df = df[df["date"] < end_date]

        if len(df) == 0:
            price_low = 0
            price_high = 0
        else:
            price_low, price_high = df.iloc[0]["price_low"], df.iloc[0]["price_high"]

        res[quarter] = "{:.0f}, {:.0f}".format(price_low, price_high)

    return res

def get_pe_valuations(tickers, quarters):

    start_date = get_date_by_report_quarter(quarters[0])[0].strftime("%Y-%m-%d")
    end_date = get_date_by_report_quarter(quarters[-1])[1].strftime("%Y-%m-%d")

    columns = ["ticker"] + quarters

    res = []
    for ticker in tickers:            

        result = [ticker]

        print("Loading pe valuation for {}".format(ticker))

        df = get_pe_valuation(ticker, start_date, end_date)
        
        if df is None:
            continue

        pe_by_quarters = extract_quarterly_info(df, quarters)        

        for quarter in quarters: 
            result.append(pe_by_quarters[quarter])

        res.append(result)
    
    df = pd.DataFrame(res, columns = columns)
    return df

def get_pb_valuations(tickers, quarters):

    start_date = get_date_by_report_quarter(quarters[0])[0].strftime("%Y-%m-%d")
    end_date = get_date_by_report_quarter(quarters[-1])[1].strftime("%Y-%m-%d")

    columns = ["ticker"] + quarters

    res = []
    for ticker in tickers:            

        result = [ticker]

        print("Loading pb valuation for {}".format(ticker))

        df = get_pb_valuation(ticker, start_date, end_date)
        
        if df is None:
            continue

        pb_by_quarters = extract_quarterly_info(df, quarters)        

        for quarter in quarters: 
            result.append(pb_by_quarters[quarter])

        res.append(result)
    
    df = pd.DataFrame(res, columns = columns)
    return df

def estimate_price(stock_list, stock_infos, estimate_price_pe, estimate_price_pb):

    results = []
    quarters = estimate_price_pe.columns[1:]

    for stock in stock_list:

        if stock not in estimate_price_pe.index:
            continue

        res = [stock]
        estimate_price_pes = estimate_price_pe.loc[stock]
        estimate_price_pbs = estimate_price_pb.loc[stock]
        stock_category = stock_infos.loc[stock]["category"]
        
        for quarter in quarters:
            estimate_PE_range = estimate_price_pes[quarter]
            estimate_PB_range = estimate_price_pbs[quarter]
            
            estimate_PE_low = float(estimate_PE_range.split(',')[0])
            estimate_PE_high = float(estimate_PE_range.split(',')[1])
            
            estimate_PB_low = float(estimate_PB_range.split(',')[0])
            estimate_PB_high = float(estimate_PB_range.split(',')[1])
            
            if stock_category in ["Banks", "Financial_Services", "Insurance"]:    
                pb_weight = 0.62
                pe_weight = 0.38
            else:
                pb_weight = 0.38
                pe_weight = 0.62

            if estimate_PE_low > 0:
                estimated_price_low = estimate_PE_low * pe_weight + estimate_PB_low * pb_weight
            else:
                estimated_price_low = estimate_PB_low

            if estimate_PE_high > 0:
                estimated_price_high = estimate_PE_high * pe_weight + estimate_PB_high * pb_weight
            else:
                estimated_price_high = estimate_PB_high

            res.append("{:.0f}, {:.0f}".format(estimated_price_low, estimated_price_high))
        results.append(res)
        
    df = pd.DataFrame(results, columns = ["ticker"] + list(quarters))
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="crawler")
    parser.add_argument('--stock_list', default="data/Vietnam/list_stocks.npy")
    # parser.add_argument('--info_file', default="data/Vietnam/stock_infos.csv")
    parser.add_argument('--target_folder', default="data/Vietnam/")    
    parser.add_argument('--last_quarter_length', default=26)

    return parser.parse_args()

if __name__ == '__main__':

    os.environ['TZ'] = 'Asia/Saigon'
    time.tzset()
    
    args = parse_args()

    stock_list = np.load(args.stock_list, allow_pickle = True)
    curr_quarter = get_current_report_quarter(string_format = True)
    quarters, years = get_last_quarters(get_next_quarter(curr_quarter), num_periods = args.last_quarter_length)

    start_date = get_date_by_report_quarter(quarters[0])[0].strftime("%Y-%m-%d")
    end_date = get_date_by_report_quarter(quarters[-1])[1].strftime("%Y-%m-%d")

    print(start_date, end_date)
    
    estimate_price_pe = get_pe_valuations(stock_list, quarters)
    estimate_price_pe.to_csv(args.target_folder + "estimate_price_pe.csv")

    estimate_price_pb = get_pb_valuations(stock_list, quarters)
    estimate_price_pb.to_csv(args.target_folder + "estimate_price_pb.csv")



