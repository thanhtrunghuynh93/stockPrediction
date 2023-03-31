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
default_authorization = "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSIsImtpZCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSJ9.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4iLCJhdWQiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4vcmVzb3VyY2VzIiwiZXhwIjoxOTQwMDU3NTgyLCJuYmYiOjE2NDAwNTc1ODIsImNsaWVudF9pZCI6ImZpcmVhbnQudHJhZGVzdGF0aW9uIiwic2NvcGUiOlsib3BlbmlkIiwicHJvZmlsZSIsInJvbGVzIiwiZW1haWwiLCJhY2NvdW50cy1yZWFkIiwiYWNjb3VudHMtd3JpdGUiLCJvcmRlcnMtcmVhZCIsIm9yZGVycy13cml0ZSIsImNvbXBhbmllcy1yZWFkIiwiaW5kaXZpZHVhbHMtcmVhZCIsImZpbmFuY2UtcmVhZCIsInBvc3RzLXdyaXRlIiwicG9zdHMtcmVhZCIsInN5bWJvbHMtcmVhZCIsInVzZXItZGF0YS1yZWFkIiwidXNlci1kYXRhLXdyaXRlIiwidXNlcnMtcmVhZCIsInNlYXJjaCIsImFjYWRlbXktcmVhZCIsImFjYWRlbXktd3JpdGUiLCJibG9nLXJlYWQiLCJpbnZlc3RvcGVkaWEtcmVhZCJdLCJzdWIiOiI0ZTM0MDgxYi0xNzEyLTRhOGQtYTgxOC02NWJlYTg1MWJhY2YiLCJhdXRoX3RpbWUiOjE2NDAwNTc1MzAsImlkcCI6Imlkc3J2IiwibmFtZSI6InRoYW5odHJ1bmdodXluaDkzQGdtYWlsLmNvbSIsInNlY3VyaXR5X3N0YW1wIjoiYzJlNmI0ZGQtZDUwOS00Y2I0LWFmNTYtOGM3MjU0YWYwOTc1IiwicHJlZmVycmVkX3VzZXJuYW1lIjoidGhhbmh0cnVuZ2h1eW5oOTNAZ21haWwuY29tIiwidXNlcm5hbWUiOiJ0aGFuaHRydW5naHV5bmg5M0BnbWFpbC5jb20iLCJmdWxsX25hbWUiOiJUcnVuZ0hUIiwiZW1haWwiOiJ0aGFuaHRydW5naHV5bmg5M0BnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6InRydWUiLCJqdGkiOiJhNDhiMmFhNGQxNTliYzZjY2JhN2FlMjYzY2JlOWNjMCIsImFtciI6WyJwYXNzd29yZCJdfQ.YrqVoLmfX7gn38gt4iME-u_tcAl2lsK3ZHcnA9CKS_kLWCwChYql9hl8vZCwfQGEZuIihf4n0fARruetbQIF_rzPbK_N4GaBPtfgbB038QbI-pl1HbjnVA8pGj8LcoTIyLDPfKSMQ92Ewu6fmfzEjqU2IFBNEztNQNsqXtu2d_ougO6raTGzdXQyq1zfUB-Qrtf1YhLW8iOR3x45sBDdyYAsECXSkcgoo3WTfmGakrt-WtR32dkMZBPLcPFGVwM1NXXdItRS1PcL7mw5jmBS5eIudH8vpRarRnheU3X8bY9P8i5-2PIaFZWEmEqDcK7G8B1wk9PvKCSnkjn-KV50lQ"

def getFundamental(stock, authorization = default_authorization):
    url = "https://restv2.fireant.vn/symbols/{}/fundamental".format(stock)
    headers = {'Accept-Encoding' : 'utf-8'}
    headers["authorization"] = authorization
    response = requests.get(url, headers=headers)
    data = response.json()
    
    return data

def getBasicInfo(stock, authorization = default_authorization):
    url = "https://restv2.fireant.vn/symbols/{}/financial-indicators".format(stock)
    headers = {'Accept-Encoding' : 'utf-8'}
    headers["authorization"] = authorization
    response = requests.get(url, headers=headers)
    data = response.json()
    
    return data

def getBasicFA(stock_list, authorization = default_authorization):

    needed_fund_cols = ['freeShares', 'sharesOutstanding', 'beta', 'marketCap', 'avgVolume3m', 'insiderOwnership', 'institutionOwnership', 'foreignOwnership']
    needed_FA_cols = ['P/E', 'P/S', 'P/B', 'EPS', 'ROA', 'ROE']
    columns = ["ticker"] + needed_fund_cols + needed_FA_cols

    dats = []

    for stock in stock_list:
        print(stock)
        res = [stock]    
        
        data_fund = getFundamental(stock, authorization)
        data_FA = getBasicInfo(stock, authorization)
        
        for col in needed_fund_cols:        
            if col in data_fund:
                res.append(data_fund[col])
                
        for content in data_FA:        
            if content["shortName"] in columns:
                res.append(content["value"])
                
        if len(res) != len(columns):
            print(len(res))
            print(len(columns))

        dats.append(res)
        
    df = pd.DataFrame(dats, columns = columns)
    df["institutionalOwnership"] = (df["insiderOwnership"] + df["institutionOwnership"] + df["foreignOwnership"])
    df["freeFloat"] = 1 - df["institutionalOwnership"]
    
    return df

def getFinancialReport(ticker, start_quarter = 2, start_year = 2014, end_quarter = 4, end_year = 2022, mode = "quarter"):
    
    payload = {
      "ticker": ticker,
      "start_quarter": start_quarter,
      "start_year": start_year,
      "end_quarter": end_quarter, 
      "end_year": end_year, 
      "mode": mode
    }

    endpoint = "{}/financial/financial_data".format(host)  
    res = requests.post(endpoint, json = payload)

    if res.status_code != 200:
      print("Error loading quarter report for {} with status code {}".format(ticker, res.status_code))
      return None
    
    res = res.json()

    if "data" not in res:
      print("Error loading quarter report for {} with wrong return format".format(ticker))
      print(res)
      return None

    res = res["data"]
    
    df = pd.DataFrame(res)
    df["ticker"] = ticker
    if mode == "quarter":
        df["quarter"] = df["quy"].astype(int)
    df["year"] = df["nam"].astype(int)
    
    return df

def getFinancialReports(stock_list, start_quarter = 1, start_year = 2014, end_quarter = 4, end_year = 2022, mode = "year"):
    
    quarters = generate_quarters(start_quarter, start_year, end_quarter, end_year)
    merge_df = pd.DataFrame()
    
    for stock in stock_list:        
        print("Loading financial quarter data for {}".format(stock))        
        df = getFinancialReport(stock, start_quarter, start_year, end_quarter, end_year, mode)
        merge_df = pd.concat([merge_df, df], ignore_index=True)

    return merge_df

def parse_args():
    parser = argparse.ArgumentParser(description="crawler")
    parser.add_argument('--source_file', default="data/Vietnam/list_stocks.npy")
    parser.add_argument('--info_file', default="data/Vietnam/stock_infos.csv")
    parser.add_argument('--target_folder', default="data/Vietnam")
    parser.add_argument('--interval', default="1day")    
    parser.add_argument('--crawl_new', default=False, type=bool)    

    parser.add_argument('--crawl_source', default="server")
    parser.add_argument('--start_quarter', default=1)
    parser.add_argument('--start_year', default=2014)
    parser.add_argument('--end_quarter', default=1)
    parser.add_argument('--end_year', default=2023)

    return parser.parse_args()


if __name__ == '__main__':

    os.environ['TZ'] = 'Asia/Saigon'
    time.tzset()
    
    args = parse_args()

    stock_list = np.load(args.source_file, allow_pickle = True)
    stock_infos = pd.read_csv(args.info_file, index_col = "ticker")

    # basic_FA = getBasicFA(stock_list)
    # basic_FA.to_csv(args.target_folder + "/basic_FA.csv")

    # trailing_quarter_report = getFinancialReports(stock_list, args.start_quarter, args.start_year, args.end_quarter, args.end_year, mode = "quarter")
    # trailing_quarter_report.to_csv(args.target_folder + "/trailing_quarter_report.csv")
    
    trailing_year_report = getFinancialReports(stock_list, args.start_quarter, args.start_year, args.end_quarter, args.end_year, mode = "year")    
    trailing_year_report.to_csv(args.target_folder + "/trailing_year_report.csv")


    

