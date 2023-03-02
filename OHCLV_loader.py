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

stream = open("config.yaml", 'r')
config = yaml.safe_load(stream)
host = config["host"]

def load_data(ticker, start_data_date, end_data_date = None, mode = "daily"):

    if end_data_date is None:
        # now = datetime.now(timezone('Europe/Berlin')).replace(tzinfo=None) 
        # end_data_date = datetime.strptime(now, '%Y-%m-%d') 
        payload = {"ticker" : ticker, "date_start" : start_data_date, "mode" : mode}  
    else:
        payload = {"ticker" : ticker, "date_start" : start_data_date, "date_end" : end_data_date, "mode" : mode}  
    endpoint = "{}/data/stock_price".format(host)
    res = requests.get(endpoint, params = payload).json().get('data')    
    df = pd.DataFrame(res)
    df = df.set_index("date")

    return df

def crawl(stock_infos, stock_list, output_dir, crawl_new, interval, crawl_source, start_date):

    if crawl_source != "server":        
        print("Crawl source not support")
        return

    # stock_list.append("VNINDEX")
    if interval == "1day":
        crawl_by_day(stock_infos, stock_list, output_dir, crawl_new, crawl_source, start_date)
    else:
        print("Interval not support")
        return

def crawl_by_day(stock_infos, stock_list, output_dir, crawl_new, crawl_source, start_date):

    interval = "1day"
    now = datetime.now(timezone('Asia/Saigon')).replace(tzinfo=None) 
    print("Crawling time ", now)
    num_try = 3

    for code in stock_list:
        print(code)
        if code == "VNINDEX":
            exchange = "HOSE"
        else:
            exchange = stock_infos.loc[code].exchange
        start_time = time.time()

        if Path(output_dir + "/" + code + "_" + interval + ".csv").is_file() and not crawl_new:
            
            try:
                #Load the existing dataset, append and save
                data = pd.read_csv(output_dir + "/" + code + "_" + interval + ".csv", index_col = "date")
                
            except:
                print("Error: Empty data file, crawl new data")

                if crawl_source == "server":
                    data = load_data(ticker = code, start_data_date = start_date)
                    data["symbol"] = exchange + ":" + code
                    
                print("--- %s seconds ---" % (time.time() - start_time))
                print("Create new {} records".format(data.shape[0]))                            
                if data is None:        
                    print("Error")
                else:
                    data.to_csv(output_dir + "/" + code + "_" + interval + ".csv")            
                continue

            last_date = datetime.strptime(str(data.tail(1).index.values[0]), '%Y-%m-%d')               
            difference_day = (now - last_date).days
            last_date = last_date.strftime('%Y-%m-%d')

            #Check if today is weekend
            if now.weekday() > 4:
                difference_day = difference_day - (now.weekday() - 4)

            if difference_day == 0:
                for i in range(num_try):
                    try:
                        print("The data is up-to-date, update only the data today")                        
                        if crawl_source == "server":                            
                            new_data = load_data(ticker = code, start_data_date = last_date)
                            new_data["symbol"] = exchange + ":" + code
                            
                        data.loc[data.index[-1]] = new_data.values[0]                    
                        data.to_csv(output_dir + "/" + code + "_" + interval + ".csv")            
                        print("--- %s seconds ---" % (time.time() - start_time))                                
                        break
                    except:
                        print("Error, try again")            
            else:                
                for i in range(num_try):
                    try:                 
                        if crawl_source == "server":
                            new_data = load_data(ticker = code, start_data_date = last_date)
                            new_data["symbol"] = exchange + ":" + code

                        new_data = new_data[new_data.index > last_date]
                        print("Found new {} records".format(len(new_data)))
                        if len(new_data) == 0:
                            print("Error")
                        else:                    
                            data = pd.concat((data, new_data), join='outer', copy=True)
                            data.to_csv(output_dir + "/" + code + "_" + interval + ".csv")            
                            print("--- %s seconds ---" % (time.time() - start_time))                  
                            break              
                    except:
                        print("Error, try again")            

        else:
            print("Crawl new data")
            data = load_data(ticker = code, start_data_date = start_date)
            data["symbol"] = exchange + ":" + code
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Create new {} records".format(data.shape[0]))                            
            if data is None:        
                print("Error")
            else:
                data.to_csv(output_dir + "/" + code + "_" + interval + ".csv")            
            continue

def parse_args():
    parser = argparse.ArgumentParser(description="crawler")
    parser.add_argument('--source_file', default="data/Vietnam/list_stocks.npy")
    parser.add_argument('--info_file', default="data/Vietnam/stock_infos.csv")
    parser.add_argument('--target_folder', default="data/Vietnam/OHCLV")
    parser.add_argument('--interval', default="1day")    
    parser.add_argument('--crawl_new', default=False, type=bool)    

    parser.add_argument('--crawl_source', default="server")
    parser.add_argument('--start_date', default="2017-02-01")

    return parser.parse_args()


if __name__ == '__main__':

    os.environ['TZ'] = 'Asia/Saigon'
    time.tzset()
    
    args = parse_args()

    stock_list = np.load(args.source_file, allow_pickle = True)
    stock_infos = pd.read_csv(args.info_file, index_col = "ticker")

    crawl(stock_infos, stock_list.tolist(), args.target_folder, args.crawl_new, args.interval, args.crawl_source, args.start_date)
    # crawl(crawler, stock_infos, stock_list.tolist(), args.interval, args.nbars, args.target_folder, args.crawl_new)
    



