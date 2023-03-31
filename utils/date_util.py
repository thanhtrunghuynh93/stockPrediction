import os, time
from datetime import datetime, timedelta
from pytz import timezone
import pandas as pd
import math

def get_most_recent_trade_day(current_date = None):
        
    difference_day = 0
    
    if current_date is None:
        current_date = datetime.now().replace(tzinfo=None) 
    if type(current_date) is str:
        print("Warning: current_date is string, use datetime.strptime function with format '%Y-%m-%d %H:%M:%S'!")
        current_date = datetime.strptime(current_date, '%Y-%m-%d %H:%M:%S')
    if current_date.weekday() > 4:
        difference_day = (current_date.weekday() - 4)
        current_date = current_date - timedelta(days = difference_day)
    
    current_date = current_date.strftime('%Y-%m-%d ') + "09:00:00"
    current_date = datetime.strptime(current_date, '%Y-%m-%d %H:%M:%S') 
    
    return current_date, difference_day

def get_previous_trade_day(current_date = None, delta = 20):
    
    if delta < 0:
        raise Exception("ERROR: delta must be greater than 0 !")
    if type(current_date) is str:
        print("Warning: current_date is string, use datetime.strptime function with format '%Y-%m-%d %H:%M:%S'!")
        current_date = datetime.strptime(current_date, '%Y-%m-%d %H:%M:%S')
    
    res, difference_day = get_most_recent_trade_day(current_date)
    
    #Separate delta into weeks and days
    num_weeks = int(delta / 5)
    num_days = delta % 5
    
    if num_weeks > 0:
        res = res - timedelta(days = num_weeks * 7) 
        difference_day += num_weeks * 7
        
    if num_days > 0:
        res = res - timedelta(days = num_days)
        difference_day += num_days
        
        res, diff = get_most_recent_trade_day(res)
        difference_day += diff    
    
    return res, difference_day

def format_str_datetime(str_date):
    date = str(pd.to_datetime(str_date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return date.strftime('%Y-%m-%d')

def add_days_to_string_time(str_date, days=1):
    date = str(pd.to_datetime(str_date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    new_date = date + timedelta(days=days)
    return new_date.strftime('%Y-%m-%d')

def _get_start_date_with_offset(start_date, offset):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')

    weekdays = [0, 1]
    count = 0
    days = timedelta(1)
    while count != offset:
        start_date = start_date - days
        if start_date.weekday() not in weekdays: 
            count += 1
    return start_date.strftime('%Y-%m-%d')
        

def generate_date_index(start_date, periods):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    list_index = []
    weekdays = [0, 1]
    count = 0
    days = timedelta(1)
    while count != periods:
        start_date = start_date + days
        if start_date.weekday() not in weekdays: 
            count += 1
            list_index.append(start_date.strftime('%Y-%m-%d'))
    return list_index


def days_between(start_time, end_time):
    d1 = str(pd.to_datetime(start_time))
    d2 = str(pd.to_datetime(end_time))
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    return (d2-d1).days


def get_data_by_symbol_and_date(symbol, start_date, end_date, offset=0):
    if offset:
        start_date = _get_start_date_with_offset(start_date, offset)
    params = {
        "ticker": symbol,
        "start_date": start_date,
        "end_date": end_date
    }
    res = requests.post("http://192.168.67.129:9997/data/daily_history", params=params)
    # res = requests.post("http://202.191.57.62:9997/data/daily_history", params=params)
    data = res.json()
    if 'data' in data:
        data = data["data"]
    else:
        print("Error when fetching data!")
        exit()
    df_data = pd.DataFrame(data)
    df_data.index = pd.to_datetime(df_data['date'])
    df_data.index = df_data.index.map(str)
    df_data = df_data.drop(columns=['symbol', 'date'])
    df_data = df_data.astype(float)
    return df_data

def generate_quarters(start_quarter = 3, start_year = 2016, end_quarter = 2, end_year = 2022):

    if (start_quarter > end_quarter) and (start_year >= end_year):
        print("Wrong input for generate quarters")
        return []

    current_quarter = start_quarter
    current_year = start_year
    quarters = []
    
#     while ((current_quarter < end_quarter) and (current_year < end_year)):
    while True :        
        quarters.append("Q{}/{}".format(current_quarter, current_year))
        current_quarter += 1        
        if current_quarter == 5:
            current_quarter = 1
            current_year += 1
        if current_quarter == end_quarter and current_year == end_year:
            quarters.append("Q{}/{}".format(current_quarter, current_year))
            break
    return quarters

def get_current_report_quarter(date = None, string_format = False):
    
    if date is None: 
        date = get_most_recent_trade_day(datetime.now())[0]

    date = str(pd.to_datetime(date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    #Determine the quarter of the last report    
    if date.month == 1:
        q = 4
        y = date.year - 1
    else:
        q = math.ceil((date.month - 1) / 3)
        y = date.year
    if string_format:
        quarter = "Q{}/{}".format(q, y)
        return quarter
    else:
        return q, y

def get_last_quarter(current_quarter):
    curr_quarter = int(current_quarter.split("/")[0][1])
    curr_year = int(current_quarter.split("/")[1])    
    
    if curr_quarter == 1:
        res_quarter = 4
        res_year = curr_year - 1
    else:
        res_quarter = curr_quarter - 1
        res_year = curr_year
    
    return "Q{}/{}".format(res_quarter, res_year)

def get_next_quarter(current_quarter):
    curr_quarter = int(current_quarter.split("/")[0][1])
    curr_year = int(current_quarter.split("/")[1])    
    
    if curr_quarter == 4:
        res_quarter = 1
        res_year = curr_year + 1
    else:
        res_quarter = curr_quarter + 1
        res_year = curr_year
    
    return "Q{}/{}".format(res_quarter, res_year)

def get_last_quarters(current_quarter, num_periods = 12):
    
    res_quarters = []
    res_years = []
    
    curr_quarter = current_quarter
    # res_years.insert(0, curr_quarter.split("/")[1])
    for i in range(num_periods):
        curr_quarter = get_last_quarter(curr_quarter)
        res_quarters.insert(0, curr_quarter)
        
        if int(curr_quarter.split("/")[0][1]) == 4:
            res_years.insert(0, curr_quarter.split("/")[1])
        
    # res_years = np.unique(res_years)    
    return res_quarters, res_years    

def get_date_by_report_quarter(current_quarter):
    curr_quarter = int(current_quarter.split("/")[0][1])
    curr_year = int(current_quarter.split("/")[1])    

    start_month = curr_quarter * 3 - 1
    end_month = curr_quarter * 3 + 1

    start_year = curr_year
    end_year = curr_year 
    
    if curr_quarter == 4:
        end_month = 1
        end_year += 1
       
    start_date = datetime.strptime("{}-{}-01 09:00:00".format(start_year, start_month), '%Y-%m-%d %H:%M:%S')
    if end_month == 1 or end_month == 7 or end_month == 10:
        end_date = datetime.strptime("{}-{}-31 09:00:00".format(end_year, end_month), '%Y-%m-%d %H:%M:%S')
    elif end_month == 4:
        end_date = datetime.strptime("{}-{}-30 09:00:00".format(end_year, end_month), '%Y-%m-%d %H:%M:%S')
    else:
        print("ERROR:", end_month)

    return start_date, end_date
        