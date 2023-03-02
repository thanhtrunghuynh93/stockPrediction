import matplotlib.pyplot as plt
from utils.date_util import get_most_recent_trade_day, get_current_report_quarter, get_last_quarters, get_last_quarter, get_date_by_report_quarter, generate_quarters
import matplotlib.pyplot as plt

class FA_analysis:

    def __init__(self, stock_infos, basicFA, trailing_quarter):

        self.stock_infos = stock_infos
        self.basicFA = basicFA
        self.trailing_quarter = trailing_quarter

    def calculate_indices(self, stock, stock_data):

        start_quarter = get_current_report_quarter(stock_data.index[0])
        end_quarter = get_current_report_quarter(stock_data.index[-1])
        trade_quarters = generate_quarters(start_quarter[0], start_quarter[1], end_quarter[0], end_quarter[1])
        NetProfit_index = list(self.trailing_quarter.loc[stock]["Symbol"].values).index("NetProfit")
        num_stock = self.basicFA.loc[stock]["sharesOutstanding"]

        eps_quarter = {}
        book_value_quarter = {}

        for quarter in trade_quarters:
            #Calculate infos by quarter
            last_quarters = get_last_quarters(quarter, 4)[0]
            last_quarter = last_quarters[-1]

            #Calculate eps, epsg, bv
            eps = self.trailing_quarter.loc[stock][last_quarters].values[NetProfit_index].sum() / num_stock
            eps_quarter[last_quarter] = eps
            
            last_quarters = get_last_quarters(last_quarters[3], 4)[0]    
            epsg = eps / (self.trailing_quarter.loc[stock][last_quarters].values[NetProfit_index].sum() / num_stock)    
            # epsgs_quarter[last_quarter] = epsg

            book_value = self.trailing_quarter.loc[(self.trailing_quarter.index == stock) & (self.trailing_quarter["Symbol"] == "BookValue")][last_quarter].values[0]    
            book_value_quarter[last_quarter] = book_value

            start_date, end_date = get_date_by_report_quarter(quarter)
            insight = stock_data[stock_data.index >= start_date]
            insight = insight[insight.index <= end_date]

            stock_data.loc[insight.index, "eps"] = eps
            stock_data.loc[insight.index, "eps_growth"] = epsg
            stock_data.loc[insight.index, "quarter"] = quarter
            stock_data.loc[insight.index, "book_value"] = book_value

        stock_data["PE"] = stock_data["close"] / stock_data["eps"]
        stock_data["PB"] = stock_data["close"] / stock_data["book_value"] * num_stock  

        stock_data = stock_data.fillna(0)

        return stock_data, eps_quarter, book_value_quarter

    def plot(self, stock, source_df):
        
        plt.figure(figsize=(16,4))
        plt.title(stock)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('PE', fontsize=18)
        # plt.plot(train['close'])
        plt.plot(source_df[['PE']])
        plt.show()

        plt.figure(figsize=(16,4))
        plt.title(stock)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('PB', fontsize=18)
        # plt.plot(train['close'])
        plt.plot(source_df[['PB']])
        plt.show()

        plt.figure(figsize=(16,4))
        plt.title(stock)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        # plt.plot(train['close'])
        plt.plot(source_df[['close']])
        plt.show()

        plt.figure(figsize=(16,4))
        plt.title(stock)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('EPS', fontsize=18)
        # plt.plot(train['close'])
        plt.plot(source_df[['eps']])
        plt.show()

        plt.figure(figsize=(16,4))
        plt.title(stock)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('EPSG', fontsize=18)
        # plt.plot(train['close'])
        plt.plot(source_df[['eps_growth']])
        plt.show()





