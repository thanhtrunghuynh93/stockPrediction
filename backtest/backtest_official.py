import numpy as np

from backtest.portfolio import Portfolio
from backtest.transaction_history import TransactionHistory
from backtest.filters import FilterFactory
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

"""Args:
    trade_sessions: List of backtest trade index (datetime)
    env_df: A dataframe containing of all necessary data
    pred: prediction result from the model in numpy array, the order is as the order from env_df 
"""

def prepare_env_df(df_storage):

    env_df = pd.DataFrame()
    for df in df_storage:    
        # df = df.filter(needed_columns)
        df = df.reset_index()
        env_df = pd.concat([env_df, df], ignore_index=True)

    return env_df

def prepare_env_df_datetime_indexing(env_df, indices):
    env_df_indexing = {}    
    for idx in indices:
        env_df_indexing[idx] = env_df[env_df["datetime"] == idx]
    
    return env_df_indexing


class Backtest():

    def __init__(self, env_df, preds, filters = [], max_number_of_stock = 5, commission = 0.0015, buy_threshold = 0.08, sell_threshold = 0.04, stop_loss = 0.07, initial_budget = 1000000000, min_num_share_per_purchase = 100, t_delay = 3):
        
        self.env_df = env_df
        self.preds = preds        
        self.max_number_of_stock = max_number_of_stock
        self.commission = commission 
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss = stop_loss
        self.initial_budget = initial_budget 
        self.min_num_share_per_purchase = min_num_share_per_purchase
        self.t_delay = t_delay

        self.trade_sessions = self.env_df["datetime"].unique()
        self.ticker_symbols = self.env_df["symbol"].unique()
        self.ticker_symbols_idx = {}
        for idx in range(len(self.ticker_symbols)):
            ticker = self.ticker_symbols[idx]
            self.ticker_symbols_idx[ticker] = idx

        self.env_df_indexing = prepare_env_df_datetime_indexing(env_df, self.trade_sessions)
        self.prices = env_df["close"].to_numpy().reshape(len(self.ticker_symbols), -1)
        self.filters = FilterFactory(filters, env_df, self.trade_sessions, self.ticker_symbols)
        

        self.portfolio_storage = []
        self.transaction_history_storage = []
        self.final_num_win_trades = []
        self.final_num_lose_trades = []
        self.final_max_win_trades = []
        self.final_max_lose_trades = []
        
        self.portfolio_storage = []
        self.transaction_history_storage = []

        self.trade_data = []
        self.hold_data = []
        self.final_trade_net_values = []
        self.final_sharp_ratios = []
        self.final_gain_ratios = []

    def execute(self, num_test = 100, print_per = 10):

        # len(train_trade_sessions)
        for t in tqdm(range(num_test)):
            
            if t % print_per == 0:
                print("Test {}".format(t))
            
            portfolio = Portfolio(self.initial_budget, self.commission)
            transaction_history = TransactionHistory()
            
            start_prices = self.env_df[self.env_df["datetime"] == self.trade_sessions[0]]["close"]
            average_hold = self.initial_budget / start_prices.sum()
            
            trades = []
            holds = []

            for day in range(len(self.trade_sessions)):
                
                current_date = self.trade_sessions[day]        
                pred = self.preds[day, :]
                
                current_data = self.env_df[self.env_df["datetime"] == current_date]
                current_prices = current_data[["symbol", "close"]]
                current_prices = current_prices.set_index("symbol")
                
                portfolio.update_status(current_date, current_prices)        
                hold_stocks = portfolio._portfolio.index
                
                #Check portfolio and sell
                for stock in hold_stocks:
                    
                    stock_data = portfolio._portfolio.loc[stock]
                    avail = stock_data["avail"]
                    stock_idx = self.ticker_symbols_idx[stock]
                    current_price = current_prices.loc[stock, "close"]
                    max_record_price = stock_data["max_record_close_price"]                               
                    
                    
                    if avail > 0: 
                        # Sell by trailing stop loss
                        if current_price < max_record_price * (1 - self.stop_loss):
                            
                            size_to_sell = avail 
                            bought_avg_price = stock_data["bought_avg_price"]                            
                            sold_avg_price = current_price * (1 - self.commission) 
                            profit_rate = sold_avg_price / bought_avg_price - 1
                            # bought_value = portfolio._portfolio.loc[stock, "bought_avg_price"] * size_to_sell
                            # sold_value = size_to_sell * sold_price
                            # commission_value = sold_value * self.commission
                            # avg_sold_price = sold_price * (1 - self.commission) 
                            # profit = sold_value - commission_value - bought_value
                            
                            
                            portfolio.update_status_sell(current_date, current_price, size_to_sell, stock)
                            transaction_history.append(current_date, stock, "sell", size_to_sell, bought_avg_price, bought_avg_price, current_price, sold_avg_price, profit_rate, note = "trailing_stop")
                            
                                            
                        elif pred[stock_idx] < -self.sell_threshold:
                            
                            size_to_sell = avail 
                            bought_avg_price = stock_data["bought_avg_price"]                            
                            sold_avg_price = current_price * (1 - self.commission) 
                            profit_rate = sold_avg_price / bought_avg_price - 1
                            
                            portfolio.update_status_sell(current_date, current_price, size_to_sell, stock)
                            transaction_history.append(current_date, stock, "sell", size_to_sell, bought_avg_price, bought_avg_price, current_price, sold_avg_price, profit_rate, note = "prediction")
                            # transaction_history.append(current_date, stock, "sell", size_to_sell, bought_price, bought_value, sold_price, sold_value, commission_value, profit, profit_rate, note = "prediction")
                        
        #                 elif current_prices[stock] > portfolio._portfolio[stock]["bought_avg_price"] * (1 + take_profit):
                            
                            
                hold_stocks = portfolio._portfolio.index
                # Check portfolio and buy   
                max_num_stock_to_buy = self.max_number_of_stock - len(hold_stocks)
                
                candidate_stocks = np.where(pred >= self.buy_threshold)[0]
                filter_cache = self.filters.cache[day]
                candidate_stocks = np.intersect1d(candidate_stocks, filter_cache)
                candidate_stocks = self.ticker_symbols[candidate_stocks]                                
                already_hold = np.intersect1d(candidate_stocks, hold_stocks)
                candidate_stocks = np.setdiff1d(candidate_stocks, already_hold)
                np.random.shuffle(candidate_stocks)
                        
                for i in range(min(max_num_stock_to_buy, len(candidate_stocks))):        

                    candidate = candidate_stocks[i]
                    budget_for_stock = portfolio._budget / (max_num_stock_to_buy - i)
                    current_price = current_prices.loc[candidate, "close"]
                    
                    size_to_buy = (int)(budget_for_stock / current_price / self.min_num_share_per_purchase) * self.min_num_share_per_purchase        
                    if size_to_buy == 0:
                        break
                        
                    avg_bought_price = current_price * (1 + self.commission)
                    # avg_bought_value = avg_bought_price * size_to_buy
                    # commission_value = current_price * self.commission * size_to_buy
                    
                    portfolio.update_status_buy(current_date, current_price, size_to_buy, candidate)
                    
                    transaction_history.append(current_date, candidate, "buy", size_to_buy, current_price, avg_bought_price, None, None, None, "prediction")
                        
                # Update value
                trade_net_value = portfolio.get_net_value()
                budget_if_hold = average_hold * current_prices.values.sum()
                gain_ratio = trade_net_value / budget_if_hold
                sharp_ratio = trade_net_value / self.initial_budget

                trades.append(trade_net_value)
                holds.append(budget_if_hold)
                
            data = transaction_history.to_dataframe()            
            data = data[data["type"] == "sell"]
            num_win_trades = len(data[data["profit_rate"] > 0])
            num_lose_trades = len(data[data["profit_rate"] < 0])
            max_win_trade = len(data[data["profit_rate"] < 0])
            max_win_trade = max(data["profit_rate"])
            max_lose_trade = min(data["profit_rate"]) 
            
            self.final_num_win_trades.append(num_win_trades)
            self.final_num_lose_trades.append(num_lose_trades)
            self.final_max_win_trades.append(max_win_trade)
            self.final_max_lose_trades.append(max_lose_trade)    
            
            self.portfolio_storage.append(portfolio)
            self.transaction_history_storage.append(transaction_history)

            self.trade_data.append(trades)
            self.hold_data.append(holds)
            self.final_trade_net_values.append(trade_net_value)
            self.final_sharp_ratios.append(sharp_ratio)
            self.final_gain_ratios.append(gain_ratio)

            
    def statistic(self):
        print("Best trade")
        idx = np.argmax(self.final_sharp_ratios)
        print("Sharp ratio: {}".format(self.final_sharp_ratios[idx]))
        print("Gain ratio: {}".format(self.final_gain_ratios[idx]))
        total_trade = self.final_num_win_trades[idx] + self.final_num_lose_trades[idx]
        print("Total trade: {}".format(total_trade))
        win_rate = self.final_num_win_trades[idx] / total_trade
        print("Win rate: {}".format(win_rate))
        print("Max profit: {}".format(self.final_max_win_trades[idx]))
        print("Max lost: {}".format(self.final_max_lose_trades[idx]))

        print("Worst trade")
        idx = np.argmin(self.final_sharp_ratios)
        print("Sharp ratio: {}".format(self.final_sharp_ratios[idx]))
        print("Gain ratio: {}".format(self.final_gain_ratios[idx]))
        total_trade = self.final_num_win_trades[idx] + self.final_num_lose_trades[idx]
        print("Total trade: {}".format(total_trade))
        win_rate = self.final_num_win_trades[idx] / total_trade
        print("Win rate: {}".format(win_rate))
        print("Max profit: {}".format(self.final_max_win_trades[idx]))
        print("Max lost: {}".format(self.final_max_lose_trades[idx]))

        print("Mean")
        print(np.mean(self.final_sharp_ratios))

    def visualize(self, idx):

        result = pd.DataFrame(data={"hold_budget" : self.hold_data[idx], "trade_budget" : self.trade_data[idx], "datetime" : self.trade_sessions})
        result.index = pd.DatetimeIndex(result['datetime'])

        plt.figure(figsize=(16,6))
        plt.title("Best result")
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Budget', fontsize=18)
        # plt.plot(train['close'])
        plt.plot(result[['hold_budget', 'trade_budget']])
        plt.legend(['Buy and hold', 'Trade'], loc='upper left')
        plt.show()

    

