import numpy as np

from backtest.portfolio import Portfolio
from backtest.transaction_history import TransactionHistory
from backtest.filters import FilterFactory
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.date_util import format_str_datetime
from preprocess.dataloader import prepare_env_df, prepare_env_df_datetime_indexing

"""Args:
    trade_sessions: List of backtest trade index (datetime)
    env_df: A dataframe containing of all necessary data
    pred: prediction result from the model in numpy array, the order is as the order from env_df 
"""

class Backtest():

    def __init__(self, env_df, stock_info, preds, filters = [], max_number_of_stock = 5, commission = 0.0015, buy_thresholds = None, sell_thresholds = None, stop_loss = 0.07, initial_budget = 1000000000, min_num_share_per_purchase = 100, t_delay = 3, setting = "regression", ta_score_threshold = 4, intrinsic_mos = 0.4):
        
        self.env_df = env_df
        self.stock_info = stock_info
        self.preds = preds        
        self.max_number_of_stock = max_number_of_stock
        self.commission = commission 
        self.buy_thresholds = buy_thresholds
        self.sell_thresholds = sell_thresholds
        self.stop_loss = stop_loss
        self.initial_budget = initial_budget 
        self.min_num_share_per_purchase = min_num_share_per_purchase
        self.t_delay = t_delay
        self.setting = setting
        # self.is_unique_category_in_portfolio = is_unique_category_in_portfolio

        self.trade_sessions = self.env_df["date"].unique()
        self.ticker_symbols = self.env_df["symbol"].unique()
        self.tickers = self.env_df["ticker"].unique()
        self.ticker_symbols_idx = {}
        for idx in range(len(self.ticker_symbols)):
            ticker = self.ticker_symbols[idx]
            self.ticker_symbols_idx[ticker] = idx

        self.env_df_indexing = prepare_env_df_datetime_indexing(env_df, self.trade_sessions)
        self.prices = env_df["close"].to_numpy().reshape(len(self.ticker_symbols), -1)
        self.high_prices = env_df["high"].to_numpy().reshape(len(self.ticker_symbols), -1)
        self.open_prices = env_df["open"].to_numpy().reshape(len(self.ticker_symbols), -1)

        self.filter = FilterFactory(filters, env_df, stock_info, ta_score_threshold = ta_score_threshold, intrinsic_mos = intrinsic_mos)

        self.transaction_history_storage = []
        self.final_num_win_trades = []
        self.final_num_lose_trades = []
        self.final_max_win_trades = []
        self.final_max_lose_trades = []
        
        self.logs = []
        self.transaction_history_storage = []

        self.trade_data = []
        self.hold_data = []
        self.final_trade_net_values = []
        self.final_sharp_ratios = []
        self.final_gain_ratios = []

    def execute(self, num_test = 100, print_per = 10):
        if self.setting == "regression":
            self.execute_regression(num_test, print_per)
        elif self.setting == "classification":
            self.execute_classification(num_test, print_per)
        else:
            pass

    def execute_regression(self, num_test = 100, print_per = 10):

        # len(train_trade_sessions)
        for t in tqdm(range(num_test)):
            
            if t % print_per == 0:
                print("Test {}".format(t))
                        
            held_stocks = np.zeros(len(self.ticker_symbols))
            held_prices = np.zeros(len(self.ticker_symbols))
            held_day = np.zeros(len(self.ticker_symbols))
            max_held_prices = np.zeros(len(self.ticker_symbols))
            budget = self.initial_budget
            log = []
            
            transaction_history = TransactionHistory()

            start_prices = self.prices[:, 0]
            average_hold = self.initial_budget / start_prices.sum()
            
            trades = []
            holds = []
            candidate_stocks = []

            for day in range(len(self.trade_sessions)):
                
                current_date = self.trade_sessions[day]   
                log.append("==========================================================")     
                log.append("Day {}".format(format_str_datetime(current_date)))
                pred = self.preds[day, :]
                
                # current_data = self.env_df_indexing[current_date]
                # current_prices = current_data["close"].to_numpy()

                current_prices = self.prices[:, day]
                high_prices = self.high_prices[:, day]

                if day > 0:

                    open_prices = self.open_prices[:, day]                    
                    hold_stocks = np.where(held_stocks > 0)[0]                    

                    # Check portfolio and buy   
                    max_num_stock_to_buy = self.max_number_of_stock - len(hold_stocks)
                
                    for i in range(min(max_num_stock_to_buy, len(candidate_stocks))):        

                        candidate = candidate_stocks[i]
                        ticker = self.ticker_symbols[candidate]
                        budget_for_stock = budget / (max_num_stock_to_buy - i) / (1 + self.commission)
                        current_price = open_prices[candidate]
                        
                        size_to_buy = (int)(budget_for_stock / current_price / self.min_num_share_per_purchase) * self.min_num_share_per_purchase        
                        if size_to_buy == 0:
                            break
                        
                        bought_price = current_price                    
                        avg_bought_price = (bought_price * size_to_buy * (1 + self.commission) + held_prices[candidate] * held_stocks[candidate]) / (size_to_buy + held_stocks[candidate])  
                        budget -= avg_bought_price * size_to_buy
                        log.append("Buy {} {} at price {} with pred {:.3f}".format(size_to_buy, ticker, current_price, self.preds[day - 1, candidate]))
                    
                        held_prices[candidate] = avg_bought_price
                        held_stocks[candidate] += size_to_buy
                        max_held_prices[candidate] = max(held_prices[candidate], max_held_prices[candidate])
                        held_day[candidate] = day

                        transaction_history.append(current_date, ticker, "buy", size_to_buy, bought_price, avg_bought_price, None, None, None, "prediction")
                
                hold_stocks = np.where(held_stocks > 0)[0]
                
                #Check portfolio and sell
                for s in hold_stocks:

                    if high_prices[s] > max_held_prices[s]:            
                        max_held_prices[s] = high_prices[s]                   
                    
                    if day - held_day[s] >= self.t_delay:                
                        # Sell by trailing stop loss
                        if current_prices[s] < max_held_prices[s] * (1 - self.stop_loss):
                            
                            size_to_sell = held_stocks[s]
                            profit_rate = current_prices[s] / held_prices[s] - 1
                            log.append("Sell {} {} at price {} by trailing stop loss with profit {:.3f}%".format(held_stocks[s], self.ticker_symbols[s], current_prices[s], profit_rate * 100))
                            bought_price = held_prices[s]
                            bought_avg_price = bought_price * (1 + self.commission)
                            sold_price = current_prices[s]
                            sold_avg_price = sold_price * (1 - self.commission)
                            budget += sold_avg_price * size_to_sell

                            held_prices[s] = 0
                            max_held_prices[s] = 0
                            held_stocks[s] = 0                                           
                            transaction_history.append(current_date, self.ticker_symbols[s], "sell", size_to_sell, bought_price, bought_avg_price, sold_price, sold_avg_price, profit_rate, note = "trailing_stop")
                            
                                            
                        elif pred[s] < self.sell_thresholds[s]:
                            
                            size_to_sell = held_stocks[s]
                            profit_rate = current_prices[s] / held_prices[s] - 1
                            log.append("Sell {} {} at price {} by prediction with profit {:.3f}%".format(held_stocks[s], self.ticker_symbols[s], current_prices[s], profit_rate * 100))
                            bought_price = held_prices[s]
                            bought_avg_price = bought_price * (1 + self.commission)
                            sold_price = current_prices[s]
                            sold_avg_price = sold_price * (1 - self.commission)
                            budget += sold_avg_price * size_to_sell

                            held_prices[s] = 0
                            max_held_prices[s] = 0
                            held_stocks[s] = 0                                           
                            transaction_history.append(current_date, self.ticker_symbols[s], "sell", size_to_sell, bought_price, bought_avg_price, sold_price, sold_avg_price, profit_rate, note = "prediction")
                        
        #                 elif current_prices[stock] > portfolio._portfolio[stock]["bought_avg_price"] * (1 + take_profit):

                log.append("{} / {} ticker growths".format(len(pred[pred > 0]), len(pred)))
                # Update stock candidate to buy at next day ATO only if the market is okay                
                
                candidate_stocks = np.where(pred >= self.buy_thresholds)[0]
                filter_cache = self.filter.cache[day]
                candidate_stocks = np.intersect1d(candidate_stocks, filter_cache)
                already_hold = np.intersect1d(candidate_stocks, hold_stocks)
                candidate_stocks = np.setdiff1d(candidate_stocks, already_hold)

                # #Remove candidate that has category already in the portfolio
                # if self.is_unique_category_in_portfolio:
                #     filtered_candidate_stocks = []
                #     hold_categories = self.stock_info.loc[self.tickers[hold_stocks]]["category"].unique()
                    

                #     for i in candidate_stocks:
                #         stock = self.tickers[i]
                #         cat = self.stock_info.loc[stock]["category"]
                #         if cat not in hold_categories:
                #             filtered_candidate_stocks.append(i)
                #     candidate_stocks = filtered_candidate_stocks

                np.random.shuffle(candidate_stocks)                        
                                        
                # Update value
                hold_stock = np.where(held_stocks > 0)[0]
                trade_net_value = budget
                
                log.append("Portfolio")
                for s in hold_stock:
                    ticker = self.ticker_symbols[s]
                    log.append("{} : {}, current price: {}, current profit: {:.3f}%".format(ticker, held_stocks[s], current_prices[s], (current_prices[s] / held_prices[s] - 1) * 100))
                    trade_net_value += held_stocks[s] * current_prices[s]

                log.append("Portfolio cash: {}, net value: {}".format(budget, trade_net_value))

                budget_if_hold = average_hold * current_prices.sum()
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
            
            self.logs.append("\n".join(log))                        
            self.transaction_history_storage.append(transaction_history)

            self.trade_data.append(trades)
            self.hold_data.append(holds)
            self.final_trade_net_values.append(trade_net_value)
            self.final_sharp_ratios.append(sharp_ratio)
            self.final_gain_ratios.append(gain_ratio)

    # def execute_classification(self, num_test = 100, print_per = 10):

    #     # len(train_trade_sessions)
    #     for t in tqdm(range(num_test)):
            
    #         if t % print_per == 0:
    #             print("Test {}".format(t))
                        
    #         held_stocks = np.zeros(len(self.ticker_symbols))
    #         held_prices = np.zeros(len(self.ticker_symbols))
    #         held_day = np.zeros(len(self.ticker_symbols))
    #         max_held_prices = np.zeros(len(self.ticker_symbols))
    #         budget = self.initial_budget
    #         log = []
    #         candidate_stocks = []
            
    #         transaction_history = TransactionHistory()

    #         start_prices = self.prices[:, 0]
    #         average_hold = self.initial_budget / start_prices.sum()
            
    #         trades = []
    #         holds = []

    #         for day in range(len(self.trade_sessions)):
                
    #             current_date = self.trade_sessions[day]   
    #             log.append("==========================================================")     
    #             log.append("Day {}".format(format_str_datetime(current_date)))
    #             buy_pred = self.preds[day, :, 1]
    #             sell_pred = self.preds[day, :, 2]
                
    #             # current_data = self.env_df_indexing[current_date]
    #             # current_prices = current_data["close"].to_numpy()
    #             current_prices = self.prices[:, day]
    #             high_prices = self.high_prices[:, day]

    #             # Check candidate list and buy at ATO
    #             if day > 0:

    #                 open_prices = self.open_prices[:, day]                    
    #                 hold_stocks = np.where(held_stocks > 0)[0]
                    
    #                 # Check portfolio and buy   
    #                 max_num_stock_to_buy = self.max_number_of_stock - len(hold_stocks)
                
    #                 for i in range(min(max_num_stock_to_buy, len(candidate_stocks))):        

    #                     candidate = candidate_stocks[i]
    #                     ticker = self.ticker_symbols[candidate]
    #                     budget_for_stock = budget / (max_num_stock_to_buy - i) / (1 + self.commission)
    #                     current_price = open_prices[candidate]
                        
    #                     size_to_buy = (int)(budget_for_stock / current_price / self.min_num_share_per_purchase) * self.min_num_share_per_purchase        
    #                     if size_to_buy == 0:
    #                         break
                        
    #                     bought_price = current_price                    
    #                     avg_bought_price = (bought_price * size_to_buy * (1 + self.commission) + held_prices[candidate] * held_stocks[candidate]) / (size_to_buy + held_stocks[candidate])  
    #                     budget -= avg_bought_price * size_to_buy
    #                     log.append("Buy {} {} at price {} with pred {:.3f}".format(size_to_buy, ticker, current_price, self.preds[day - 1, candidate, 1]))
                    
    #                     held_prices[candidate] = avg_bought_price
    #                     held_stocks[candidate] += size_to_buy
    #                     max_held_prices[candidate] = max(held_prices[candidate], max_held_prices[candidate])
    #                     held_day[candidate] = day

    #                     transaction_history.append(current_date, ticker, "buy", size_to_buy, bought_price, avg_bought_price, None, None, None, "prediction")
                
                
    #             hold_stocks = np.where(held_stocks > 0)[0]
                
    #             #Check portfolio and sell
    #             for s in hold_stocks:

    #                 if high_prices[s] > max_held_prices[s]:            
    #                     max_held_prices[s] = high_prices[s]                   
                    
    #                 if day - held_day[s] >= self.t_delay:                
    #                     # Sell by trailing stop loss
    #                     if current_prices[s] < max_held_prices[s] * (1 - self.stop_loss):
                            
    #                         size_to_sell = held_stocks[s]
    #                         profit_rate = current_prices[s] / held_prices[s] - 1
    #                         log.append("Sell {} {} at price {} by trailing stop loss with profit {:.3f}%".format(held_stocks[s], self.ticker_symbols[s], current_prices[s], profit_rate * 100))
    #                         bought_price = held_prices[s]
    #                         bought_avg_price = bought_price * (1 + self.commission)
    #                         sold_price = current_prices[s]
    #                         sold_avg_price = sold_price * (1 - self.commission)
    #                         budget += sold_avg_price * size_to_sell

    #                         held_prices[s] = 0
    #                         max_held_prices[s] = 0
    #                         held_stocks[s] = 0                                           
    #                         transaction_history.append(current_date, self.ticker_symbols[s], "sell", size_to_sell, bought_price, bought_avg_price, sold_price, sold_avg_price, profit_rate, note = "trailing_stop")
                            
                                            
    #                     elif sell_pred[s] < self.sell_threshold[s]:
                            
    #                         size_to_sell = held_stocks[s]
    #                         profit_rate = current_prices[s] / held_prices[s] - 1
    #                         log.append("Sell {} {} at price {} by prediction with profit {:.3f}%".format(held_stocks[s], self.ticker_symbols[s], current_prices[s], profit_rate * 100))
    #                         bought_price = held_prices[s]
    #                         bought_avg_price = bought_price * (1 + self.commission)
    #                         sold_price = current_prices[s]
    #                         sold_avg_price = sold_price * (1 - self.commission)
    #                         budget += sold_avg_price * size_to_sell

    #                         held_prices[s] = 0
    #                         max_held_prices[s] = 0
    #                         held_stocks[s] = 0                                           
    #                         transaction_history.append(current_date, self.ticker_symbols[s], "sell", size_to_sell, bought_price, bought_avg_price, sold_price, sold_avg_price, profit_rate, note = "prediction")
                        
    #     #                 elif current_prices[stock] > portfolio._portfolio[stock]["bought_avg_price"] * (1 + take_profit):
                            
                

    #             # Update stock candidate to buy at next day ATO
    #             candidate_stocks = np.where(buy_pred >= self.buy_threshold)[0]
    #             filter_cache = self.filters.cache[day]
    #             candidate_stocks = np.intersect1d(candidate_stocks, filter_cache)
    #             already_hold = np.intersect1d(candidate_stocks, hold_stocks)
    #             candidate_stocks = np.setdiff1d(candidate_stocks, already_hold)
    #             np.random.shuffle(candidate_stocks)
                
                                        
    #             # Update value
    #             hold_stock = np.where(held_stocks > 0)[0]
    #             trade_net_value = budget
                
    #             log.append("Portfolio")
    #             for s in hold_stock:
    #                 ticker = self.ticker_symbols[s]
    #                 log.append("{} : {}, current price: {}, current profit: {:.3f}%".format(ticker, held_stocks[s], current_prices[s], (current_prices[s] / held_prices[s] - 1) * 100))
    #                 trade_net_value += held_stocks[s] * current_prices[s]

    #             log.append("Portfolio cash: {}, net value: {}".format(budget, trade_net_value))

    #             budget_if_hold = average_hold * current_prices.sum()
    #             gain_ratio = trade_net_value / budget_if_hold
    #             sharp_ratio = trade_net_value / self.initial_budget

    #             trades.append(trade_net_value)
    #             holds.append(budget_if_hold)
                        
    #         data = transaction_history.to_dataframe()
    #         data = data[data["type"] == "sell"]
    #         num_win_trades = len(data[data["profit_rate"] > 0])
    #         num_lose_trades = len(data[data["profit_rate"] < 0])
    #         max_win_trade = len(data[data["profit_rate"] < 0])
    #         max_win_trade = max(data["profit_rate"])
    #         max_lose_trade = min(data["profit_rate"]) 
            
    #         self.final_num_win_trades.append(num_win_trades)
    #         self.final_num_lose_trades.append(num_lose_trades)
    #         self.final_max_win_trades.append(max_win_trade)
    #         self.final_max_lose_trades.append(max_lose_trade)    
            
    #         self.logs.append("\n".join(log))                        
    #         self.transaction_history_storage.append(transaction_history)

    #         self.trade_data.append(trades)
    #         self.hold_data.append(holds)
    #         self.final_trade_net_values.append(trade_net_value)
    #         self.final_sharp_ratios.append(sharp_ratio)
    #         self.final_gain_ratios.append(gain_ratio)

            
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

    

