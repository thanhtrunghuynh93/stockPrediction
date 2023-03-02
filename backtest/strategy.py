from typing import List
from backtest.order import Order
from abc import abstractmethod
from backtest.transactions import Transactions
from utils.common import days_between
from pandas import DataFrame
import pandas as pd
import random
from backtest.portfolio import Portfolio

class Strategy():
    """A base of a strategy. Defining a strategy by yourself in "src/backtest/strategies/".
    """    
    def __init__(self, gts_dict: dict, gts_df:DataFrame, preds_df: DataFrame, config: dict, t_plus: int = 3) -> None:     
        """Initialization of Strategy.

        Args:
            strategy (Strategy): A strategy is defined by yourself in "src/backtest/strategies/".
            gts_dict (dict): A dict contains ground-truth prices of each stock, keys are symbols.
            gts_df (DataFrame): A dataframe contains close prices of each stock, column names are symbols.
            preds_df (DataFrame): A dataframe contains preds. Postprocessing in the strategy.
            config (dict): A dict of hyperparameters.
            t_plus (int, optional): Number of days that a stock from the day it was bought 
            come into our accounts (T+ in a stocks market). Defaults to 3 in the Vietnam market.
        """  
        self._orders_buy: List[Order] = []
        self._orders_sell: List[Order] = []
        self._gts_dict = gts_dict
        self._gts_df = gts_df
        self._preds_df = preds_df
        self._config = config
        self._t_plus = t_plus
        self._transactions = Transactions(config)
        symbols = gts_df.columns
        self._mapping_stocks = dict.fromkeys(symbols, 0)
        self._mapping_stocks_price = dict.fromkeys(symbols, 0)
        self._max_stocks = min(self._config["backtest"]["max_stocks"], len(symbols))
        self._trailing_sl = self._config["backtest"]["stop_loss"]
        self._portfolio = Portfolio(config)
        

    @property
    def transactions(self):
        return self._transactions

    @property
    def portfolio(self):
        return self._portfolio

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def execute(self):
        # Xét giá của ngày hôm x, mua bán vào ngày hôm x+1
        all_stocks_to_buy = self.stocks_to_buy()
        all_stocks_to_sell = self.stocks_to_sell()
        for i in range(len(self._gts_df)-1):
            current_date = self._gts_df.index[i]
            next_date = self._gts_df.index[i+1]
            current_market_prices = self._gts_df.iloc[i]
            next_market_prices = self._gts_df.iloc[i+1]
            self._portfolio.update_status(current_date, current_market_prices)
            # Bán khi xác suất cổ phiếu nắm giữ lớn hơn một ngưỡng
            possible_stocks_to_sell = all_stocks_to_sell[i]
            for stock in possible_stocks_to_sell:
                if self._mapping_stocks[stock] == 1:
                    next_price = self._gts_df[stock].iloc[i+1]
                    sold = self.sell(next_date, next_price, size, stock, "sell")
                    if sold:
                        self._mapping_stocks[stock] = 0
                        self._mapping_stocks_price[stock] = 0
                        size = self._portfolio.get_latest_avai(stock)
                        self._portfolio.update_status_sell(next_date, next_price, size, stock)


            # Mua khi xác suất cổ phiếu nắm giữ lớn hơn một ngưỡng
            if sum(self._mapping_stocks.values()) < self._max_stocks:
                num_stocks_to_buy = self._max_stocks - sum(self._mapping_stocks.values())
                possible_stocks_to_buy = all_stocks_to_buy[i]
                # Nếu đã mua rồi thì không mua cổ phiếu này nữa
                for stock, bought in self._mapping_stocks.items():
                    if stock in possible_stocks_to_buy and bought:
                        possible_stocks_to_buy.remove(stock)
                if len(possible_stocks_to_buy) > 0:
                    stocks_to_buy = []
                    while len(stocks_to_buy) < min(len(possible_stocks_to_buy), num_stocks_to_buy):
                        ran_stock = random.randint(0, len(possible_stocks_to_buy)-1)
                        stock_to_buy = possible_stocks_to_buy[ran_stock]
                        stocks_to_buy.append(stock_to_buy)
                        possible_stocks_to_buy.remove(stock_to_buy) # Tránh random vào cùng 1 cổ phiếu
                    # current_budget = self._transactions.budget
                    current_budget = self._portfolio.budget
                    bugdet_each_stock = int(current_budget / num_stocks_to_buy)
                    for stock in stocks_to_buy:
                        next_price = self._gts_df[stock].iloc[i+1]
                        size = bugdet_each_stock // next_price // 100 * 100
                        self.buy(next_date, next_price, size, stock, "buy")
                        self._mapping_stocks[stock] = 1
                        self._mapping_stocks_price[stock] = next_price
                        self._portfolio.update_status_buy(next_date, next_price, size, stock)

            # Bán khi chạm trailing stop loss
            for stock, max_price in self._mapping_stocks_price.items():
                if max_price != 0:
                    if self._gts_df[stock].iloc[i] > max_price:
                        self._mapping_stocks_price[stock] = self._gts_df[stock].iloc[i]
                    elif ((max_price - self._gts_df[stock].iloc[i]) / max_price) < self._trailing_sl: # Giam 7% thi ban
                        next_price = self._gts_df[stock].iloc[i+1]
                        sold = self.sell(next_date, next_price, size, stock, "sell")
                        if sold:
                            self._mapping_stocks[stock] = 0
                            self._mapping_stocks_price[stock] = 0
                            size = self._portfolio.get_latest_avai(stock)
                            self._portfolio.update_status_sell(next_date, next_price, size, stock)

            # Bán hết các cổ phiếu có thể bán
            if i == len(self._gts_df)-2:
                self._portfolio.sell_all_latest_avai(next_date, next_market_prices)
                

    @abstractmethod
    def stocks_to_buy(self, ind):
        pass

    @abstractmethod
    def stocks_to_sell(self, ind):
        pass

    def buy(self,
            current_date: str,
            current_price: float,
            size: int,
            symbol: str,
            type: str,
            sl: float = None,
            tp: float = None):
        order = Order(current_date, current_price, size, symbol, type, sl, tp)
        # self._orders_buy.append(order)
        # self._transactions.append_order(order)
        return True

    def sell(self,
             current_date: str,
             current_price: float,
             size: int,
             symbol: str,
             type: str
             ):
        # latest_symbol_order = self._transactions.get_latest_symbol_order(symbol)
        # size = latest_symbol_order['size']
        # days = days_between(latest_symbol_order['date'], current_date)
        # print("days btw: ", days)
        # T+3 (in Vietnam market)
        # if days >= self._t_plus:
        avai_size = self._portfolio.get_latest_avai(symbol)
        if avai_size > 0:
            order = Order(current_date, current_price, size, symbol, type)
            # self._orders_sell.append(order)
            # self._transactions.append_order(order)
            return True
        else:
            return False