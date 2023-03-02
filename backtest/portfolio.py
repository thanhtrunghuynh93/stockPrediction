import pandas as pd
from pandas import DataFrame
from utils.date_util import format_str_datetime
pd.options.display.float_format = '{:,.3f}'.format


class Portfolio:
    """Portfolio and budget
    >>> for i in range(len(close_prices)):
    >>>     market_prices = close_prices.iloc[i]
    >>>     portfolio.update_status(market_prices)
    >>>     if buy_condition:
    >>>         portfolio.update_status_buy()
    >>>     if sell_condition:
    >>>         portfolio.update_status_sell()
    >>> print(portfolio.portfolio)
    >>> print(portfolio.budget)
    >>> print(portfolio.get_portfolio_history_by_date("2021-05-07"))
    """    
    # def __init__(self, config):
    #     columns = ["avai", "T0", "T1", "T2", "total", "market_price", "market_value", "bought_avg_price", "value", "profit", "profit_rate"]
    #     self._portfolio = pd.DataFrame(columns=columns)
    #     self._commission = config["backtest"]["commission"]
    #     self._budget = config["backtest"]['initial_equity'] * (1 - config["backtest"]['per_cash'])
    #     self._portfolio_history = {}
    #     self._budget_history = {}

    def __init__(self, initial_budget = 1000000000, commission = 0.0015):
        columns = ["avail", "T0", "T1", "T2", "total", "market_price", "market_value", "bought_avg_price", "bought_value", "profit", "profit_rate", "max_record_close_price"]
        self._portfolio = pd.DataFrame(columns=columns)
        self._commission = commission
        self._budget = initial_budget
        self._portfolio_history = {}
        self._budget_history = {}        

    def update_status(self, date, market_prices: DataFrame):
        """Update status for every day. Gọi function này để cập nhật trạng thái sau mỗi ngày (index).

        Args:
            market_prices (DataFrame): Giá đóng cửa của các mã cổ phiếu.
        """       
        self._portfolio["market_price"] = market_prices.loc[self._portfolio.index]
        self._portfolio["market_value"] = self._portfolio["market_price"] * self._portfolio["total"]
        self._portfolio["avail"] += self._portfolio["T2"]
        self._portfolio["T2"] = self._portfolio["T1"]
        self._portfolio["T1"] = self._portfolio["T0"]
        self._portfolio["T0"] = 0
        self._portfolio["profit"] = self._portfolio["market_value"] * (1 - self._commission) - self._portfolio["bought_value"]
        self._portfolio["profit_rate"] = self._portfolio["profit"] / self._portfolio["bought_value"]
        self._portfolio["max_record_close_price"] = self._portfolio[['market_price','max_record_close_price']].max(axis=1)

        date = format_str_datetime(date)
        self._portfolio_history[date] = self._portfolio.copy()
        self._budget_history[date] = self._budget

    def update_status_buy(self, date, price, size, symbol):
        price_with_com = price * (1 + self._commission)
        self._budget -= price_with_com * size
        if symbol not in self._portfolio.index:
            self._init_portfolio(symbol, price_with_com, size)
        else:
            prev_total = self._portfolio.loc[symbol, "total"]
            prev_value = self._portfolio.loc[symbol, "bought_value"]
            self._portfolio.loc[symbol, "T0"] += size
            self._portfolio.loc[symbol, "bought_avg_price"] = (prev_value + price_with_com * size) / (prev_total + size) 
            self._portfolio.loc[symbol, "total"] += size
            self._portfolio.loc[symbol, "bought_value"] = self._cal_value(symbol)
            self._portfolio.loc[symbol, "profit"] = self._cal_profit(symbol)
            self._portfolio.loc[symbol, "profit_rate"] = self._cal_profit_rate(symbol)
        date = format_str_datetime(date)
        self._portfolio_history[date] = self._portfolio.copy()
        self._budget_history[date] = self._budget        

    
    def update_status_sell(self, date, price, size, symbol):
        price_with_com = price * (1 - self._commission)
        self._budget += price_with_com * size 
        if symbol not in self._portfolio.index:
            self._init_portfolio(symbol, price, size)
        else:
            if self._portfolio.loc[symbol, "total"] == size:
                self._portfolio = self._portfolio.drop(index=symbol)
            else:
                if size > self._portfolio.loc[symbol, "avail"]:
                    print("Xu ly sau")
                    exit()
                self._portfolio.loc[symbol, "total"] -= size
                self._portfolio.loc[symbol, "avail"] -= size
                self._portfolio.loc[symbol, "bought_value"] = self._cal_value(symbol)
                self._portfolio.loc[symbol, "profit"] = self._cal_profit(symbol)
                self._portfolio.loc[symbol, "profit_rate"] = self._cal_profit_rate(symbol)
        date = format_str_datetime(date)
        self._portfolio_history[date] = self._portfolio.copy()
        self._budget_history[date] = self._budget


    def get_portfolio_history_by_date(self, date):
        date = format_str_datetime(date)
        return self._portfolio_history[date]

    def get_latest_avail(self, symbol):
        return self._portfolio.loc[symbol, "avail"]

    def get_latest_value(self, symbol):
        return self._portfolio.loc[symbol, "bought_value"]

    def get_net_value(self):
        net_value = self._portfolio["market_value"].sum() * (1 - self._commission) + self._budget        
        return net_value

    # def get_net_value_history(self, portfolio, budget):
    #     net_value = 0
    #     for symbol in portfolio.index:
    #         net_value += portfolio.loc[symbol, "bought_value"] * (1 - self._commission)
    #     net_value += budget
    #     return net_value


    # def sell_all_latest_avail(self, date, market_prices):
    #     for symbol in self._portfolio.index:
    #         price = market_prices[symbol]
    #         size = self.get_latest_avail(symbol)
    #         self.update_status_sell(date, price, size, symbol)


    @property
    def portfolio(self):
        return self._portfolio
    
    @property
    def budget(self):
        return self._budget

    def _cal_value(self, symbol):
        return self._portfolio.loc[symbol, "total"] * self._portfolio.loc[symbol, "bought_avg_price"]

    def _cal_profit(self, symbol):
        return self._portfolio.loc[symbol, "market_value"] * (1 - self._commission) - self._portfolio.loc[symbol, "bought_value"]
    
    def _cal_profit_rate(self, symbol):
        return self._cal_profit(symbol) / self._portfolio.loc[symbol, "bought_value"] - 1


    def _init_portfolio(self, symbol, price, size):
        price_with_com = price * (1 + self._commission)
        self._portfolio.loc[symbol, "avail"] = 0
        self._portfolio.loc[symbol, "T0"] = size
        self._portfolio.loc[symbol, "T1"] = 0
        self._portfolio.loc[symbol, "T2"] = 0
        self._portfolio.loc[symbol, "total"] = self._get_total_size(symbol)
        self._portfolio.loc[symbol, "market_price"] = price
        self._portfolio.loc[symbol, "market_value"] = price * size
        self._portfolio.loc[symbol, "bought_avg_price"] = price_with_com
        self._portfolio.loc[symbol, "bought_value"] = self._portfolio.loc[symbol, "total"] * self._portfolio.loc[symbol, "bought_avg_price"]
        self._portfolio.loc[symbol, "profit"] = 0
        self._portfolio.loc[symbol, "profit_rate"] = 0
        self._portfolio.loc[symbol, "max_record_close_price"] = price

    def _get_total_size(self, symbol):
        return self._portfolio.loc[symbol, ["T0", "T1", "T2"]].sum()