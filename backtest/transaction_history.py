import pandas as pd
import numpy as np
from utils.date_util import format_str_datetime

class TransactionHistory:
    """Structure of an transaction.
    """    
    def __init__(self):
        """Initialization of an profit.

        Args:
            current_date (str): Buy/sell date.
            current_price (float): Buy/sell price.
            size (int): Number of stocks bought/sold.
            symbol (str): Symbol bought/sold.
            type (str): "buy" or "sell".
            sl_price (float, optional): Stop loss price. Defaults to None.
            tp_price (float, optional): Take profit price. Defaults to None.
        """    
        # self._data = pd.DataFrame(columns=['date', 'symbol', 'type', 'size', 'bought_price', 'bought_value', 'sold_price', "sold_value", 'commission_value', "profit", "profit_rate", 'note'])
        self._data = None
        self.raw_data = []
        self.columns = ['date', 'symbol', 'type', 'size', 'bought_price', 'bought_avg_price', 'sold_price', 'sold_avg_price', "profit_rate", 'note']
        
    def append(self, date, symbol, type, size, bought_price, bought_avg_price, sold_price, sold_avg_price, profit_rate, note = ""):

        self.raw_data.append([date, symbol, type, size, bought_price, bought_avg_price, sold_price, sold_avg_price, profit_rate, note])

    def to_dataframe(self):
        data = np.array(self.raw_data)
        self._data = pd.DataFrame(data, columns = self.columns)
        return self._data

        # new_data = {'date': format_str_datetime(date), 'symbol': symbol, 'type': type, 'size': size, 'bought_price' : bought_price, 'sold_price' : sold_price, 'profit_rate': profit_rate, 'note' : note}
        # self._data = self._data.append(new_data, ignore_index=True)

    def getHistoryByDate(self, date):
        pass