import pandas as pd
from backtest.order import Order


class Transactions:
    """Structure of a porfolio. Restore orders to do the report.
    """    
    def __init__(self, config: dict):
        """Initialization. There are two structure types of porfolios to process orders easily.

        Args:
            config (dict): A dict of hyperparameters.
        """        
        self._config = config
        self._commission = config["backtest"]['commission']
        self._budget = config["backtest"]['initial_equity'] * (1 - config["backtest"]['per_cash'])
        self._equity = config["backtest"]['initial_equity']
        self._cash = config["backtest"]['initial_equity'] - self._budget
        self._orders = pd.DataFrame(
            columns=['date', 'symbol', 'type', 'size', 'price', 'commission', 'budget', 'equity'])
        self._transactions = pd.DataFrame(columns=['symbol', 'size', 'avg_buy_price'])
        self._matching_orders = pd.DataFrame(
            columns=['symbol', 'buy_date',
                     'buy_price', 'buy_size',
                     'sell_date', 'sell_price',
                     'sell_size'])
        self._mapping_matching_orders = {}

    def append_order(self, order: Order):
        """Append an order to the transactions.

        Args:
            order (Order): An order.

        Raises:
            TypeError: "Type must be buy or sell!" 
        """        
        order_value = order.size * order.current_price
        commission = self._commission * order_value
        if order.type == 'buy':
            if order.symbol not in self._mapping_matching_orders:
                self._mapping_matching_orders[order.symbol] = [
                    order.current_date, order.current_price, order.size]
                self._budget = self._budget - commission - order_value
                self._transactions = self._transactions.append({
                    'symbol': order.symbol,
                    'size': order.size,
                    'avg_buy_price': order.current_price
                }, ignore_index=True)
        elif order.type == 'sell':
            if order.symbol in self._mapping_matching_orders:
                self._matching_orders = self._matching_orders.append({'symbol': order.symbol,
                                                                      'buy_date': self._mapping_matching_orders[order.symbol][0],
                                                                      'buy_price': self._mapping_matching_orders[order.symbol][1],
                                                                      'buy_size': self._mapping_matching_orders[order.symbol][2],
                                                                      'sell_date': order.current_date,
                                                                      'sell_price': order.current_price,
                                                                      'sell_size': order.size}, ignore_index=True)
                del self._mapping_matching_orders[order.symbol]
                self._transactions = self._transactions.drop(self._transactions[self._transactions['symbol'] == order.symbol].index)
                self._budget = self._budget - commission + order_value
                
        else:
            raise TypeError("Type must be buy or sell!")
        self._equity = (self._transactions["size"] * self._transactions["avg_buy_price"]).sum() + self._budget + self._cash
        self._orders = self._orders.append({'date': order.current_date,
                                            'symbol': order.symbol,
                                            'type': order.type,
                                            'size': order.size,
                                            'price': order.current_price,
                                            'commission': commission,
                                            'budget': self._budget,
                                            'equity': self._equity}, ignore_index=True)

    @property
    def orders(self):
        return self._orders

    @orders.setter
    def orders(self, orders):
        self._orders = orders
    
    @property
    def matching_orders(self):
        return self._matching_orders

    @matching_orders.setter
    def matching_orders(self, matching_orders):
        self._matching_orders = matching_orders

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, budget):
        self._budget = budget

    def get_latest_symbol_order(self, symbol):
        # Get latest symbol order of one symbol, to sell it (exclusive=True)
        return self._orders[(self._orders['symbol'] == symbol) & (self._orders['type'] == 'buy')].iloc[-1]