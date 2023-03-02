class Order:
    """Structure of an order.
    """    
    def __init__(self,
                current_date:  str,
                current_price: float,
                size: int,
                symbol: str,
                type: str,
                sl_price: float = None,
                tp_price: float = None):
        """Initialization of an order.

        Args:
            current_date (str): Buy/sell date.
            current_price (float): Buy/sell price.
            size (int): Number of stocks bought/sold.
            symbol (str): Symbol bought/sold.
            type (str): "buy" or "sell".
            sl_price (float, optional): Stop loss price. Defaults to None.
            tp_price (float, optional): Take profit price. Defaults to None.
        """    
        self._current_date = current_date
        self._current_price = current_price
        self._size = size
        self._sl_price = sl_price
        self._tp_price = tp_price
        self._symbol = symbol
        self._type = type


    @property
    def current_date(self) -> str:
        return self._current_date

    @current_date.setter
    def current_date(self, current_date) -> str:
        self._current_date = current_date

    @property
    def current_price(self) -> float:
        return self._current_price

    @current_price.setter
    def current_price(self, current_price) -> float:
        self._current_price = current_price

    @property
    def symbol(self) -> str:
        return self._symbol

    @symbol.setter
    def symbol(self, symbol) -> str:
        self._symbol = symbol

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, type) -> str:
        self._type = type

    @property
    def sl(self) -> float:
        return self._sl_price

    @property
    def tp(self) -> float:
        return self._tp_price

    @property
    def size(self) -> float:
        return self._size