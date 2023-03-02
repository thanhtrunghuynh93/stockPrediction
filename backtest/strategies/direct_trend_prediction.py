from backtest.strategy import Strategy
from utils.backtest import covert_data_for_backtest_with_direct_trend
import random

class Direct_Trend_Trediction(Strategy):
    def init(self) -> None:
        self._buy_prob_threshold = self._config["buy_prob_threshold"]
        self._sell_prob_threshold = self._config["sell_prob_threshold"]
        self._max_stocks = self._config["max_stocks"]
        self._trend_df = covert_data_for_backtest_with_direct_trend(self._preds_df, self._symbols)
        self._mapping_stocks = dict.fromkeys(self._symbols, 0)
        self._mapping_stocks_price = dict.fromkeys(self._symbols, 0)
        self._max_stocks = min(self._max_stocks, len(self._symbols))
        self._trailing_sl = 0.07

    def execute(self):
        for i in range(len(self._gts_df)):
            trend = self._trend_df.iloc[i]
            current_date = self._gts_df.index[i]

            # # Bán khi xác suất cổ phiếu nắm giữ lớn hơn một ngưỡng
            # possible_stocks_to_sell = trend.index[(trend == 2)]
            # for stock in possible_stocks_to_sell:
            #     if self._mapping_stocks[stock] == 1:
            #         current_price = self._gts_df[stock].iloc[i]
            #         latest_symbol_order = self._portfolio.get_latest_symbol_order(stock)
            #         size = latest_symbol_order['size']
            #         self.sell(current_date, current_price, size, stock, "sell")
            #         self._mapping_stocks[stock] = 0

            # Mua khi xác suất cổ phiếu nắm giữ lớn hơn một ngưỡng
            if sum(self._mapping_stocks.values()) < self._max_stocks:
                num_stocks_to_buy = self._max_stocks - sum(self._mapping_stocks.values())
                possible_stocks_to_buy = trend.index[(trend == 1)]
                if len(possible_stocks_to_buy) > 0:
                    stocks_to_buy = []
                    while len(stocks_to_buy) < min(len(possible_stocks_to_buy), num_stocks_to_buy):
                        ran_stock = random.randint(0, len(possible_stocks_to_buy)-1)
                        stock_to_buy = possible_stocks_to_buy[ran_stock]
                        # Nếu đã mua rồi thì không mua cổ phiếu này nữa
                        if self._mapping_stocks[stock_to_buy] == 1:
                            continue
                        else:
                            stocks_to_buy.append(stock_to_buy)
                    current_budget = self._portfolio.budget
                    bugdet_each_stock = int(current_budget / num_stocks_to_buy)
                    for stock in stocks_to_buy:
                        current_price = self._gts_df[stock].iloc[i]
                        size = bugdet_each_stock // current_price
                        self.buy(current_date, current_price, size, stock, "buy")
                        self._mapping_stocks[stock] = 1
                        self._mapping_stocks_price[stock] = current_price

            # Bán khi chạm trailing stop loss
            for stock, max_price in self._mapping_stocks_price.items():
                if max_price != 0:
                    if self._gts_df[stock].iloc[i] > max_price:
                        self._mapping_stocks_price[stock] = self._gts_df[stock].iloc[i]
                    elif ((max_price - self._gts_df[stock].iloc[i]) / max_price) < self._trailing_sl: # Giam 7% thi ban
                        current_price = self._gts_df[stock].iloc[i]
                        sold = self.sell(current_date, current_price, size, stock, "sell")
                        if sold:
                            self._mapping_stocks[stock] = 0
                            self._mapping_stocks_price[stock] = 0