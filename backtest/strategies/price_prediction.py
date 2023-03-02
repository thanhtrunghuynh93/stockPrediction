from backtest.strategy import Strategy
import random

class PricePrediction(Strategy):
    def init(self) -> None:
        self._buy_prob_threshold = self._config["buy_prob_threshold"]
        self._max_stocks = self._config["max_stocks"]
        symbols = self._gts_df.columns
        self._mapping_stocks = dict.fromkeys(symbols, 0)
        self._mapping_stocks_price = dict.fromkeys(symbols, 0)
        self._max_stocks = min(self._max_stocks, len(symbols))
        self._trailing_sl = 0.07

    def execute(self):
        for i in range(1, len(self._gts_df)):
            prev_gt_prices = self._gts_df.iloc[i - 5]
            pred_prices = self._preds_df.iloc[i]
            current_date = self._gts_df.index[i]

            # Mua khi xác suất cổ phiếu nắm giữ lớn hơn một ngưỡng
            if sum(self._mapping_stocks.values()) < self._max_stocks:
                num_stocks_to_buy = self._max_stocks - sum(self._mapping_stocks.values())
                possible_stocks_to_buy = pred_prices.index[((pred_prices - prev_gt_prices)/prev_gt_prices >= self._buy_prob_threshold)].tolist()
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
                    current_budget = self._portfolio.budget
                    bugdet_each_stock = int(current_budget / num_stocks_to_buy)
                    for stock in stocks_to_buy:
                        current_price = self._gts_df[stock].iloc[i]
                        size = bugdet_each_stock // current_price // 100 * 100
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