from backtest.strategy import Strategy
from backtest.filter import StocksFilter
from utils.backtest import lists_intersection
class TrendReturnPrediction(Strategy):
    def init(self) -> None:
        self._buy_trend_return_threshold = self._config["backtest"]["buy_trend_return_threshold"]
        self._sell_trend_return_threshold = self._config["backtest"]["sell_trend_return_threshold"]
        self._preds = self._preds_df.copy()
        self._gts = self._gts_dict.copy()

    def stocks_to_sell(self):
        possible_stocks_to_sell = []
        for i in range(len(self._preds)):
            pred_trend_return = self._preds.iloc[i]
            filtered_symbols_by_trend_return = pred_trend_return.index[(pred_trend_return <= -self._sell_trend_return_threshold)].tolist()
            final_symbols = filtered_symbols_by_trend_return
            possible_stocks_to_sell.append(final_symbols)
        return possible_stocks_to_sell

    def stocks_to_buy(self):
        fitler = StocksFilter(self._gts)
        filtered_symbols_by_sma = fitler.sma(10, 30)
        possible_stocks_to_buy = []
        for i in range(len(self._preds)):
            pred_trend_return = self._preds.iloc[i]
            filtered_symbols_by_trend_return = pred_trend_return.index[(pred_trend_return >= self._buy_trend_return_threshold)].tolist()
            final_symbols = lists_intersection(filtered_symbols_by_sma[i], filtered_symbols_by_trend_return)
            final_symbols = filtered_symbols_by_trend_return
            possible_stocks_to_buy.append(final_symbols)

        return possible_stocks_to_buy