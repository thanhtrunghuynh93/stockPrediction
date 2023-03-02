from backtest.portfolio import Portfolio
from utils.backtest import count_consecutive_pos_values, count_consecutive_neg_values
from utils.common import days_between
import pandas as pd
import numpy as np
import statistics
import math
import warnings
warnings.filterwarnings("ignore")

class Report:
    """Report. Processings orders.
    """    
    def __init__(self, portfolio: Portfolio, config):
        self._report = {}
        self._config = config

        # Basic information
        self._commission = config["backtest"]["commission"]
        self._start_time = config["data"]["start_backtest"]
        self._end_time = config["data"]["end_backtest"]
        self._duration = days_between(self._start_time, self._end_time)
        self._initial_equity = config["backtest"]["initial_equity"]
        self._cash = self._initial_equity * config["backtest"]["per_cash"]
        self._budget = self._initial_equity - self._cash
        self._exposure_time = 0

        self._gross_win = 0
        self._gross_loss = 0
        self._equity_final = 0
        self._equity_peak = 0
        self._equity_trough = 0
        self._profit = 0
        self._return_final = 0
        self._acc_return_by_quater = []
        self._return_by_quater = []
        self._votatility = 0 # Chua biet la gi
        self._buy_and_hold_return = 0
        self._max_drawdown = 0
        self._max_drawdown_duration = 0 # max drawdown duration is the worst (the maximum/longest) amount of time an investment has seen between peaks 

        self._sharpe_ratio = 0
        self._sortino_ratio = 0
        self._calmar_ratio = 0
        self._gain_ratio = 0
        self._profit_factor = 0
        self._expectancy_ratio = 0
        self._sqn = 0 # system quality number indicator

        self._num_buy_trades = 0
        self._num_sell_trades = 0
        self._num_won_trade = 0
        self._num_lost_trade = 0
        self._num_consecutive_win_trade = 0
        self._num_consecutive_lost_trade = 0
        self._win_rate = 0
        self._best_trade = 0
        self._worst_trade = 0
        self._max_trade_duration = 0
        self._avg_trade_duration = 0

        self._portfolio = portfolio

    
    @property
    def report(self):
        return self._report

    def compute_stats(self):
        total_value = 0
        for symbol in self._portfolio.portfolio.index:
            total_value += self._portfolio.get_latest_value(symbol)
        self._equity_final = self._portfolio.budget + total_value
        print(self._equity_final)
        self._report = self._format_report()

    def _format_report(self):
        report = {}
        report["start_date"] = str(self._start_time) 
        report["end_date"] = str(self._end_time)
        report["duration"] = int(self._duration)
        report["initial_equity"] = int(self._initial_equity)
        report["cash"] = int(self._cash)
        report["budget"] = int(self._budget)
        report["exposure"] = int(self._exposure_time)

        report["gross_win"] = int(self._gross_win)
        report["gross_loss"] = int(self._gross_loss)
        report["final_equity"] = int(self._equity_final)
        report["equity_peak"] = int(self._equity_peak)
        report["equity_trough"] = int(self._equity_trough)
        report["profit"] = int(self._profit)
        report["final_return"] = round(self._return_final, 2)
        report["acc_return_by_quater"] = self._acc_return_by_quater
        report["return_by_quater"] = self._return_by_quater
        report["buy_and_hold_return"] = int(self._buy_and_hold_return)
        report["max_drawdown"] = int(self._max_drawdown)
        report["max_drawdown_duration"] = int(self._max_drawdown_duration)

        report["sharpe_ratio"] = round(self._sharpe_ratio, 2)
        report["sortino_ratio"] = round(self._sortino_ratio, 2)
        report["calmar_ratio"] = round(self._calmar_ratio, 2)
        report["gain_ratio"] = round(self._gain_ratio, 2)
        report["profit_factor"] = round(self._profit_factor, 2)
        report["expectancy_ratio"] = round(self._expectancy_ratio, 2)
        report["SQN"] = round(self._sqn, 2)

        report["num_buy_trades"] = int(self._num_buy_trades)
        report["num_sell_trades"] = int(self._num_sell_trades)
        report["num_won_trade"] = int(self._num_won_trade)
        report["num_lost_trade"] = int(self._num_lost_trade)
        report["num_consecutive_win_trade"] = int(self._num_consecutive_win_trade)
        report["num_consecutive_lost_trade"] = int(self._num_consecutive_lost_trade)
        report["win_rate"] = round(self._win_rate, 2)
        report["best_trade"] = int(self._best_trade)
        report["worst_trade"] = int(self._worst_trade)
        report["max_trade_duration"] = int(self._max_trade_duration)
        report["avg_trade_duration"] = int(self._avg_trade_duration)

        return report
