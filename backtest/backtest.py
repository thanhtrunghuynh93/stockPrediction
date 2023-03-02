import numpy as np
from backtest.report import Report
from backtest.strategy import Strategy
from pandas import DataFrame
from backtest.report import Report
import pandas as pd

class Backtest():
    """Backtest
    >>> bt = Backtest(Prob_Trend_Prediction, gts_dict, preds_df, list_symbols, config)
    >>> report = bt.run_backtest()
    """    
    def __init__(self, strategy: Strategy, gts_dict: dict, preds_df: DataFrame, config: dict) -> None:
        """Initialization of Backtest.

        Args:
            strategy (Strategy): A strategy is defined by yourself in "src/backtest/strategies/".
            gts_dict (dict): A dict contains ground-truth prices of each stock, keys are symbols.
            preds_df (DataFrame): A dataframe contains preds (columns usually are symbols). 
            Postprocessing in the strategy.
            config (dict): A dict of hyperparameters.
        """        
        self._config = config
        self._gts_df = self._process_ground_truth(gts_dict)
        self._strategy = strategy(gts_dict, self._gts_df, preds_df, config)

    def run_backtest(self) -> Report:      
        """Run backtest.

        Returns:
            Report: Return a final report containing multiple metrics/indicators.
        """              
        self._strategy.init()
        self._strategy.execute()
        # report = Report(self._gts_df, self._strategy.portfolio, self._config)
        report = Report(self._strategy.portfolio, self._config)
        report.compute_stats()
        return report

    @property
    def strategy(self):
        return self._strategy

    def _process_ground_truth(self, gts_dict) -> DataFrame:
        gts_df = pd.DataFrame()
        history_window = self._config["data"]["history_window"]
        for sym, df in gts_dict.items():
            gt_price = (df["open"].iloc[history_window:] + df["high"].iloc[history_window:]) / 2
            gt_price = pd.DataFrame(gt_price.to_list(), index=gt_price.index, columns=[sym])
            gts_df = pd.concat([gts_df, gt_price], axis=1)
        return gts_df