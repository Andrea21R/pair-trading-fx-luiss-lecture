import pandas as pd
import numpy as np
from typing import *


class Analyst():

    def __init__(self, strategy_rets: pd.Series) -> NoReturn:
        self.strategy_rets = strategy_rets

    @staticmethod
    def _calc_total_return(rets: pd.Series) -> float:
        return rets.cumsum().iloc[-1]

    @staticmethod
    def _calc_avg_ann_rets(rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return rets.mean() * 252

    @staticmethod
    def _calc_avg_ann_volatility(rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return rets.std() * np.sqrt(252)

    @classmethod
    def build_equity_curve(cls, rets: pd.Series, compound: bool = False) -> pd.DataFrame:
        if not compound:
            equity_curve: pd.DataFrame = rets.cumsum() + 1
        else:
            equity_curve: pd.DataFrame = (1 + rets).cumprod()
        return equity_curve

    @classmethod
    def _calc_dd_max(cls, rets: pd.Series, compound: bool = False) -> float:
        equity_curve = cls.build_equity_curve(rets, compound)
        return float((equity_curve / equity_curve.cummax() - 1).min())

    @staticmethod
    def _calc_ann_semivolatility(rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return float(rets[rets < 0].std() * np.sqrt(252))

    @staticmethod
    def _calc_daily_skew(rets: pd.Series) -> float:
        return rets.skew()

    @staticmethod
    def _calc_daily_kurt(rets: pd.Series) -> float:
        return rets.kurt()

    @staticmethod
    def _calc_ann_sharpe(rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return (rets.mean() / rets.std()) * np.sqrt(252)

    @classmethod
    def _calc_ann_sortino(cls, rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return (rets.mean() * 252) / cls._calc_ann_semivolatility(rets)

    @staticmethod
    def _calc_best_day(rets: pd.Series) -> float:
        return float(rets.max())

    @staticmethod
    def _calc_worst_day(rets: pd.Series) -> float:
        return float(rets.min())

    @staticmethod
    def _calc_var(rets: pd.Series, parametric: bool, var_confidence: float = 0.99) -> float:
        if parametric:
            return -rets.std() * 2.33
        else:
            return rets.quantile(q=1-var_confidence, interpolation='linear')

    def get_performance_analysis(self):
        return pd.Series(
            {
                'Total Return':             self._calc_total_return(self.strategy_rets),
                'Avg Ann. Return':          self._calc_avg_ann_rets(self.strategy_rets),
                'Avg Ann. Volatility':      self._calc_avg_ann_volatility(self.strategy_rets),
                'Avg Ann. Semi-Volatility': self._calc_ann_semivolatility(self.strategy_rets),
                'Max Drawdown':             self._calc_dd_max(self.strategy_rets),
                'VaR(99%) Parametric':      self._calc_var(self.strategy_rets, parametric=True),
                'VaR(99%) Not Parametric':  self._calc_var(self.strategy_rets, parametric=False, var_confidence=0.99),
                'Daily Skewness':           self._calc_daily_skew(self.strategy_rets),
                'Daily Kurtosis':           self._calc_daily_kurt(self.strategy_rets),
                'Ann. Sharpe':              self._calc_ann_sharpe(self.strategy_rets),
                'Ann. Sortino':             self._calc_ann_sortino(self.strategy_rets),
                'Best Day':                 self._calc_best_day(self.strategy_rets),
                'Worst Day':                self._calc_worst_day(self.strategy_rets)
            }
        )
