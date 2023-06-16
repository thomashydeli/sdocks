import numpy as np
from numba import jit
from typing import Any,  Dict, List, Optional, Tuple, Union


class InvestmentData:

    def __init__(
        self,
        dates: List[str],
        prices: List[float],
    ):
        self.dates=dates
        self.prices=np.array(prices, dtype=np.float32)
        self.dataframe=None


    @staticmethod
    @jit(nopython=True)
    def _get_return(prices):
        return prices[1:]/prices[:-1]-1


    @staticmethod
    @jit(nopython=True)
    def _get_log_return(prices):
        return np.log(prices[1:]/prices[:-1])


    def get_return(self):
        self.returns=self._get_return(self.prices)
        return self.returns


    def get_log_return(self):
        self.log_returns=self._get_log_return(self.prices)
        return self.log_returns
    

    def _create_data_frame(self):


    def get_equity_curve(self):


    def get_return_curve(self):
        

    def get_log_return_curve(self):