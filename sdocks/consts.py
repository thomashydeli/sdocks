import numpy as np
import pandas as pd
from numba import jit
from plotly import graph_objects as go
from typing import Any,  Dict, List, Optional, Tuple, Union


class InvestmentData:


    def __init__(
        self,
        dates: List[str],
        prices: List[float],
        ticker: Optional[str]=None,
        highs:Optional[List[float]]=None,
        lows:Optional[List[float]]=None,
        volumes:Optional[List[float]]=None,
    ):
        self.dates=dates
        self.prices=np.array(prices, dtype=np.float32)
        self._n=len(self.prices) # how many datapoints
        self.ticker=ticker

        self._create_data_frame() # creating a dataframe for preserving information

        if highs is not None: self.highs=np.array(highs, dtype=np.float32) 
        else: self.highs=None
        if lows is not None: self.lows=np.array(lows, dtype=np.float32) 
        else: self.lows=None
        if volumes is not None: self.volumes=np.array(volumes, dtype=np.float32) 
        else: self.volumes=None


    @staticmethod
    @jit(nopython=True)
    def _get_return(prices):
        return prices[1:]/prices[:-1]-1


    @staticmethod
    @jit(nopython=True)
    def _get_log_return(prices):
        return np.log(prices[1:]/prices[:-1])


    def _get_shifted_lines(self, feature, function):
        values=function(self.prices)
        self.dataframe[feature]=-np.NaN  
        self.dataframe.loc[1:,feature]=values
        return values


    def get_return(self):
        self.returns=self._get_shifted_lines('return',self._get_return)
        return self.returns


    def get_log_return(self):
        self.log_returns=self._get_shifted_lines('log return',self._get_log_return)
        return self.log_returns
    

    def _create_data_frame(self):
        self.dataframe=pd.DataFrame(
            {
                'date':pd.to_datetime(self.dates),
                'price':self.prices,
            }
        )

    
    def _make_line_chart(self, col):
        fig=go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.dataframe['date'],
                y=self.dataframe[col]
            )
        )
        title=f'{col} curve' if not self.ticker else f'{col} curve for {self.ticker}'
        fig.update_layout(dict(title=title))

        fig.show()


    def get_return_curve(self):
        self._make_line_chart('return')
        

    def get_log_return_curve(self):
        self._make_line_chart('log return')


    def get_price_curve(self):
        self._make_line_chart('price')

    
    def get_data(self):
        return self.dataframe