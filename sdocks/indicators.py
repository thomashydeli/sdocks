import numpy as np
from numba import jit
from datetime import datetime, timedelta
from sdocks.consts import InvestmentData

class TechnicalIndicator:


    def __init__(
        self,
        data: InvestmentData,
    ):
        self.data=data

    
    @staticmethod
    @jit(nopython=True)
    def _get_ma(prices, length, n):
        ma=np.zeros(len(prices)-(n-1))
        currentSum=prices[:n].sum()
        ma[0]=currentSum/n

        for i in range(n,length):
            currentSum+=prices[i]
            currentSum-=prices[i-n]
            ma[i-n+1]=currentSum/n
        return ma
        

    def sma(self, n=5):
        return {
            'dates':self.data.dates[n:],
            'values':self._get_ma(self.data.prices, self.data._n, n)
        }
