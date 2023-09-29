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

    
    @staticmethod
    @jit(nopython=True)
    def _get_msum(prices, length, n):
        msum=np.zeros(len(prices)-(n-1))
        currentSum=prices[:n].sum()
        msum[0]=currentSum

        for i in range(n,length):
            currentSum+=prices[i]
            currentSum-=prices[i-n]
            msum[i-n+1]=currentSum
        return msum

    
    @staticmethod
    @jit(nopython=True)
    def _get_msd(prices, length, n):
        msd=np.zeros(len(prices)-(n-1))
        currentPrices=prices[:n]
        msd[0]=np.std(currentPrices)

        for i in range(n,length):
            msd[i-n+1]=np.std(prices[(i-n+1):i])
        return msd
        

    def sma(self, n=5):
        return {
            'dates':self.data.dates[n:],
            'values':self._get_ma(self.data.prices, self.data._n, n)
        }


    def smsd(self, n=5):
        return {
            'dates':self.data.dates[n:],
            'values':self._get_msd(self.data.prices, self.data._n, n)
        }


    def macd(self,n1=5,n2=34):
        short=np.array(self.sma(n=n1)['values'][n2-n1:])
        long=np.array(self.sma(n=n2)['values'])
        return {
            'dates':self.data.dates[n2:],
            'values':short-long
        }

    
    def bollingerBand(self,n=20):
        mid=np.array(self.sma(n=n)['values'])
        sd=np.array(self.smsd(n=n)['values'])
        lower,upper=mid-2*sd, mid+2*sd
        return {
            'dates':self.data.dates[(n-1):],
            'lower':lower,
            'mid':mid,
            'upper':upper,
        }

    
    def chaikinMoneyFlow(self,n=20):
        if self.data.highs is None or self.data.lows is None or self.data.volumes is None:
            raise Exception("missing information: highs, lows, volumes")
        else:
            intermediate=self.data.volumes * (
                (2 * self.data.prices - self.data.highs - self.data.lows) / (self.data.highs - self.data.lows)
            )
            rolling_sum_money_flow_volume=self._get_msum(intermediate, self.data._n, n)
            rolling_sum_volume=self._get_msum(self.data.volumes, self.data._n, n)
            return {
                'dates':self.data.dates[n:],
                'chaikin_money_flow':rolling_sum_money_flow_volume/rolling_sum_volume
            }

    
    def macd_signal(self,n1=5,n2=34):
        macd_op=self.macd(n1,n2)
        macd_sign=np.sign(macd_op['values'])
        macd_shifted_sign=macd_sign[1:]
        macd_sign_origin=macd_sign[:-1]
        macd_signal=macd_shifted_sign * (macd_sign_origin != macd_shifted_sign)
        return {
            'dates':macd_op['dates'][1:],
            'signal':macd_signal
        }
        

    def bollinger_signal(self,n=20):
        bollinger_op=self.bollingerBand(n)

        sell=self.data.prices[(n-1):]>np.array(bollinger_op['upper'])
        buy=self.data.prices[(n-1):]<np.array(bollinger_op['lower'])

        return {
            'dates': bollinger_op['dates'],
            'signal':(1*buy-1*sell)
        }