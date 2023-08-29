import numpy as np
from numba import jit

class PerfMetrics:


    def __init__(
        self, 
        data,
        benchmark_rate,
        return_field='return',
        drawdown_method='log',
    ):
        self.data=data
        self.benchmark_rate=benchmark_rate
        self.dates=self.data['date'].values.astype(np.datetime64)
        self.prices=self.data['price'].values.astype(np.float32)
        self._return=self.data[return_field].dropna().values.astype(np.float32)
        self.years_past=self._get_year_past()
        self.entries_per_year=len(self.data)/self.years_past
        self.information_payload={}

        self.drawdown_method=drawdown_method
        self.drawdown_evaluator={
            'dollar': lambda price, peak: peak - price,
            'percent': lambda price, peak: 1 - price / peak,
            'log': lambda price, peak: np.log(peak / price)
        }


    def _get_year_past(self):
        return (
            self.data.iloc[-1]['date']-self.data.iloc[0]['date']
        ).days/365.25


    @staticmethod
    @jit(nopython=True)
    def _get_volatility(returns, entries_per_year):
        # default using log return
        return np.std(returns) * np.sqrt(entries_per_year)


    def get_volatility(self):
        self.information_payload['volatility']=self._get_volatility(self._return, self.entries_per_year)


    @staticmethod
    @jit(nopython=True)
    def _get_cagr(prices, years_past):
        return (prices[-1] / prices[0]) ** (1 / years_past) - 1


    def get_cagr(self):
        self.information_payload['cagr']=self._get_cagr(self.prices, self.years_past)


    def get_sharpe(
        self,
    ):
        if 'volatility' not in self.information_payload: self.get_volatility()
        if 'cagr' not in self.information_payload: self.get_cagr()
        self.information_payload['sharpe']=(
            self.information_payload['cagr'] - self.benchmark_rate
        ) / self.information_payload['volatility']

    
    @staticmethod
    @jit(nopython=True)
    def _get_downside_deviation(returns, entries_per_year, benchmark_rate):
        adj_benchmark_rate = (1 + benchmark_rate) ** (1 / entries_per_year) - 1
        downside_series = adj_benchmark_rate - returns
        downside_ss = (downside_series[downside_series > 0] ** 2).sum() # downside sum of squares
        return np.sqrt(downside_ss / (len(returns) - 1)) * np.sqrt(entries_per_year)
 

    def get_downside_deviation(self):
        self.information_payload['downside_deviation']=self._get_downside_deviation(
            self._return, self.entries_per_year, self.benchmark_rate
        )


    @staticmethod
    @jit(nopython=True)
    def _get_sortino_ratio(cagr, benchmark_rate, downside_deviation):
        return (cagr - benchmark_rate) / downside_deviation


    def get_sortino_ratio(self):
        if 'cagr' not in self.information_payload: self.get_cagr()
        if 'downside_deviation' not in self.information_payload: self.get_downside_deviation()
        self.information_payload['sortino_ratio']=self._get_sortino_ratio(
            self.information_payload['cagr'], 
            self.benchmark_rate,
            self.information_payload['downside_deviation']
        )
    

    def _get_max_drawdown(self, dates, prices, method, evaluator):
        max_drawdown=0
        local_peak_date = peak_date = trough_date = dates[0]
        local_peak_price = peak_price = trough_price = prices[0]
        for date, price in zip(dates, prices):
            if price > local_peak_price:
                local_peak_price, local_peak_date = price, date

            drawdown=evaluator[method](price, local_peak_price)

            if drawdown > max_drawdown:
                max_drawdown = drawdown

                peak_date, peak_price = local_peak_date, local_peak_price
                trough_date, trough_price = date, price

        return {
            "max_drawdown":max_drawdown,
            "peak_date":peak_date,
            "peak_price":peak_price,
            "trough_date":trough_date,
            "trough_price":trough_price,
        }


    def get_max_drawdown(self,method='log'):
        assert method in self.drawdown_evaluator, f'Method: "{method}" must by one of {list(self.drawdown_evaluator.keys())}'
        self.information_payload['max_drawdown']=self._get_max_drawdown(
            self.dates,
            self.prices,
            method,
            self.drawdown_evaluator,
        )


    def get_payload(
        self,
    ):
        if 'volatility' not in self.information_payload: self.get_volatility()
        if 'cagr' not in self.information_payload: self.get_cagr()
        if 'sharpe' not in self.information_payload: self.get_sharpe()
        if 'sortino_ratio' not in self.information_payload: self.get_sortino_ratio()
        if 'max_drawdown' not in self.information_payload: self.get_max_drawdown(method=self.drawdown_method)
        return self.information_payload
