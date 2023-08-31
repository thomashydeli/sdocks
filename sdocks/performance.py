import numpy as np
from numba import jit
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

class PerfMetrics:


    def __init__(
        self, 
        data,
        benchmark_rate,
        benchmark_log_return,
        return_field='return',
        drawdown_method='log',
        calmer_years=3,
    ):
        self.data=data
        self.benchmark_rate=benchmark_rate
        self.benchmark_log_return=benchmark_log_return
        self.dates=self.data['date'].values.astype(np.datetime64)
        self.prices=self.data['price'].values.astype(np.float32)
        self._return=self.data[return_field].dropna().values.astype(np.float32)
        self._log_return=self.data[f'log {return_field}'].dropna().values.astype(np.float32)
        self.years_past=self._get_year_past()
        self.entries_per_year=len(self.data)/self.years_past
        self.information_payload={}

        self.drawdown_method=drawdown_method
        self.drawdown_evaluator={
            'dollar': lambda price, peak: peak - price,
            'percent': lambda price, peak: 1 - price / peak,
            'log': lambda price, peak: np.log(peak / price)
        }
        self.calmer_years=calmer_years


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


    def get_max_drawdown(self,method='log', get_return=False):
        assert method in self.drawdown_evaluator, f'Method: "{method}" must by one of {list(self.drawdown_evaluator.keys())}'
        output=self._get_max_drawdown(
            self.dates,
            self.prices,
            method,
            self.drawdown_evaluator,
        )
        if not get_return:
            self.information_payload['max_drawdown']=output
        else:
            return output


    def get_log_max_drawdown_ratio(self):
        log_max_drawdown=self.get_max_drawdown(method='log', get_return=True)
        self.information_payload['log_max_drawdown_ratio']=(np.log(self.prices[-1]) - np.log(self.prices[0])) - log_max_drawdown['max_drawdown']


    def get_calmer_ratio(self):
        first_day=self.data.iloc[-1]['date'] - timedelta(days=365.25*self.calmer_years)
        _filtered_series=self.data[self.data.date > first_day]
        _prices=_filtered_series.price.values.astype(np.float32)
        _dates=_filtered_series.date.values.astype(np.datetime64)
        _cagr=self._get_cagr(_prices, self.calmer_years)
        _max_drawdown=self._get_max_drawdown(
            _dates,
            _prices,
            'percent',
            self.drawdown_evaluator,
        )["max_drawdown"]
        self.information_payload['calmer_ratio']=_cagr / _max_drawdown


    def get_pure_profit_score(self):
        if 'cagr' not in self.information_payload: self.get_cagr()

        lr=LinearRegression()
        x, y=np.arange(len(self.prices)).reshape([-1,1]), self.prices.reshape([-1,1])
        fitted=lr.fit(x,y)
        r2=fitted.score(x,y)
        self.information_payload['pps']=self.information_payload['cagr'] * r2


    def get_jensens_alpha(self):
        lr=LinearRegression()
        x, y = self.benchmark_log_return.reshape([-1,1]), self._log_return.reshape([-1,1])
        fitted=lr.fit(x,y)
        self.information_payload['jensens_alpha']=fitted.intercept_[0]


    def get_payload(
        self,
    ):
        if 'volatility' not in self.information_payload: self.get_volatility()
        if 'cagr' not in self.information_payload: self.get_cagr()
        if 'sharpe' not in self.information_payload: self.get_sharpe()
        if 'sortino_ratio' not in self.information_payload: self.get_sortino_ratio()
        if 'max_drawdown' not in self.information_payload: self.get_max_drawdown(method=self.drawdown_method)
        if 'log_max_drawdown_ratio' not in self.information_payload: self.get_log_max_drawdown_ratio()
        if 'calmer_ratio' not in self.information_payload: self.get_calmer_ratio()
        if 'pps' not in self.information_payload: self.get_pure_profit_score()
        if 'jensens_alpha' not in self.information_payload: self.get_jensens_alpha()
        return self.information_payload
