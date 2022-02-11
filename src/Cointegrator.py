import numpy as np
import queue as q
import multiprocessing as mp

from enum import Enum, unique
from typing import Dict, Tuple, List
from numpy import array
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import adfuller
from warnings import simplefilter

from src.DataRepository import DataRepository
from src.DataRepository import Universes
from src.Window import Window
from src.Features import Features
from src.Tickers import Tickers, SnpTickers, EtfTickers
from src.Kalman import parallel_kalman

# modules needed for testing cointegrator
from src.Clusterer import Clusterer
from datetime import date, timedelta

simplefilter(action='ignore', category=FutureWarning)


@unique
class AdfPrecisions(Enum):
    ONE_PCT = r'1%'
    FIVE_PCT = r'5%'
    TEN_PCT = r'10%'


class CointegratedPair:
    def __init__(self,
                 pair: List[Tickers],
                 beta: float,
                 spread: float):
        self.pair: List[Tickers] = pair
        self.beta: float = beta
        self.spread: float = spread


class Cointegrator:
    def __init__(self,
                 repository: DataRepository,
                 adf_confidence_level: AdfPrecisions):
        self.repository: DataRepository = repository
        self.adf_confidence_level: AdfPrecisions = adf_confidence_level

    @staticmethod
    def load_queue(clustering_results: Dict[int, List]):
        """
        Parameters:
            clustering_results: dictionary of clusters

        Returns:
            queue: queue of SNP + ETF pairs to be tested for cointegration
        """
        manager = mp.Manager()
        queue = manager.Queue()

        tickers_per_cluster = [clustering_results[key] for key in clustering_results.keys() if key != -1]
        for cluster in tickers_per_cluster:
            for i in cluster:
                if type(i) == SnpTickers:
                    for j in reversed(cluster):
                        if type(j) == EtfTickers:
                            pair = [i, j]
                            queue.put(pair)
        return queue

    def regression_tests(self, current_window: Window, test_queue, results_queue, hurst_threshold: float,
                         adf_lag: int) -> None:
        """
        Fetches SNP + ETF pairs from queue, performs cointegration test and puts pair in queue if cointegration test
        passed

        Parameters:
            current_window: time window
            hurst_threshold: threshold for hurst exponent test
            test_queue: queue of pairs to be tested for cointegration
            results_queue: queue of cointegrated pairs (objects in queue in the form List[List[SnpTicker, EtfTicker],
            cointegration_rank [abs(t-stat - critical value)])
            adf_lag: lag in adf test set to length of lookback window
        """
        while not test_queue.empty():
            try:
                # timeout prevents deadlock between threads when queue is empty
                pair = test_queue.get(timeout=1)
            except q.Empty:
                break

            t_snp = current_window.get_data(universe=Universes.SNP,
                                            tickers=[pair[0]],
                                            features=[Features.CLOSE])
            t_etf = current_window.get_data(universe=Universes.ETFs,
                                            tickers=[pair[1]],
                                            features=[Features.CLOSE])

            # Added as sometimes a row is duplicated (likely a problem in src.DataRepository)
            t_etf = t_etf[~t_etf.index.duplicated(keep='first')]
            t_snp = t_snp[~t_snp.index.duplicated(keep='first')]

            try:
                residuals = self.__lin_reg(x=t_etf, y=t_snp)
            except:
                print(True)
                continue

            adf_test_statistic, adf_critical_values = self.__adf(residuals=residuals, adf_lag=adf_lag)
            if adf_test_statistic < adf_critical_values[self.adf_confidence_level.value]:
                hurst_test = self.__hurst_exponent_test(residuals, current_window)
                if hurst_test < hurst_threshold:
                    cointegration_rank = self.__score_coint(t_stat=adf_test_statistic,
                                                            confidence_level=self.adf_confidence_level,
                                                            crit_values=adf_critical_values)
                    results_queue.put([pair, cointegration_rank])

    def parallel_generate_pairs(self, clustering_results: Dict[int, List],
                                current_window: Window, hurst_threshold: float, adf_lag: int):
        """
        Parameters:
            clustering_results: dictionary of clusters
            current_window: time window
            hurst_threshold: threshold for hurst exponent test
            adf_lag: adf test lag

        Returns:
             current_cointegrated_pairs: list of all cointegrated pairs ranked according to t-stat
        """
        num_processes = mp.cpu_count()
        processes = []
        pair_queue = self.load_queue(clustering_results=clustering_results)
        result_queue = mp.Manager().Queue()

        for i in range(num_processes):
            new_process = mp.Process(target=self.regression_tests, args=(current_window,
                                                                         pair_queue, result_queue, hurst_threshold,
                                                                         adf_lag))
            processes.append(new_process)
            new_process.start()

        for p in processes:
            p.join()

        current_cointegrated_pairs = []
        for k in range(result_queue.qsize()):
            current_cointegrated_pairs.append(result_queue.get())

        # x[1] is the cointegration score
        current_cointegrated_pairs = sorted(current_cointegrated_pairs, key=lambda x: x[1], reverse=True)

        return current_cointegrated_pairs

    @staticmethod
    def __lin_reg(x: DataFrame, y: DataFrame) -> array:
        """
        Runs linear regression of y on x

        Returns:
            residuals: regression residuals
        """
        x = np.array(np.log(x))
        y = np.array(np.log(y))

        reg_output = LinearRegression().fit(x, y)
        residuals = y - reg_output.predict(x)
        residuals = np.concatenate(residuals, axis=0)

        return residuals

    @staticmethod
    def __adf(residuals: array, adf_lag: int) -> Tuple[float, Dict[str, float]]:
        """
        Performs Augmented Dickey-Fuller test (with constant but no trend)

        Returns:
            adf_test_statistic: t-statistic
            adf_critical_values: dictionary of critical values at 1%, 5%, 10% confidence levels
        """
        adf_results = adfuller(x=residuals, regression="c", maxlag=adf_lag, autolag=None)
        adf_test_statistic: float = adf_results[0]
        adf_critical_values: Dict[str, float] = adf_results[4]

        return adf_test_statistic, adf_critical_values

    @staticmethod
    def __hurst_exponent_test(residuals, current_window: Window) -> float:
        """
        Returns Hurst exponent by calculating variance of the difference between lagged returns and performing
        regression of the variance on the lag.
        https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e
        Parameters:
            residuals: regression residuals
            current_window: time window
        Returns
            hurst_exp: Hurst exponent
        """
        # Lag vector
        tau_vector = []
        # Variance vector
        variance_delta_vector = []
        max_lags = int(current_window.window_length.days*0.5)

        for lag in range(2, max_lags):
            tau_vector.append(lag)
            delta_res = residuals[lag:] - residuals[:-lag]
            variance_delta_vector.append(np.var(delta_res))

        variance_delta_vector = np.log(variance_delta_vector)
        tau_vector = np.log(tau_vector)

        reg_output = LinearRegression().fit(tau_vector.reshape(-1, 1), variance_delta_vector.reshape(-1, 1))
        beta = float(reg_output.coef_[0])
        hurst_exp = beta / 2

        return hurst_exp

    @staticmethod
    def __score_coint(t_stat: float, confidence_level: AdfPrecisions,
                      crit_values: Dict[str, float]) -> float:
        dist = abs(t_stat - crit_values[confidence_level.value])
        return dist


if __name__ == '__main__':

    lookback_win_len = 90
    adflag = int((lookback_win_len / 2) - 3)
    win = Window(window_start=date(2009, 10, 6), trading_win_len=timedelta(days=lookback_win_len),
                 repository=DataRepository())

    clusterer = Clusterer()
    clusters = clusterer.dbscan(eps=0.1, min_samples=2, window=win)

    c = Cointegrator(repository=DataRepository(), adf_confidence_level=AdfPrecisions.ONE_PCT)
    cointegrated_pairs = c.parallel_generate_pairs(clustering_results=clusters, current_window=win,
                                                   hurst_threshold=0.3, adf_lag=adflag)

    print(cointegrated_pairs)
