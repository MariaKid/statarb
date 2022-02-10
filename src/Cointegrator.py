import numpy as np
import queue as q
import multiprocessing as mp

from enum import Enum, unique
from typing import Dict, Tuple, List
from numpy import array
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import adfuller

from src.DataRepository import DataRepository
from src.DataRepository import Universes
from src.Window import Window
from src.Features import Features
from src.Tickers import Tickers, SnpTickers, EtfTickers
from src.Kalman import kalman

# modules needed for testing cointegrator
from src.Clusterer import Clusterer
from datetime import date, timedelta


@unique
class AdfPrecisions(Enum):
    ONE_PCT = r'1%'
    FIVE_PCT = r'5%'
    TEN_PCT = r'10%'


class CointegratedPair:
    def __init__(self,
                 pair: List[Tickers],
                 beta: float,
                 intercept: float,
                 z_score: float,
                 adf_tstat: float,
                 hurst_stat: float,
                 cointegration_rank: float):
        self.pair: List[Tickers] = pair
        self.beta: float = beta
        self.intercept: float = intercept
        self.z_score: float = z_score
        self.adf_tstat: float = adf_tstat
        self.hurst_stat: float = hurst_stat
        self.cointegration_rank: float = cointegration_rank


class Cointegrator:
    def __init__(self,
                 repository: DataRepository,
                 adf_confidence_level: AdfPrecisions,
                 entry_z: float,
                 exit_z: float):
        self.repository: DataRepository = repository
        self.adf_confidence_level: AdfPrecisions = adf_confidence_level
        self.entry_z: float = entry_z
        self.exit_z: float = exit_z

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

    def regression_tests(self, current_window: Window, test_queue, results_queue,
                         hurst_threshold: float, zero_crossings_ratio: int, adf_lag: int) -> None:
        """
        Fetches SNP + ETF pairs from queue, performs cointegration test and puts pair in queue if cointegration test
        passed

        Parameters:
            current_window: time window
            hurst_threshold: threshold for hurst exponent test
            zero_crossings_ratio: divides by window length to find threshold number of zero crossings
            test_queue: queue of pairs to be tested for cointegration
            results_queue: queue of cointegrated pairs
            adf_lag: lag in adf test set to length of lookback window
        """
        while not test_queue.empty():
            try:
                # timeout prevents deadlock between threads when queue is empty
                pair = test_queue.get(timeout=1)
            except q.Empty:
                break
            print(pair)
            t_snp = current_window.get_data(universe=Universes.SNP,
                                            tickers=[pair[0]],
                                            features=[Features.CLOSE])
            t_etf = current_window.get_data(universe=Universes.ETFs,
                                            tickers=[pair[1]],
                                            features=[Features.CLOSE])
            try:
                residuals, beta, intercept = self.__lin_reg(x=t_etf, y=t_snp)
            except:
                continue

            adf_test_statistic, adf_critical_values = self.__adf(residuals=residuals, adf_lag=adf_lag)
            if adf_test_statistic < adf_critical_values[self.adf_confidence_level.value] and (beta > 0):
                hurst_test = self.__hurst_exponent_test(residuals, current_window)
                zero_crossings = np.where(np.diff(np.sign(residuals)))[0]
                if hurst_test < hurst_threshold and len(zero_crossings) > int(current_window.window_length.days/zero_crossings_ratio):
                    snp_current_price = t_snp.iloc[-1, 0]
                    etf_current_price = t_etf.iloc[-1, 0]
                    z = np.log(snp_current_price) - beta * np.log(etf_current_price) - intercept
                    cointegration_rank = self.__score_coint(t_stat=adf_test_statistic,
                                                            confidence_level=self.adf_confidence_level,
                                                            crit_values=adf_critical_values)
                    cointegrated_pair = CointegratedPair(pair=pair, intercept=intercept, beta=beta,
                                                         z_score=float(z), adf_tstat=adf_test_statistic,
                                                         hurst_stat=hurst_test, cointegration_rank=cointegration_rank)
                    results_queue.put(cointegrated_pair)

    def parallel_generate_pairs(self, clustering_results: Dict[int, List],
                                current_window: Window, hurst_threshold: float,
                                zero_crossings_ratio: int, adf_lag: int) -> Tuple[List, List]:
        """
        Parameters:
            clustering_results: dictionary of clusters
            current_window: time window
            hurst_threshold: threshold for hurst exponent test
            zero_crossings_ratio: divides by window length to find threshold number of zero crossings
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
                                                                         zero_crossings_ratio, adf_lag))
            processes.append(new_process)
            new_process.start()

        for p in processes:
            p.join()

        current_cointegrated_pairs = []
        for k in range(result_queue.qsize()):
            current_cointegrated_pairs.append(result_queue.get())

        current_cointegrated_pairs = sorted(current_cointegrated_pairs,
                                            key=lambda coint_pair: coint_pair.cointegration_rank, reverse=True)

        selected_cointegrated_pairs = []
        for p in current_cointegrated_pairs:
            if abs(p.z_score) > self.entry_z:
                selected_cointegrated_pairs.append(p)

        return current_cointegrated_pairs, selected_cointegrated_pairs

    @staticmethod
    def __lin_reg(x: DataFrame, y: DataFrame) -> Tuple[array, array, float]:
        """
        Runs linear regression of y on x

        Returns:
            residuals: regression residuals
            residuals_sum: cumulative residuals
            beta: beta of regression
        """
        x = np.array(np.log(x))
        y = np.array(np.log(y))

        reg_output = LinearRegression().fit(x, y)
        residuals = y - reg_output.predict(x)
        residuals = np.concatenate(residuals, axis=0)
        beta = float(reg_output.coef_[0])
        intercept = reg_output.intercept_

        return residuals, beta, intercept

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











