import math
import numpy as np
import os
import pandas as pd

from enum import Enum, unique
from fractions import Fraction
from typing import Dict, Tuple, List
from datetime import date, timedelta
from numpy import array
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import adfuller

from src.DataRepository import DataRepository
from src.DataRepository import Universes
from src.Clusterer import Clusterer
from src.Window import Window
from src.Features import Features
from src.Tickers import Tickers, SnpTickers, EtfTickers
import multiprocessing as mp


@unique
class AdfPrecisions(Enum):
    ONE_PCT = r'1%'
    FIVE_PCT = r'5%'
    TEN_PCT = r'10%'


class CointegratedPair:
    def __init__(self,
                 pair: List[Tickers],
                 beta: float,
                 rev_speed: float,
                 ou_mu: float,
                 ou_theta: float,
                 ou_sigma: float,
                 z_score: float,
                 cointegration_rank: float):
        self.pair: List[Tickers] = pair
        self.beta: float = beta
        self.rev_speed: float = rev_speed
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.z_score: float = z_score
        self.cointegration_rank: float = cointegration_rank


class Cointegrator:
    def __init__(self,
                 max_coint_queue_size: int,
                 repository: DataRepository,
                 adf_confidence_level: AdfPrecisions,
                 max_mean_rev_time: int,
                 entry_z: float,
                 exit_z: float):
        self.max_coint_queue_size: int = max_coint_queue_size
        self.repository: DataRepository = repository
        self.adf_confidence_level: AdfPrecisions = adf_confidence_level
        self.max_mean_rev_time: int = max_mean_rev_time
        self.entry_z: float = entry_z
        self.exit_z: float = exit_z

    def load_queue(self, clustering_results: Dict[int, List]):
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

    def regression_tests(self, hurst_exp_threshold: float, current_window: Window, queue, results_queue) -> None:
        """
        Fetches SNP + ETF pairs from queue, performs cointegration test and puts pair in queue if cointegration test
        passed

        Parameters:
            hurst_exp_threshold: threshold to pass Hurst exponent test
            current_window: time window
            queue: queue of pairs to be tested for cointegration
            results_queue: queue of cointegrated pairs
        """
        while not queue.empty():
            try:
                pair = queue.get()
                # print("Process {} got pair {} from the initial queue".format(os.getpid(), pair))
                t_snp = current_window.get_data(universe=Universes.SNP,
                                                tickers=[pair[0]],
                                                features=[Features.CLOSE])
                t_etf = current_window.get_data(universe=Universes.ETFs,
                                                tickers=[pair[1]],
                                                features=[Features.CLOSE])

                if not len(t_snp.index) == len(t_etf.index):
                    t_etf = t_etf[~t_etf.index.duplicated(keep='first')]
                    t_snp = t_snp[~t_snp.index.duplicated(keep='first')]

                try:
                    #print("Process {} performs the regression".format(os.getpid()))
                    residuals, x_t, beta = self.__lin_reg(t_etf, t_snp)
                    #print("Regression successfully completed")
                except:
                    #print("Regression error!")
                    continue

                adf_test_statistic, adf_critical_values = self.__adf(x_t.flatten())
                #print("Completed the adf test with adf_test_statistic {}".format(adf_test_statistic))
                if adf_test_statistic < adf_critical_values[self.adf_confidence_level.value] and (beta > 0):
                    he_test = self.__hurst_exponent_test(x_t, current_window)
                    if he_test < hurst_exp_threshold:
                        mu, theta, sigma, s = self.__ou_params(x_t)
                        if theta == False:
                            continue
                        rev_speed = 1 / theta
                        if rev_speed < (self.max_mean_rev_time / 252):
                            cointegration_rank = self.__score_coint(adf_test_statistic, self.adf_confidence_level,
                                                                    adf_critical_values, he_test, hurst_exp_threshold,
                                                                    10)
                            if not results_queue.full():
                                cointegrated_pair = CointegratedPair(pair, beta, rev_speed, mu, theta, sigma, s,
                                                                     cointegration_rank)
                                results_queue.put(cointegrated_pair)
                                #print("Pair {} added to the results queue".format(cointegrated_pair))
                            else:
                                #print("Results queue full")
                                break
            except:
                break

    def parallel_generate_pairs(self, clustering_results: Dict[int, List],
                                hurst_exp_threshold: float, current_window: Window) -> Tuple[List, List]:
        """
        Parameters:
            clustering_results: dictionary of clusters
            hurst_exp_threshold: threshold to pass Hurst exponent test
            current_window: time window

        Returns:
             current_cointegrated_pairs: list of all cointegrated pairs
             selected_cointegrated_pairs: list of cointegrated pairs with z_score > z_entry ranked according to their
             cointegration score
        """
        current_cointegrated_pairs = []
        num_processes = mp.cpu_count()
        processes = []
        pair_queue = self.load_queue(clustering_results)
        result_queue = mp.Manager().Queue(maxsize=self.max_coint_queue_size)

        for i in range(num_processes):
            new_process = mp.Process(target=self.regression_tests, args=(hurst_exp_threshold, current_window,
                                                                         pair_queue, result_queue))
            processes.append(new_process)
            new_process.start()

        for p in processes:
            p.join()

        for k in range(result_queue.qsize()):
            current_cointegrated_pairs.append(result_queue.get())

        current_cointegrated_pairs = sorted(current_cointegrated_pairs,
                                            key=lambda coint_pair: coint_pair.cointegration_rank, reverse=True)
        selected_cointegrated_pairs = []
        for pa in current_cointegrated_pairs:
            if abs(pa.z_score) > self.entry_z:
                selected_cointegrated_pairs.append(pa)

        return current_cointegrated_pairs, selected_cointegrated_pairs

    def __lin_reg(self, x: DataFrame, y: DataFrame) -> Tuple[array, array, float]:
        """
        Runs linear regression of y on x

        Returns:
            residuals: regression residuals
            residuals_sum: cumulative residuals
            beta: beta of regression
        """
        x = x.pct_change().values[1:]
        y = y.pct_change().values[1:]

        try:
            reg_output = LinearRegression().fit(x, y)
        except Exception as e:
            print(e)
        #print(reg_output)
        residuals = np.array(y - reg_output.predict(x))
        #print("Regression Residuals: {}".format(residuals[:3, :]))
        beta = float(reg_output.coef_[0])
        #print("Regression beta {}".format(beta))
        residuals_sum = []
        k = 0
        for res in residuals:
            k += float(res)
            residuals_sum.append(k)
        #print("Residuals sum {}".format(residuals_sum))
        return residuals, np.c_[residuals_sum], beta

    def __adf(self, residuals: array) -> Tuple[float, Dict[str, float]]:
        """
        Performs Augmented Dickey-Fuller test (with constant but no trend)

        Returns:
            adf_test_statistic: t-statistic
            adf_critical_values: dictionary of critical values at 1%, 5%, 10% confidence levels
        """
        adf_results = adfuller(residuals)
        adf_test_statistic: float = adf_results[0]
        adf_critical_values: Dict[str, float] = adf_results[4]

        return adf_test_statistic, adf_critical_values

    def __ou_params(self, residuals: array) -> Tuple[float, float, float, float]:
        """
        Ornstein–Uhlenbeck process fitted to cumulative residuals of cointegrated pair regression

        Parameters:
            residuals: cumulative residuals

        Returns:
            ou_mu: OU mean
            ou_theta: OU theta (speed of reversion parameter)
            ou_sigma: OU sigma (volatility)
            z_score: z_score
        """
        lagged_residuals = residuals[:-1]
        residuals = residuals[1:]
        reg = LinearRegression().fit(lagged_residuals, residuals)
        errors = residuals - reg.predict(lagged_residuals)

        a = float(reg.intercept_)
        b = float(reg.coef_[0])

        ou_mu = a / (1 - b)
        try:
            ou_theta = -252 * math.log(b)
        except:
            ou_sigma = False
            ou_mu = False
            ou_theta = False
            z_score = False

            return ou_mu, ou_theta, ou_sigma, z_score

        ou_sigma = math.sqrt(float(np.var(errors)) * 2 * ou_theta / (1 - (b ** 2)))
        sigma_eq = math.sqrt(float(np.var(errors)) / (1 - (b ** 2)))
        z_score = -ou_mu / sigma_eq

        return ou_mu, ou_theta, ou_sigma, z_score

    def __hurst_exponent_test(self, residuals, current_window: Window) -> float:
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
        max_lags = int(current_window.window_length.days * 0.5)

        for lag in range(2, max_lags):
            tau_vector.append(lag)
            delta_res = residuals[lag:] - residuals[:-lag]
            variance_delta_vector.append(np.var(delta_res))

        variance_delta_vector = np.log(variance_delta_vector)
        tau_vector = np.log(tau_vector)

        # Avoid 0 values for variance_delta_vector
        # variance_delta_vector = [value if value != 0 else 1e-10 for value in variance_delta_vector]

        reg_output = LinearRegression().fit(tau_vector.reshape(-1, 1), variance_delta_vector.reshape(-1, 1))
        beta = float(reg_output.coef_[0])
        hurst_exp = beta / 2

        return hurst_exp

    def __score_coint(self, t_stat: float,
                      confidence_level: AdfPrecisions,
                      crit_values: Dict[str, float],
                      hurst_stat: float,
                      hurst_threshold: float,
                      n_pair):
        """?"""
        dist = abs(t_stat - crit_values[confidence_level.value])
        hurst = abs(hurst_stat - hurst_threshold)
        weights = Fraction.from_float(-0.0042 * n_pair ** 2 + 0.1683 * n_pair + 0.1238).limit_denominator(
            max_denominator=1000000)
        delta = weights.numerator * dist + weights.denominator * hurst
        return delta


if __name__ == "__main__":
    clusterer = Clusterer()
    win = Window(window_start=date(2008, 10, 3), trading_win_len=timedelta(days=60), repository=DataRepository())
    print("Clustering...")
    clusters = clusterer.dbscan(eps=0.1, min_samples=2, window=win)
    print("Clustering completed...")

    c = Cointegrator(150, DataRepository(), AdfPrecisions.ONE_PCT, 10, 1.25, 0.6)
    cointegrated_pairs = c.parallel_generate_pairs(clusters, 0.2, win)
    for pair in cointegrated_pairs[1]:
        print(pair.pair, pair.beta, pair.z_score)
