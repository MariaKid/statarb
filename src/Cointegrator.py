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

#modules needed for testing cointegrator
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
                 cointegration_rank: float):
        self.pair: List[Tickers] = pair
        self.beta: float = beta
        self.intercept: float = intercept
        self.z_score: float = z_score
        self.cointegration_rank: float = cointegration_rank


class Cointegrator:
    def __init__(self,
                 repository: DataRepository,
                 adf_confidence_level: AdfPrecisions,
                 max_rev_time: float,
                 entry_z: float,
                 exit_z: float):
        self.repository: DataRepository = repository
        self.adf_confidence_level: AdfPrecisions = adf_confidence_level
        self.max_rev_time: float = max_rev_time
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

    def regression_tests(self, current_window: Window, test_queue, results_queue) -> None:
        """
        Fetches SNP + ETF pairs from queue, performs cointegration test and puts pair in queue if cointegration test
        passed

        Parameters:
            current_window: time window
            test_queue: queue of pairs to be tested for cointegration
            results_queue: queue of cointegrated pairs
        """
        while not test_queue.empty():
            try:
                # timeout prevents deadlock between threads which can occur when queue is empty
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
            except Exception as e:
                print(e)
                continue

            adf_test_statistic, adf_critical_values = self.__adf(residuals=residuals)
            if adf_test_statistic < adf_critical_values[self.adf_confidence_level.value] and (beta > 0):
                #z score based on current prices
                snp_current_price = t_snp.iloc[-1, 0]
                erf_current_price =  t_etf.iloc[-1, 0]
                z = np.log(snp_current_price) - beta * np.log(erf_current_price) - intercept
                cointegration_rank = self.__score_coint(t_stat=adf_test_statistic,
                                                        confidence_level=self.adf_confidence_level,
                                                        crit_values=adf_critical_values)
                cointegrated_pair = CointegratedPair(pair=pair, intercept=intercept, beta=beta,
                                                     z_score=float(z), cointegration_rank=cointegration_rank)
                results_queue.put(cointegrated_pair)


    def parallel_generate_pairs(self, clustering_results: Dict[int, List],
                                current_window: Window) -> Tuple[List, List]:
        """
        Parameters:
            clustering_results: dictionary of clusters
            current_window: time window

        Returns:
             current_cointegrated_pairs: list of all cointegrated pairs
             selected_cointegrated_pairs: list of cointegrated pairs with z_score > z_entry ranked according to their
             cointegration score
        """
        num_processes = mp.cpu_count()
        processes = []
        pair_queue = self.load_queue(clustering_results=clustering_results)
        result_queue = mp.Manager().Queue()

        for i in range(num_processes):
            new_process = mp.Process(target=self.regression_tests, args=(current_window,
                                                                         pair_queue, result_queue))
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
        beta = float(reg_output.coef_[0])
        intercept = reg_output.intercept_

        return residuals, beta, intercept

    @staticmethod
    def __adf(residuals: array) -> Tuple[float, Dict[str, float]]:
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

    @staticmethod
    def __score_coint(t_stat: float, confidence_level: AdfPrecisions,
                      crit_values: Dict[str, float]) ->float:
        dist = abs(t_stat - crit_values[confidence_level.value])
        return dist


if __name__ == "__main__":
    win = Window(window_start=date(2008, 10, 3), trading_win_len=timedelta(days=60), repository=DataRepository())

    clusterer = Clusterer()
    #can we play around with eps?
    clusters = clusterer.dbscan(eps=0.1, min_samples=2, window=win)
    print(clusters)

    #spreads of cointegrated pairs with log prices lie on the range (-0.3, 0.3)
    c = Cointegrator(repository=DataRepository(), adf_confidence_level=AdfPrecisions.ONE_PCT,
                     max_rev_time=10, entry_z=0.15, exit_z=0.2)
    cointegrated_pairs = c.parallel_generate_pairs(clustering_results=clusters,  current_window=win)

    for pair in cointegrated_pairs[1]:
        print(pair.pair, round(float(pair.beta),3), round(float(pair.intercept),3), round(float(pair.z_score),5))