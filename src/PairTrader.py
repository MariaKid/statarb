import logging
import time
from datetime import date, timedelta, datetime
from typing import Optional

from src.Clusterer import Clusterer
from src.Cointegrator import Cointegrator, AdfPrecisions
from src.DataRepository import DataRepository
from src.Filters import Filters
from src.Portfolio import Portfolio
from src.SignalGenerator import SignalGenerator
from src.Window import Window
from src.Kalman import parallel_kalman


class PairTrader:
    def __init__(self,
                 logger: logging.Logger,
                 backtest_start: date = date(2009, 10, 6),
                 max_active_pairs: int = 10,
                 trading_window_length: timedelta = timedelta(days=90),
                 adf_lag: int = int((90/2)-3),
                 trading_freq: timedelta = timedelta(days=10),
                 backtest_end: Optional[date] = None,
                 adf_confidence_level: AdfPrecisions = AdfPrecisions.ONE_PCT,
                 hurst_exp_threshold: float = 0.3,
                 emergency_delta_spread: float = 1):

        self.logger: logging.Logger = logger
        self.backtest_start: date = backtest_start
        self.max_active_pairs: int = max_active_pairs
        self.window_length: timedelta = trading_window_length
        self.adf_lag: int = adf_lag
        self.trading_freq: timedelta = trading_freq
        self.backtest_end = backtest_end
        self.adf_confidence_level: AdfPrecisions = adf_confidence_level
        self.hurst_exp_threshold: float = hurst_exp_threshold
        self.emergency_delta_spread: float = emergency_delta_spread

        self.logger.info("Creating new portfolio...")

        self.repository = DataRepository()
        self.current_window: Window = Window(window_start=self.backtest_start,
                                             trading_win_len=self.window_length,
                                             repository=self.repository)

        self.today = self.current_window.window_end
        print(self.today)
        self.day_count: int = 0

        self.clusterer = Clusterer()
        self.cointegrator = Cointegrator(repository=self.repository,
                                         adf_confidence_level=self.adf_confidence_level)

        self.filters = Filters()

        self.portfolio: Portfolio = Portfolio(cash=100_000,
                                              window=self.current_window,
                                              max_active_pairs=self.max_active_pairs,
                                              logger=self.logger)

        self.dm = SignalGenerator(port=self.portfolio,
                                  time_stop_loss=60,
                                  emergency_delta_spread=self.emergency_delta_spread)

        self.counter: int = 0

    def trade(self):

        while self.today < self.backtest_end:
            print(f"Today: {self.today.strftime('%Y-%m-%d')}\t"
                  f"Win Start: {self.current_window.window_start.strftime('%Y-%m-%d')}\t"
                  f"Win End: {self.current_window.window_end.strftime('%Y-%m-%d')}\n")

            if self.counter % self.trading_freq.days == 0:
                self.counter = 0

                print("Clustering...")
                clusters = self.clusterer.dbscan(eps=0.1, min_samples=2, window=self.current_window)
                # eps = 0.2

                print("Cointegrating...")
                cointegrated_pairs = self.cointegrator.parallel_generate_pairs(clustering_results=clusters,
                                                                               current_window=self.current_window,
                                                                               adf_lag=self.adf_lag,
                                                                               hurst_threshold=self.hurst_exp_threshold)

                print(cointegrated_pairs)

                print("Kalman filtering...")
                kalman_parallel_pairs = parallel_kalman(current_cointegrated_pairs=cointegrated_pairs,
                                                        start_date=self.backtest_start, end_date=self.today,
                                                        window=self.current_window)

                print(kalman_parallel_pairs)

                print("Making decisions...")
                decisions = self.dm.make_decision(kalman_parallel_pairs, self.backtest_start)
                print("Executing...")
                self.portfolio.execute_trades(decisions)
                print("active pairs: ", self.portfolio.number_active_pairs)

            print("Evolving...")
            self.__evolve()

        for position in self.portfolio.cur_positions:
            self.portfolio.close_position(position)

        self.portfolio.update_portfolio(self.backtest_end)

        self.portfolio.summary()
        # self.portfolio.get_port_hist().to_csv('backtest_results' + self.portfolio.timestamp)
        return

    def __evolve(self):
        self.portfolio.update_portfolio(self.today)
        self.day_count += 1
        self.counter += 1
        self.current_window.roll_forward_one_day()
        self.today = self.current_window.lookback_win_dates[-1]


if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(filename='log' + datetime.now().strftime("%Y%M%d%H%M%S"),
                        filemode='a',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    global_logger = logging.getLogger(__name__)

    PairTrader(
        backtest_start=date(2009, 10, 6),
        logger=global_logger,
        backtest_end=date(2015, 1, 2)
    ).trade()

    print(f"Backtest took {time.time() - start_time:.4f}s to run.")
