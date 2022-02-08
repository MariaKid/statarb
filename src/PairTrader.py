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


class PairTrader:
    def __init__(self,
                 logger: logging.Logger,
                 target_number_of_coint_pairs: int = 100,
                 backtest_start: date = date(2008, 1, 2),
                 max_active_pairs: int = 10,
                 trading_window_length: timedelta = timedelta(days=60),
                 trading_freq: timedelta = timedelta(days=5),
                 backtest_end: Optional[date] = None,
                 adf_confidence_level: AdfPrecisions = AdfPrecisions.ONE_PCT,
                 max_mean_rev_time: int = 15,
                 hurst_exp_threshold: float = 0.20,
                 entry_z: float = 1.25,
                 emergency_delta_z: float = 2.5,
                 exit_z: float = 0.6):

        self.logger: logging.Logger = logger
        self.backtest_start: date = backtest_start
        self.max_active_pairs: int = max_active_pairs
        self.window_length: timedelta = trading_window_length
        self.trading_freq: timedelta = trading_freq
        self.backtest_end = backtest_end
        self.adf_confidence_level: AdfPrecisions = adf_confidence_level
        self.max_mean_rev_time: int = max_mean_rev_time
        self.hurst_exp_threshold: float = hurst_exp_threshold
        self.entry_z: float = entry_z
        self.emergency_delta_z: float = emergency_delta_z
        self.exit_z: float = exit_z

        self.logger.info("Creating new portfolio...")

        self.repository = DataRepository()
        self.current_window: Window = Window(window_start=backtest_start,
                                             trading_win_len=trading_window_length,
                                             repository=self.repository)
        self.today = self.current_window.lookback_win_dates[-1]
        self.day_count: int = 0

        self.clusterer = Clusterer()
        self.cointegrator = Cointegrator(max_coint_queue_size=target_number_of_coint_pairs,
                                         repository=self.repository,
                                         adf_confidence_level=self.adf_confidence_level,
                                         max_mean_rev_time=self.max_mean_rev_time,
                                         entry_z=self.entry_z,
                                         exit_z=self.exit_z)
        self.filters = Filters()
        self.portfolio: Portfolio = Portfolio(cash=100_000,
                                              window=self.current_window,
                                              max_active_pairs=self.max_active_pairs,
                                              logger=self.logger)
        self.dm = SignalGenerator(port=self.portfolio,
                                  entry_z=entry_z,
                                  exit_z=exit_z,
                                  emergency_delta_z=emergency_delta_z)

        self.counter: int = 0

    def trade(self):

        print("Clustering...")
        clusters = self.clusterer.dbscan(eps=0.1, min_samples=2, window=self.current_window)

        while self.today < self.backtest_end:
            print(f"Today: {self.today.strftime('%Y-%m-%d')}\t"
                  f"Win Start: {self.current_window.window_start.strftime('%Y-%m-%d')}\t"
                  f"Win End: {self.current_window.window_end.strftime('%Y-%m-%d')}\n")

            if self.counter % self.trading_freq.days == 0:
                self.counter = 0
                print("Cointegrating...")
                cointegrated_pairs = self.cointegrator.parallel_generate_pairs(clusters,
                                                                               self.hurst_exp_threshold,
                                                                               self.current_window)
                print("Making decisions...")
                decisions = self.dm.make_decision(cointegrated_pairs[0], cointegrated_pairs[1])
                print("Executing...")
                self.portfolio.execute_trades(decisions)
                print("active pairs: ", self.portfolio.number_active_pairs)

            print("Evolving...")
            self.__evolve()

        for position in self.portfolio.cur_positions:
            self.portfolio.close_position(position)

        self.portfolio.update_portfolio(self.backtest_end)

        self.portfolio.summary()
        self.portfolio.get_port_hist().to_csv('backtest_results' + self.portfolio.timestamp)
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
        backtest_start=date(2014, 1, 2),
        trading_window_length=timedelta(days=60),
        trading_freq=timedelta(days=10),
        target_number_of_coint_pairs=150,
        max_active_pairs=10,
        logger=global_logger,
        hurst_exp_threshold=0.20,
        backtest_end=date(2015, 1, 2),
        adf_confidence_level=AdfPrecisions.ONE_PCT,
        max_mean_rev_time=15,
        entry_z=1.25,
        exit_z=0.6,
        emergency_delta_z=2
    ).trade()

    print(f"Backtest took {time.time() - start_time:.4f}s to run.")
