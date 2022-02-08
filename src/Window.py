import sys
import pandas as pd
import numpy as np
from pandas import DataFrame

from datetime import date, timedelta
from typing import List, Optional
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import adfuller

from src.DataRepository import DataRepository, Universes
from src.Features import Features
from src.Tickers import Tickers, SnpTickers


class Window:
    def __init__(self, window_start: date, trading_win_len: timedelta, repository: DataRepository):
        self.window_start: date = window_start
        self.window_length: timedelta = trading_win_len
        self.repository: DataRepository = repository

        self.window_end = self.__get_nth_working_day_ahead(window_start, trading_win_len.days-1)
        self.lookback_win_dates = self.__get_window_trading_days(window_start, trading_win_len)
        self.__update_window_data(self.lookback_win_dates)

    def __get_window_trading_days(self, window_start: date, window_length: timedelta) -> List[date]:
        """
        Parameters:
            window_start = window start date
            window_length = window length in days

        Returns:
             window_trading_days: list of trading days formed by checking common dates with the repository's all_dates
             attribute (dates for which both SNP and ETF data is available from the csv files)
        """
        start_idx = None
        for idx, d in enumerate(self.repository.all_dates):
            if self.repository.check_date_equality(window_start, d):
                start_idx = idx
                break

        if start_idx is None:
            print("The window start date was not in the list of all dates.")
            print("Ensure backtest is started with a day that is in the dataset.")
            sys.exit()

        window_trading_days = self.repository.all_dates[start_idx: start_idx + window_length.days]
        return window_trading_days

    def __update_window_data(self, trading_dates_to_get_data_for: List[date]):
        """
        Creates two new attributes, self.etf_data and self.snp_data, by assigning them the DataFrame outputted from
        DataRepository.remove_dead_tickers()
        Checks if data has been uploaded, if not, uploads it from disk. If the window ends after ETF or SNP data's last
        day, it loads the data for the next window_length days

        Parameters:
            trading_dates_to_get_data_for = trading days range for data
        """
        # Remove duplicates
        trading_dates_to_get_data_for = sorted(set(trading_dates_to_get_data_for))

        # Load data if no data available
        if self.repository.all_data[Universes.SNP] is None or self.repository.all_data[Universes.ETFs] is None:
            self.repository.get(Universes.ETFs, trading_dates_to_get_data_for)
            self.repository.get(Universes.SNP, trading_dates_to_get_data_for)

        # Set date from which new data should be loaded if available
        next_load_start_date = min(max(self.repository.all_data[Universes.SNP].index),
                                   max(self.repository.all_data[Universes.ETFs].index))

        if self.window_end > next_load_start_date.date():
            look_forward_win_dates = self.__get_window_trading_days(next_load_start_date, self.window_length)
            self.repository.get(Universes.ETFs, look_forward_win_dates)
            self.repository.get(Universes.SNP, look_forward_win_dates)

        lookback_temp_etf_data = self.repository.all_data[Universes.ETFs].loc[trading_dates_to_get_data_for]
        lookback_temp_snp_data = self.repository.all_data[Universes.SNP].loc[trading_dates_to_get_data_for]

        # ignore the first value of the output of remove_dead_tickers() with '_'
        _, self.etf_data = self.repository.remove_dead_tickers(Universes.ETFs, lookback_temp_etf_data)
        _, self.snp_data = self.repository.remove_dead_tickers(Universes.SNP, lookback_temp_snp_data)

    def roll_forward_one_day(self):
        """
        Shifts the window to the next day and updates the data for the new window by calling self.__update_window_data
        """
        self.window_start = self.__get_nth_working_day_ahead(self.window_start, 1)
        self.window_end = self.__get_nth_working_day_ahead(self.window_end, 1)

        self.lookback_win_dates = self.__get_window_trading_days(self.window_start, self.window_length)
        # last window trading date should be today + 1 because today gets updated after this function gets called
        self.__update_window_data(self.lookback_win_dates)

    def __get_nth_working_day_ahead(self, target: date, n: int) -> date:
        """
        Parameters:
            target: starting date
            n: number of trading days ahead

        Returns:
             nth trading day ahead of target date
        """
        for idx, d in enumerate(self.repository.all_dates):
            if self.repository.check_date_equality(d, target):
                return self.repository.all_dates[idx + n]

        print("The window start date was not in the list if all dates.")
        print("Ensure backtest is started with a day that is in the dataset.")

    def get_data(self,
                 universe: Universes,
                 tickers: Optional[List[Tickers]] = None,
                 features: Optional[List[Features]] = None) -> DataFrame:
        """
        Gets data with tickers and features specified. Example:

        1. self.current_window.get_data(Universes.SNP, tickers=[SnpTickers.ALLE, SnpTickers.WU])
        2. all tickers' ['LOW', 'HIGH']
        self.current_window.get_data(Universes.SNP, features=[SnpFeatures.LOW, SnpFeatures.HIGH])
        3. ['BID','LOW'] for ['FITE','ARKW'], which is from ETF universe
        self.current_window.get_data(Universes.ETFs,
                                    tickers = [EtfTickers.FITE , EtfTickers.ARKW],
                                    features = [EtfFeatures.BID, EtfFeatures.LOW]

        Parameters:
            universe: Universes.SNP or Universes.ETFs
            tickers: list of Tickers or None, if None, return all tickers
            features: list of Features or None, if None, return all features
        """
        if tickers is None and features is None:
            if universe is Universes.SNP:
                return self.snp_data
            if universe is Universes.ETFs:
                return self.etf_data

        if universe is Universes.SNP:
            data = self.snp_data
        else:
            data = self.etf_data

        if tickers is not None and features is None:
            return data.loc[:, pd.IndexSlice[tickers, :]]
        elif tickers is None and features is not None:
            return data.loc[:, pd.IndexSlice[:, features]]
        elif tickers is not None and features is not None:
            return data.loc[:, pd.IndexSlice[tickers, features]]

    def get_fundamental(self):
        return self.repository.get_fundamental(self.window_end)


if __name__ == "__main__":
    win_length = timedelta(days=15)
    today = date(year=2008, month=1, day=2)
    window = Window(window_start=today,
                    trading_win_len=win_length,
                    repository=DataRepository())

    print(window.repository)
