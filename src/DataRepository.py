import math
import re
import numpy as np
import pandas as pd

from datetime import date
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Optional, Set, List
from pandas import DataFrame, IndexSlice
from pandas import Series as Se

from src.Features import Features
from src.Tickers import EtfTickers, SnpTickers, Tickers


@unique
class Universes(Enum):
    ETFs = Path(f"../resources/all_etfs2.csv")
    SNP = Path(f"../resources/all_snp2.csv")


class DataRepository:
    def __init__(self):
        self.all_data: Dict[Universes, Optional[DataFrame]] = {Universes.SNP: None, Universes.ETFs: None}
        self.tickers: Dict[Universes, Optional[Set[Tickers]]] = {Universes.SNP: None, Universes.ETFs: None}
        self.features: Dict[Universes, Optional[Set[Features]]] = {Universes.SNP: None, Universes.ETFs: None}
        self.all_dates: List[date] = self.__load_dates()

    def get(self,
            datatype: Universes,
            trading_dates: List[date]):
        self.__get_from_disk_and_store(datatype, trading_dates)

    def remove_dead_tickers(self, datatype: Universes, alive_and_dead_ticker_data: DataFrame) -> (List[Tickers], DataFrame):
        """
        Cleans the given ticker data by removing columns (tickers) with NaN values

        Parameters:
            datatype: Universes.SNP or Universes.ETF
            alive_and_dead_ticker_data: ticker data

        Returns:
            alive_tickers, alive_ticker_data: tuple with list of live tickers and portion of initial dataframe with
            live tickers
        """
        alive_tickers = [i for i in self.tickers[datatype]]
        junk_val = 'XXXXX'

        for idx, ticker in enumerate(self.tickers[datatype]):
            # Fetch first column of data for ticker (feature e.g. close price, volume... is irrelevant)
            column = alive_and_dead_ticker_data.loc[:, ticker].iloc[:, 0]
            # Check data in column is NaN
            is_nans = [True if math.isnan(i) else False for i in column]

            if any(is_nans):
                # Ticker is live for current window
                alive_tickers[idx] = junk_val

        # Remove tickers with all NaNs
        alive_tickers = [i for i in alive_tickers if i != junk_val]
        alive_ticker_data = alive_and_dead_ticker_data.loc[:, IndexSlice[alive_tickers, :]]

        return alive_tickers, alive_ticker_data

    def __load_dates(self) -> List[date]:
        """
        Returns:
             common_dates: list of dates for which both SNP and ETF data is available from the csv files
        """
        def _f(datatype: Universes):
            d = pd.read_csv(datatype.value,
                            squeeze=True,
                            header=0,
                            index_col=0,
                            usecols=[0])
            return [i.date() for i in pd.to_datetime(d.index, format='%d/%m/%Y')]

        self.all_dates_snp = sorted(set(_f(Universes.SNP)))
        self.all_dates_etf = sorted(set(_f(Universes.ETFs)))
        common_dates = sorted(set(_f(Universes.SNP)).intersection(set(_f(Universes.ETFs))))

        return common_dates

    def check_date_equality(self, d1: date, d2: date) -> bool:
        """
        Returns:
             True if day, month, year are the same
        """
        return (d1.day == d2.day and
                d1.month == d2.month and
                d1.year == d2.year)

    def __normalised_true_range(self, row: Se) -> float:
        """
        Returns:
            normalised true range defined by max(high-low, high-close, close-low)/close

        https://www.investopedia.com/terms/a/atr.asp
        """
        try:
            return max(row[1] - row[2], abs(row[1] - row[0]), abs(row[2] - row[0])) / row[0]

        except RuntimeError:
            return np.nan()

    def _intraday_vol(self, data: DataFrame, ticker) -> DataFrame:
        """
        Returns:
             data: dataframe with filled values for INTRADAY_VOL column defined as normalised true range
        """
        data.loc[:, IndexSlice[ticker, Features.INTRADAY_VOL]] \
            = np.apply_along_axis(self.__normalised_true_range, 1,
                                  data.loc[:, IndexSlice[ticker, [Features.HIGH, Features.LOW, Features.CLOSE]]])

        return data

    def __get_from_disk_and_store(self, datatype: Universes, trading_dates: List[date]):
        """
        Updates self.all_data for given asset group (ETF or SNP) for the time period specified

        Parameters:
            datatype: Universes.SNP or Universes.ETF
            trading_dates: time period specified
        """
        print(f'Getting data for dates {", ".join(str(d) for d in trading_dates)} in universe {datatype}')
        idxs_to_read = []

        # Find indexes corresponding to trading dates in ETF file
        if 'etf' in str(datatype.value):
            for d1 in trading_dates:
                for idx, d2 in enumerate(self.all_dates_etf):
                    if self.check_date_equality(d1, d2):
                        idxs_to_read.append(idx)
                        break

        # Find indexes corresponding to trading dates in SNP file
        else:
            for d1 in trading_dates:
                for idx, d2 in enumerate(self.all_dates_snp):
                    if self.check_date_equality(d1, d2):
                        idxs_to_read.append(idx)
                        break

        d = pd.read_csv(datatype.value,
                        squeeze=True,
                        header=0,
                        index_col=0,
                        skiprows=range(1, idxs_to_read[0] + 1),
                        low_memory=False,
                        nrows=len(idxs_to_read))

        d.index = pd.to_datetime(d.index, format='%d/%m/%Y')

        # Deconstruct column names e.g. A Adj Close to ['A', 'Adj', 'Close']
        match_results = [re.findall(r"(\w+)", col) for col in d.columns]

        if datatype is Universes.SNP:
            tickers = [SnpTickers(r[0].upper()) for r in match_results]
            features = [Features(r[-1].upper()) for r in match_results]
        else:
            tickers = [EtfTickers(r[0].upper()) for r in match_results]
            features = [Features(r[-1].upper()) for r in match_results]

        self.tickers[datatype] = set(tickers)
        self.features[datatype] = set(features)

        d.columns = pd.MultiIndex.from_tuples(
            tuples=list(zip(tickers, features)),
            names=['Ticker', 'Feature']
        )

        #d = d.fillna(method='ffill')
        d = self.forward_fill(d)

        if Features.INTRADAY_VOL not in self.features[datatype]:
            print("- Engineering intraday volatility...\n")
            for tick in self.tickers[datatype]:
                d.loc[:, IndexSlice[tick, Features.INTRADAY_VOL]] = self._intraday_vol(d, tick)

        d = d[d.index.isin(self.all_dates)]
        d = d.drop_duplicates(keep='first')

        self.all_data[datatype] = pd.concat([self.all_data[datatype], d], axis=0).drop_duplicates(keep='first')
        self.all_data[datatype] = self.all_data[datatype].drop_duplicates(keep='first')

    def _intraday_vol(self, data: DataFrame, ticker) -> DataFrame:
        """
        the function returns a pd.DataFrame with filled values for INTRADAY_VOL column defined as normalised true range
        """

        data.loc[:, IndexSlice[ticker, Features.INTRADAY_VOL]] \
            = np.apply_along_axis(self.__normalised_true_range, 1,
                                  data.loc[:, IndexSlice[ticker, [Features.HIGH, Features.LOW, Features.CLOSE]]])

        return data

    def __normalised_true_range(self, row: Se) -> float:
        """returns normalised true range as defined by:
        max(close-low, high-low,high-close)/close """
        try:

            ## original
            # # https://www.investopedia.com/terms/a/atr.asp - we use the formula for TR and then divide by close
            #return max(row[0] - row[0] , abs(row[0] - row[2]), abs(row[1] - row[2])) / row[2]

            ## new
            # https://www.investopedia.com/terms/a/atr.asp - we use the formula for TR and then divide by close
            return max(row[1] - row[2], abs(row[1] - row[0]), abs(row[2] - row[0])) / row[0]

        except RuntimeError:
            return np.nan()

    def __get_fundamental_from_disk(self):
        """the function updates the value of self.fundamental_data to pd.DataFrame
        containing the fundamental characteristics for each ticker"""
        data = pd.read_csv(Path(f"../resources/fundamental_snp.csv"),
                           index_col=0)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        fundamental_start = date(2016, 3, 31)
        fundamental_date = [date for date in self.all_dates if date > fundamental_start]
        df = pd.DataFrame(index=fundamental_date)
        df = df.join(data, how='outer')
        match_results = [re.findall(r"(\w+)", col) for col in df.columns]
        funda_tickers = [SnpTickers(r[0]) for r in match_results]
        funda_features = [r[1] for r in match_results]
        df.columns = pd.MultiIndex.from_tuples(
            tuples=list(zip(funda_tickers, funda_features)),
            names=['ticker', 'feature'])
        df = df.fillna(method='ffill')
        self.fundamental_data = df
        return

    def forward_fill(self, df: DataFrame) -> DataFrame:
        """the function returns a pd.DataFrame with values filled forwards"""
        return pd.DataFrame(df).fillna(method='ffill')

