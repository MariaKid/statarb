from datetime import date, timedelta
from typing import Tuple

import src.Tickers as Tickers
from src.Tickers import SnpTickers
from src.DataRepository import DataRepository
from src.Window import Window
from src.Features import Features
from src.DataRepository import Universes
from src.Cointegrator import CointegratedPair


class Filters:
    def __init__(self, threshold_sigma: float = 1):
        self.current_window: Window = None
        # stdvs from mean to classify as shock
        self.threshold_sigma: float = threshold_sigma

    def run_volume_shock_filter_single_pair(self, pair: Tuple, current_window: Window):
        self.current_window = current_window
        first_shock = self.__is_volume_shock(pair[0])
        second_shock = self.__is_volume_shock(pair[1])
        if first_shock != second_shock:
            return True
        else:
            return False

    def run_volume_shock_filter(self, pairs, current_window):
        self.current_window = current_window
        reduced_pairs = pairs.copy()
        for pair in pairs:
            first_shock = self.__is_volume_shock(pair.pair[0])
            second_shock = self.__is_volume_shock(pair.pair[1])
            if first_shock != second_shock:
                reduced_pairs.remove(pair)
        return reduced_pairs

    def __is_volume_shock(self, ticker):
        if ticker in Tickers.EtfTickers:
            volume = self.current_window.get_data(Universes.ETFs, tickers=[ticker], features=[Features.VOLUME])
        elif ticker in Tickers.SnpTickers:
            volume = self.current_window.get_data(Universes.SNP, tickers=[ticker], features=[Features.VOLUME])

        hist_vol = volume[:-1]
        recent_vol = volume[-3:].mean()
        std = hist_vol.std()
        mean = hist_vol.mean()

        shock = (recent_vol > mean + self.threshold_sigma * std).values[0]
        if shock:
            return True
        else:
            return False


if __name__ == '__main__':
    win = Window(window_start=date(2008, 1, 2),
                 trading_win_len=timedelta(days=90),
                 repository=DataRepository(90))

    test_input = [
        CointegratedPair(pair=(SnpTickers.CRM, SnpTickers.WU),
                         mu_x_ann=0.1,
                         sigma_x_ann=0.3,
                         scaled_beta=1.2,
                         hl=7.5,
                         ou_mean=-0.017,
                         ou_std=0.043,
                         ou_diffusion_v=0.70,
                         recent_dev=0.071,
                         recent_dev_scaled=2.05,
                         beta=1,
                         reg_output=None,
                         recent_dev_scaled_hist=1,
                         cointegration_rank=1),

        CointegratedPair(pair=(SnpTickers.ABC, SnpTickers.ABT),
                         mu_x_ann=0.1,
                         sigma_x_ann=0.3,
                         scaled_beta=0.8,
                         hl=7.5,
                         ou_mean=-0.017,
                         ou_std=0.043,
                         ou_diffusion_v=0.70,
                         recent_dev=- 0.071,
                         recent_dev_scaled=-2.05,
                         beta=1,
                         reg_output=None,
                         recent_dev_scaled_hist=1,
                         cointegration_rank=1
                         )
    ]

    ft = Filters()
    test_result = ft.run_volume_shock_filter(test_input, current_window=win)
