from datetime import datetime, date, timedelta
from logging import Logger
from pandas import DataFrame, to_datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from src.DataRepository import Universes
from src.Position import Position
from src.Window import Window
from src.Features import Features, PositionType
from src.Performance import get_performance_stats


class Portfolio:
    def __init__(self, cash: float, window: Window, logger: Logger, max_active_pairs: int = 10):
        self.init_cash: float = cash
        self.cur_cash: float = cash
        self.current_window: Window = window
        self.logger: Logger = logger
        self.max_active_pairs: int = max_active_pairs

        self.cur_positions: List = list()
        self.hist_positions: List = list()
        self.total_capital: List = [cash]
        self.active_port_value: float = float(0)
        self.realised_pnl: float = float(0)
        self.log_return: float = float(0)
        self.cum_return: float = float(0)
        self.t_cost: float = float(0.005)
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.port_hist: List = list()
        self.loading: float = float(0.1)
        self.number_active_pairs: int = 0
        self.open_count: int = 0

        self.port_hist.append(
            [self.current_window.window_end + pd.DateOffset(-1), self.cur_cash, self.active_port_value,
             self.cur_cash + self.active_port_value, self.realised_pnl, self.log_return * 100,
             self.cum_return * 100, self.number_active_pairs])

    def open_position(self, position: Position) -> None:
        """
        Opens new position with quantities of assets according to prices at window end, checks if current cash is
        sufficient and number of active pairs not exceeded, appends to current portfolio positions, updates pair count,
        current cash and active portfolio value

        Parameters:
            position: Position object to store new position
        """
        if self.number_active_pairs < self.max_active_pairs:
            df_snp = self.current_window.get_data(universe=Universes.SNP, tickers=[position.asset1],
                                                  features=[Features.CLOSE])
            df_etf = self.current_window.get_data(universe=Universes.ETFs, tickers=[position.asset2],
                                                  features=[Features.CLOSE])

            cur_price = pd.concat([df_snp, df_etf], axis=1)
            price_snp = cur_price.iloc[-1, 0]
            price_etf = cur_price.iloc[-1, 1]

            position.quantity1 = int(position.value1 / price_snp)
            position.quantity2 = int(position.value2 / price_etf)

            asset1_value = price_snp*position.quantity1
            asset2_value = price_etf*position.quantity2

            commission = self.generate_commission(asset1_value, asset2_value)

            pair_dedicated_cash = asset1_value + asset2_value
            position.set_position_value(pair_dedicated_cash)

            if pair_dedicated_cash > self.cur_cash:
                self.logger.info('Not sufficient cash to open position')
            else:
                self.logger.info("%s, %s are cointegrated and spread is in trading range. Opening position....",
                                 position.asset1, position.asset2)
                self.number_active_pairs += 1
                self.open_count += 1

                self.cur_positions.append(position)
                self.hist_positions.append(position)

                self.cur_cash -= (pair_dedicated_cash + commission)
                self.active_port_value += pair_dedicated_cash

                self.logger.info('Asset 1: %s @$%s Quantity: %s Value: %s', position.asset1,
                                 round(cur_price.iloc[-1, 0], 2), round(position.quantity1, 2), round(asset1_value, 2))
                self.logger.info('Asset 2: %s @$%s Quantity: %s Value: %s', position.asset2,
                                 round(cur_price.iloc[-1, 1], 2), round(position.quantity2, 2), round(asset2_value, 2))
                self.logger.info('Cash balance: $%s', self.cur_cash)

                print("open a new position", position.asset1, round(asset1_value, 2), position.asset2,
                      round(asset2_value, 2), "spread: ", round(position.spread, 5),
                      "spread_std: ", round(position.spread_std, 5), position.position_type)
        else:
            pass

    def close_position(self, position: Position) -> None:
        """
        Closes open position, updates number of active pairs, current cash, active portfolio value and pnl

        Parameters:
            position: Position object to be closed
        """
        df1 = self.current_window.get_data(universe=Universes.SNP, tickers=[position.asset1], features=[Features.CLOSE])
        df2 = self.current_window.get_data(universe=Universes.ETFs, tickers=[position.asset2],
                                           features=[Features.CLOSE])
        cur_price = pd.concat([df1, df2], axis=1)

        if not (position in self.cur_positions):
            print("This option is not open")
        else:
            self.logger.info("Closing/emergency threshold is passed for active pair %s, %s. Closing position...",
                             position.asset1, position.asset2)
            self.number_active_pairs -= 1
            self.cur_positions.remove(position)

            asset1_value = cur_price.iloc[-1, 0]*position.quantity1
            asset2_value = cur_price.iloc[-1, 1]*position.quantity2
            commission = self.generate_commission(asset1_value, asset2_value)
            pair_residual_cash = asset1_value + asset2_value
            position.close_trade(pair_residual_cash, self.current_window)

            print("Close the position", position.asset1, position.asset2)

            self.cur_cash += (pair_residual_cash - commission)
            self.active_port_value -= pair_residual_cash
            self.realised_pnl += position.pnl

            self.logger.info('Asset 1: %s @$%s Quantity: %s', position.asset1,
                             round(cur_price.iloc[-1, 0], 2), int(position.quantity1))
            self.logger.info('Asset 2: %s @$%s Quantity: %s', position.asset2,
                             round(cur_price.iloc[-1, 1], 2), int(position.quantity2))
            print("pnl: ", round(position.pnl[0], 2))
            self.logger.info('Realised PnL for position: %s ', round(position.pnl[0], 2))

    def generate_commission(self, asset1_value: float, asset2_value: float) -> float:
        """
        Parameters:
            asset1_value: asset 1 value
            asset2_value: asset 2 value

        Returns:
            transaction cost as % of notional amount
        """
        return self.t_cost * (abs(asset1_value) + abs(asset2_value))

    def update_portfolio(self, today: date) -> None:
        """
        Loops through and updates current portfolio positions

        Parameters:
            today: current date
        """
        cur_port_val = 0

        for pair in self.cur_positions:
            df3 = self.current_window.get_data(universe=Universes.SNP, tickers=[pair.asset1],
                                               features=[Features.CLOSE]).loc[today].values
            df4 = self.current_window.get_data(universe=Universes.ETFs, tickers=[pair.asset2],
                                               features=[Features.CLOSE]).loc[today].values

            todays_prices = [df3, df4]
            asset_value = todays_prices[0] * pair.quantity1 + todays_prices[1] * pair.quantity2
            pair.update_position_pnl(asset_value, self.current_window)
            cur_port_val += asset_value

        self.active_port_value = cur_port_val
        self.total_capital.append(self.cur_cash + self.active_port_value)
        self.log_return = np.log(self.total_capital[-1]) - np.log(self.total_capital[-2])
        self.cum_return = np.log(self.total_capital[-1]) - np.log(self.total_capital[0])
        self.port_hist.append([self.current_window.window_end, self.cur_cash, self.active_port_value,
                               self.cur_cash + self.active_port_value, self.realised_pnl, self.log_return * 100,
                               self.cum_return * 100, self.number_active_pairs])

    def execute_trades(self, decisions):
        """
        For each decision, checks if new action is different from old action. If old action is not_invested,
        position is opened, if new action is not_invested, position is closed
        """
        self.logger.info(f"Executing trades for {self.current_window.window_end.strftime('%Y-%m-%d')}")
        for decision in decisions:
            if decision.old_action is not decision.new_action:
                if decision.old_action is PositionType.NOT_INVESTED:
                    self.open_position(decision.position)
                elif decision.new_action is PositionType.NOT_INVESTED:
                    self.close_position(decision.position)
        print("open_count: ", self.open_count)

    def get_port_summary(self) -> List:
        """
        Print current cash balance, dataframe with pairs assets, quantities and pnl, and portfolio pnl

        Returns:
            List of current cash, DataFrame of current positions, portfolio pnl
        """
        data = list()
        for pair in self.cur_positions:
            data.append([pair.asset1, pair.quantity1, pair.asset2, pair.quantity2, round(pair.pnl, 2)])

        df = DataFrame(data, columns=['Asset 1', 'Quantity 1', 'Asset 2', 'Quantity 2', 'PnL'])

        print('------------------Portfolio Summary------------------')
        print('Current cash balance: \n %s' % self.cur_cash)
        if len(data) != 0:
            print('Current Positions: ')
            print(df)
        else:
            print('No Current Positions')
        print('Realised PnL: \n %s' % round(self.realised_pnl, 2))
        print('-----------------------------------------------------')
        return [self.cur_cash, df, self.realised_pnl]

    def get_port_hist(self) -> DataFrame:
        """
        Returns:
            df: DataFrame (time series) of cash balance, portfolio value and pnl
        """
        pd.set_option('expand_frame_repr', False)
        df = DataFrame(self.port_hist, columns=['date', 'cash', 'port_value', 'total_capital',
                                                'realised_pnl', 'return',
                                                'cum_return', 'active_pairs'])
        df['date'] = to_datetime(df['date'])
        df = df.set_index('date')
        df = df.round(2)
        return df

    def summary(self):
        """
        Performance of portfolio
        """
        prc_hist = self.get_port_hist()['total_capital']

        tbill = pd.read_csv("../resources/3m_tbill_daily.csv", index_col='date',
                            date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y'))
        tbill = tbill.applymap(lambda x: x / 365)
        tbill = tbill.applymap(lambda x: x / 100)

        tbill.index += timedelta(1)
        tbill_mean = tbill.loc[tbill.index.intersection(prc_hist.index)].mean().values
        # print(get_performance_stats(prc_hist, tbill_mean))

        all_history = self.get_port_hist()
        sp = yf.download("^GSPC", start=min(all_history.index), end=max(all_history.index))[["Adj Close"]]["Adj Close"]

        all_history.index = [i.date() for i in all_history.index]
        sp.index = [i.date() for i in sp.index]

        common_dates = sorted(set(sp.index).intersection(set(all_history.index)))

        sp = sp[common_dates]
        all_history = all_history[all_history.index.isin(common_dates)]

        normalise = lambda series: series / (series[0] if int(series[0]) != 0 else 1.0)

        plt.figure(1, figsize=(10, 7))
        plt.plot(all_history.index, normalise(all_history["total_capital"]), label=r"Portfolio")
        plt.plot(all_history.index, normalise(sp), label=r"SnP 500")
        plt.xlabel("Date")
        plt.ylabel("Total Capital")
        plt.legend(loc=r"best")
        plt.tight_layout()
        plt.savefig("{time.time()}_total_capital.png", dpi=200)
        plt.show()
