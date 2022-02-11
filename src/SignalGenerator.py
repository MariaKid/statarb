from typing import List
import pandas as pd
from src.Portfolio import Portfolio, Position
from src.Features import PositionType, Features
from datetime import timedelta, date
from src.DataRepository import Universes
from src.Filters import Filters
from src.Kalman import kalman_single


class Decision:
    def __init__(self, position: Position, old_action: PositionType, new_action: PositionType):
        self.position: Position = position
        self.new_action: PositionType = new_action
        self.old_action: PositionType = old_action


class SignalGenerator:
    def __init__(self, port: Portfolio, time_stop_loss: int, emergency_delta_spread: float):
        self.port: Portfolio = port
        self.emergency_delta_spread: float = emergency_delta_spread
        self.time_stop_loss: int = time_stop_loss

        self.filter = Filters()
        self.natural_close_count: int = 0
        self.emergency_close_count: int = 0
        self.time_stop_loss_count: int = 0
        self.volume_shock_filter: int = 0

    def make_decision(self, kalman_parallel_pairs: List, backtest_start: date) -> List[Decision]:
        """
        Creates Decision object for each position to be used to execute trades. Decisions are either open, close or hold

        Parameters:
            kalman_parallel_pairs: list of all cointegrated pairs in latest rolling window with spread>spread_std
            backtest_start: start of backtest
        """
        today = self.port.current_window.window_end

        decisions = []
        positions = self.port.cur_positions
        current_posn_pairs = [[position.asset1, position.asset2] for position in positions]

        for position in positions:
            position_pair = [position.asset1, position.asset2]
            kalman_result = kalman_single(start_date=backtest_start, end_date=today, window=self.port.current_window,
                                          pair=position_pair)
            position_spread = kalman_result["spread"]
            position_spread_std = kalman_result["spread_std"]

            if today > (position.init_date + timedelta(self.time_stop_loss)) and (abs(position_spread) <
                                                                                  position_spread_std):
                decisions.append(Decision(position=position, old_action=position.position_type,
                                          new_action=PositionType.NOT_INVESTED))
                self.time_stop_loss_count += 1
            else:
                if position.position_type is PositionType.SHORT:
                    natural_close_required = position_spread < position_spread_std
                    emergency_close_required = position_spread > (position_spread_std + self.emergency_delta_spread)

                    if natural_close_required or emergency_close_required:
                        decisions.append(Decision(position=position, old_action=PositionType.SHORT,
                                                  new_action=PositionType.NOT_INVESTED))
                        if natural_close_required:
                            self.natural_close_count += 1
                            print("Close due to exit threshold", position_pair, "final spread: ",
                                  round(position_spread, 5), "final spread std: ", round(position_spread_std, 5))
                        else:
                            self.emergency_close_count += 1
                            print("Close due to emergency threshold", position_pair, "final spread: ",
                                  round(position_spread, 5), "final spread std: ", round(position_spread_std, 5))
                    else:
                        decisions.append(Decision(position=position, old_action=PositionType.SHORT,
                                                  new_action=PositionType.SHORT))

                elif position.position_type is PositionType.LONG:
                    natural_close_required = position_spread > -position_spread_std
                    emergency_close_required = position_spread < -(position_spread_std + self.emergency_delta_spread)

                    if natural_close_required or emergency_close_required:
                        decisions.append(Decision(position=position, old_action=PositionType.LONG,
                                                  new_action=PositionType.NOT_INVESTED))
                        if natural_close_required:
                            self.natural_close_count += 1
                            print("Close due to exit threshold ", position_pair, "final spread: ", round(position_spread, 5),
                                  "final spread std: ", round(position_spread_std, 5))
                        else:
                            self.emergency_close_count += 1
                            print("Close due to exit threshold", position_pair, "final spread: ", round(position_spread, 5),
                                  "final spread std: ", round(position_spread_std, 5))
                    else:
                        decisions.append(Decision(position=position, old_action=PositionType.LONG,
                                                  new_action=PositionType.LONG))

        for pair in kalman_parallel_pairs:
            pair_assets = pair[0]
            pair_spread = pair[1]
            pair_spread_std = pair[2]
            beta = pair[3][0]

            if beta < 0:
                continue

            if pair_assets not in current_posn_pairs:
                if pair_spread > pair_spread_std:
                    p1 = pair_assets[0]
                    p2 = pair_assets[1]
                    shock = self.filter.run_volume_shock_filter_single_pair(pair_assets, self.port.current_window)
                    if not shock:
                        df1 = self.port.current_window.get_data(universe=Universes.SNP, tickers=[p1],
                                                                features=[Features.CLOSE])
                        df2 = self.port.current_window.get_data(universe=Universes.ETFs, tickers=[p2],
                                                                features=[Features.CLOSE])

                        cur_price = pd.concat([df1, df2], axis=1)

                        price_snp = cur_price.iloc[-1, 0]
                        price_etf = cur_price.iloc[-1, 1]

                        total = price_snp + beta*price_etf
                        ratio = int(10000/total)

                        quantity_snp = ratio
                        quantity_etf = ratio*beta

                        decisions.append(
                            Decision(
                                position=Position(
                                    ticker1=p1,
                                    ticker2=p2,
                                    value1=-quantity_snp*price_snp,
                                    value2=quantity_etf*price_etf,
                                    investment_type=PositionType.SHORT,
                                    spread=pair_spread,
                                    spread_std=pair_spread_std,
                                    beta=beta,
                                    init_date=today),
                                old_action=PositionType.NOT_INVESTED,
                                new_action=PositionType.SHORT,
                            )
                        )
                    else:
                        self.volume_shock_filter += 1

                elif pair_spread < -pair_spread_std:
                    p1 = pair_assets[0]
                    p2 = pair_assets[1]
                    shock = self.filter.run_volume_shock_filter_single_pair(pair_assets, self.port.current_window)
                    if not shock:

                        df1 = self.port.current_window.get_data(universe=Universes.SNP, tickers=[p1],
                                                                features=[Features.CLOSE])
                        df2 = self.port.current_window.get_data(universe=Universes.ETFs, tickers=[p2],
                                                                features=[Features.CLOSE])

                        cur_price = pd.concat([df1, df2], axis=1)

                        price_snp = cur_price.iloc[-1, 0]
                        price_etf = cur_price.iloc[-1, 1]

                        total = price_snp + beta * price_etf
                        ratio = int(10000 / total)

                        quantity_snp = ratio
                        quantity_etf = ratio * beta

                        decisions.append(
                            Decision(
                                position=Position(
                                    ticker1=p1,
                                    ticker2=p2,
                                    value1=quantity_snp*price_snp,
                                    value2=-quantity_etf*price_etf,
                                    investment_type=PositionType.LONG,
                                    spread=pair_spread,
                                    spread_std=pair_spread_std,
                                    beta=beta,
                                    init_date=today
                                ),
                                old_action=PositionType.NOT_INVESTED,
                                new_action=PositionType.LONG,
                            )
                        )
                    else:
                        self.volume_shock_filter += 1

        print("natural close count: ", self.natural_close_count)
        print("emergency (z-score) close count: ", self.emergency_close_count)
        print("time stop-loss close count: ", self.time_stop_loss_count)
        return decisions
