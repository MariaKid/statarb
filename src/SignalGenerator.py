from typing import List
from src.Cointegrator import CointegratedPair
from src.Portfolio import Portfolio, Position
from src.Features import PositionType
from datetime import timedelta
from src.Filters import Filters


class Decision:
    def __init__(self, position: Position, old_action: PositionType, new_action: PositionType):
        self.position: Position = position
        self.new_action: PositionType = new_action
        self.old_action: PositionType = old_action


class SignalGenerator:
    def __init__(self, port: Portfolio, entry_z: float, exit_z: float, emergency_delta_z: float):
        self.port: Portfolio = port
        self.entry_z: float = entry_z
        self.exit_z: float = exit_z
        self.emergency_delta_z: float = emergency_delta_z

        self.filter = Filters()
        self.time_stop_loss = 20
        self.natural_close_count = 0
        self.emergency_coint_close_count = 0
        self.emergency_close_count = 0
        self.time_stop_loss_count = 0
        self.volume_shock_filter = 0

    def make_decision(self, pairs: List[CointegratedPair], selected_pairs: List[CointegratedPair]) -> List[Decision]:
        """
        Creates Decision object for each position to be used to execute trades. Decisions are either open, close or hold

        Parameters:
            pairs: list of all cointegrated pairs in latest rolling window
            selected_pairs: cointegrated pairs with z_score > z_entry ranked according to their cointegration score
        """
        today = self.port.current_window.window_end

        # List of tickers of current position pairs
        positions = self.port.cur_positions
        current_posn_pairs = [[i.asset1, i.asset2] for i in positions]

        # List of tickers of cointegrated pairs
        coint_pairs = [i.pair for i in pairs]

        decisions = []

        # First loop through current positions
        for position in positions:
            position_pair = [position.asset1, position.asset2]
            # if pair not cointegrated, exit position
            if position_pair not in coint_pairs:
                decisions.append(Decision(position=position, old_action=position.position_type,
                                          new_action=PositionType.NOT_INVESTED))
                self.emergency_coint_close_count += 1
            else:
                idx = coint_pairs.index(position_pair)
                coint_pair = pairs[idx]
                # If time limit reached and recent_dev not reached exit_z, but trade is profitable, close trade
                # If time limit reached, but recent_dev between emergency_delta_z and entry z in absolute values (loss)
                # trade not closed
                # Add method to change z_score
                if today > (position.init_date + timedelta(self.time_stop_loss)) and \
                        (abs(coint_pair.z_score) < self.entry_z):
                    decisions.append(Decision(position=position, old_action=position.position_type,
                                              new_action=PositionType.NOT_INVESTED))
                    self.time_stop_loss_count += 1
                else:
                    if position.position_type is PositionType.SHORT:
                        natural_close_required = coint_pair.z_score < self.exit_z
                        emergency_close_required = coint_pair.z_score > \
                                                   (self.emergency_delta_z + position.init_z)

                        if natural_close_required or emergency_close_required:
                            decisions.append(Decision(position=position, old_action=PositionType.SHORT,
                                                      new_action=PositionType.NOT_INVESTED))
                            if natural_close_required:
                                self.natural_close_count += 1
                                print("Close due to exit threshold")
                                print(coint_pair.pair, "final z-score: ", round(coint_pair.z_score, 2))
                            else:
                                self.emergency_close_count += 1
                                print("Close due to emergency threshold")
                                print(coint_pair.pair, "final z-score: ", round(coint_pair.z_score, 2))
                        else:
                            # keep position open
                            decisions.append(Decision(position=position, old_action=PositionType.SHORT,
                                                      new_action=PositionType.SHORT))

                    elif position.position_type is PositionType.LONG:
                        natural_close_required = coint_pair.z_score > -self.exit_z
                        emergency_close_required = coint_pair.z_score < \
                                                   (position.init_z - self.emergency_delta_z)

                        if natural_close_required or emergency_close_required:
                            decisions.append(Decision(position=position, old_action=PositionType.LONG,
                                                      new_action=PositionType.NOT_INVESTED))
                            if natural_close_required:
                                self.natural_close_count += 1
                                print("Close due to exit threshold")
                                print(coint_pair.pair, "final z-score: ", round(coint_pair.z_score, 2))
                            else:
                                self.emergency_close_count += 1
                                print("close due to emergency threshold")
                                print(coint_pair.pair, "final z-score: ", round(coint_pair.z_score, 2))
                        else:
                            # no need to close, keep the position open
                            decisions.append(Decision(position=position, old_action=PositionType.LONG,
                                                      new_action=PositionType.LONG))
        for coint_pair in selected_pairs:
            # if coint_pair not invested, check if position needs to be opened
            if coint_pair.pair not in current_posn_pairs:
                if coint_pair.z_score > self.entry_z:
                    p1, p2 = coint_pair.pair
                    shock = self.filter.run_volume_shock_filter_single_pair(coint_pair.pair, self.port.current_window)
                    if not shock:
                        decisions.append(
                            # Short the spread
                            Decision(
                                position=Position(
                                    ticker1=p1,
                                    ticker2=p2,
                                    value1=-10000/(coint_pair.beta + 1),
                                    value2=10000*coint_pair.beta/(coint_pair.beta + 1),
                                    investment_type=PositionType.SHORT,
                                    init_z=coint_pair.z_score,
                                    init_date=today),
                                old_action=PositionType.NOT_INVESTED,
                                new_action=PositionType.SHORT,
                            )
                        )
                    else:
                        self.volume_shock_filter += 1

                elif coint_pair.z_score < -self.entry_z:
                    p1, p2 = coint_pair.pair
                    shock = self.filter.run_volume_shock_filter_single_pair(coint_pair.pair, self.port.current_window)
                    if not shock:
                        decisions.append(
                            Decision(
                                position=Position(
                                    ticker1=p1,
                                    ticker2=p2,
                                    value1=10000/(coint_pair.beta + 1),
                                    value2=-10000*coint_pair.beta/(coint_pair.beta + 1),
                                    investment_type=PositionType.LONG,
                                    init_z=coint_pair.z_score,
                                    init_date=today),
                                old_action=PositionType.NOT_INVESTED,
                                new_action=PositionType.LONG,
                            )
                        )
                    else:
                        self.volume_shock_filter += 1

        print("natural close count: ", self.natural_close_count)
        print("emergency (not cointegrated) close count: ", self.emergency_coint_close_count)
        print("emergency (z-score) close count: ", self.emergency_close_count)
        print("time stop-loss close count: ", self.time_stop_loss_count)
        return decisions
