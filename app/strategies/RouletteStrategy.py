from app.strategies.BaseStrategy import BaseStrategy, BaseStrategyParam
from app.datafeed.DataFeeder import DataFeeder

from app.models.TradeBotDataFeed import TradeBotDataFeed
from app.models.MarketCondition import MarketCondition
from app.models.TradeDecision import TradeDecision
from app.models.TradeLog import TradeLog

from collections.abc import Callable

class DecisionFactory:
    def BuyAndHold(data: TradeBotDataFeed):
        return TradeDecision.buy
    def SellAndHold(data: TradeBotDataFeed):
        return TradeDecision.sell
    def MeanReversion(data: TradeBotDataFeed):
        def _interpretRSI(rsi: float):
            if rsi <= 30:
                return TradeDecision.buy
            elif rsi >= 70:
                return TradeDecision.sell
            else:
                return TradeDecision.hold
        def _vote(data: TradeBotDataFeed):
            VOTER_COUNT = 20
            vote_map = {
                TradeDecision.buy: 0,
                TradeDecision.sell: 0,
                TradeDecision.hold: 0
            }
            for day in range(1, VOTER_COUNT+1):
                rsi_attr_name = f"rsi_{day}"
                rsi_value = getattr(data, rsi_attr_name)
                vote = _interpretRSI(rsi_value)
                vote_map[vote] += 1
            for decision, votes in vote_map.items():
                if votes > VOTER_COUNT // 2:
                    return decision
            return TradeDecision.hold
        return _vote(data)
    
    @staticmethod
    def getDecision(predicted_condition: int):
        match predicted_condition:
            case MarketCondition.uptrend: return DecisionFactory.BuyAndHold
            case MarketCondition.sideway: return DecisionFactory.MeanReversion
            case MarketCondition.downtrend: return DecisionFactory.SellAndHold
            case _: return DecisionFactory.BuyAndHold


class RouletteCell:
    id: int # Cell ID
    capital: float # Percentage of capital to allocate to each cell
    active: bool # Whether the cell has been active
    shares: float # Number of shares allocated to the cell
    decision_function: Callable # Function to determine whether to buy or sell

    def __init__(self, id: int, allocation: float):
        self.id = id
        self.capital = allocation
        self.active = False # Whether the cell has been active
        self.shares = 0 # Number of shares allocated to the cell 
        self.decision_function = None # Function to determine whether to buy or sell 
    def __repr__(self):
        return f"<RouletteCell(id={self.id}, capital={self.capital}, active={self.active}, shares={self.shares}, decision_function={self.decision_function})>"

class Roulette:
    cells: list[RouletteCell] # List of RouletteCell objects
    decision_factory: DecisionFactory

    def __init__(self, initial_capital: float, num_cells: int, decision_factory: DecisionFactory): # Initialize with the number of cells
        self.cells = [RouletteCell(id=idx, allocation=initial_capital/num_cells) for idx in range(num_cells)] # Create a list of RouletteCell objects with the specified number of cells
        self.decision_factory = decision_factory

class RouletteStrategyParam(BaseStrategyParam):
    roulette_size: int
    decision_factory: DecisionFactory
    def __init__(self, initial_capital: float, roulette_size: int, decision_factory: DecisionFactory):
        super().__init__(initial_capital)
        self.roulette_size = roulette_size
        self.decision_factory = decision_factory

class RouletteStrategy(BaseStrategy):
    strategy_name: str = "RouletteStrategy"
    trades: list[TradeLog] = []

    def __init__(self, datafeeder: DataFeeder, param: RouletteStrategyParam):
        super().__init__(datafeeder, param)
        self.roulette = Roulette(param.initial_capital, param.roulette_size, param.decision_factory)
        pass
    
    def _refreshRoulette(self, data: TradeBotDataFeed):
        for cell in self.roulette.cells:
            match cell.decision_function:
                case DecisionFactory.BuyAndHold:
                    if cell.active:
                        return
                    cell.active = True
                    cell.shares = cell.capital / data.close
                    self._handle_buy(data.ticker, data.report_date, data.close, cell.shares, f"Cell {cell.id} | BAH Refresh")

                    cell.capital -= data.close * cell.shares
                    pass
                case DecisionFactory.SellAndHold:
                    """
                        at 34, shares = 294.11, capital = 0
                        at 24, shares = 0, capital = 7058.82352942

                        at 34, shares = -294.11, capital = 20000
                        at 24, shares = 0, capital = 12941.36
                    """
                    if cell.active:
                        return
                    cell.active = True
                    cell.shares = -1 * cell.capital / data.close
                    self._handle_buy(data.ticker, data.report_date, data.close, cell.shares, f"Cell {cell.id} | SAH Refresh")

                    cell.capital -= data.close * cell.shares
                    pass
                case DecisionFactory.MeanReversion:
                    decision = cell.decision_function(data)
                    if cell.active:
                        if decision == TradeDecision.sell:
                            cell.active = False
                            self._handle_sell(data.ticker, data.report_date, data.close, cell.shares, f"Cell {cell.id} | MR Refresh Sell")

                            cell.capital += data.close * cell.shares
                            cell.shares = 0
                        return
                    if decision == TradeDecision.buy:
                        cell.active = True
                        cell.shares = cell.capital / data.close
                        self._handle_buy(data.ticker, data.report_date, data.close, cell.shares, f"Cell {cell.id} | MR Refresh Buy")

                        cell.capital -= data.close * cell.shares               
                case _:
                    pass
    
    def _cleanUpCell(self, data: TradeBotDataFeed, cell: RouletteCell):
        if cell.active:
            self._handle_sell(data.ticker, data.report_date, data.close, cell.shares, f"Cell {cell.id} | Lockup expired / End of Strategy")
            cell.active = False
            cell.capital += data.close * cell.shares
            cell.shares = 0
        return cell

    def _setStrategy(self, data: TradeBotDataFeed, cell: RouletteCell):
        cell.decision_function = self.param.decision_factory.getDecision(data.predicted_label)
        match cell.decision_function:
            case DecisionFactory.BuyAndHold:
                cell.active = True
                cell.shares = cell.capital / data.close
                self._handle_buy(data.ticker, data.report_date, data.close, cell.shares, f"Cell {cell.id} | BAH Set Strategy")

                cell.capital -= data.close * cell.shares
                pass
            case DecisionFactory.SellAndHold:
                cell.active = True
                cell.shares = -1 * cell.capital / data.close
                self._handle_buy(data.ticker, data.report_date, data.close, cell.shares, f"Cell {cell.id} | SAH Set Strategy")

                cell.capital -= data.close * cell.shares
                pass
            case DecisionFactory.MeanReversion:
                decision = cell.decision_function(data)
                if decision == TradeDecision.buy:
                    cell.active = True
                    cell.shares = cell.capital / data.close
                    self._handle_buy(data.ticker, data.report_date, data.close, cell.shares, f"Cell {cell.id} | MR Set Strategy")

                    cell.capital -= data.close * cell.shares
                    pass                
            case _:
                pass
        return cell
    
    def run(self, ticker: str, model: str, feature_set: str):
        datafeed = self.datafeeder.pullData(ticker=ticker, classifier_model=model, feature_set=feature_set)
        current_index = 0

        for daily_data in datafeed:
            self._refreshRoulette(daily_data)

            current_cell = self.roulette.cells[current_index]
            current_cell = self._cleanUpCell(daily_data, current_cell)
            current_cell = self._setStrategy(daily_data, current_cell)

            current_index = (current_index + 1) % self.param.roulette_size
        
        for cell in self.roulette.cells:
            cell = self._cleanUpCell(daily_data, cell)
    
    def dump_trade_logs(self) -> list[TradeLog]:
        return self.trades