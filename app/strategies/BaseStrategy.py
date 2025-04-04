from app.datafeed.DataFeeder import DataFeeder
from app.models.TradeLog import TradeLog

from datetime import date

class BaseStrategyParam:
    initial_capital: float
    def __init__(self, initial_capital: float):
        self.initial_capital = 10000
        pass

class BaseStrategy:
    strategy_name: str
    trades: list[TradeLog] = []

    def _handle_buy(self, ticker: str, date: date, price: float, shares: float, reason: str=""):
        self.trades.append(
            TradeLog(
                report_date=date,
                ticker=ticker,
                strategy=self.strategy_name,
                action='BUY',
                price=price,
                shares=shares,
                note=reason
            )
        )
    
    def _handle_sell(self, ticker: str, date: date, price: float, shares: float, reason: str):
        self.trades.append(
            TradeLog(
                report_date=date,
                ticker=ticker,
                strategy=self.strategy_name,
                action='SELL',
                price=price,
                shares=shares,
                note=reason
            )
        )
    
    def __init__(self, datafeeder: DataFeeder, param: BaseStrategyParam):
        self.datafeeder = datafeeder
        self.param = param

    def run(self):
        pass

    def reset(self):
        self.trades = []
        self.__init__(self.datafeeder, self.param)

    def dump_trade_logs(self)->list[TradeLog]:
        pass