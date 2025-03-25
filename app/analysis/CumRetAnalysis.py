from app.datafeed.DataFeeder import DataFeeder
from app.datafeed.TradeLogger import TradeLogger
from app.models.TradeLog import TradeLog
import pandas as pd
import numpy as np
from datetime import date
from dataclasses import dataclass

@dataclass
class CumRetDatum:
    report_date: date
    cash: float
    shares: float
    cumulative_return: float
    cumulative_return_change: float

@dataclass
class CumRetAnalysisParam:
    benchmark_strategy: str
    benchmark_initial_capital: float
    target_strategy: str
    target_initial_capital: float

    classifier_model: str
    feature_set: str
    ticker: str

class CumRetAnalysis:
    feeder: DataFeeder
    trade_logger: TradeLogger
    param: CumRetAnalysisParam

    def __init__(self, feeder:DataFeeder, trade_logger: TradeLogger):
        self.feeder = feeder
        self.trade_logger = trade_logger

    def setParam(self, param: CumRetAnalysisParam):
        self.param = param

    def run(self):
        datafeed = self.feeder.pullData(ticker=self.param.ticker, 
                                    classifier_model=self.param.classifier_model, 
                                    feature_set=self.param.feature_set)
        benchmark_tradelog = self.trade_logger.getTradeLogs(ticker=self.param.ticker, 
                                                        strategy=self.param.benchmark_strategy)
        target_tradelog = self.trade_logger.getTradeLogs(ticker=self.param.ticker, 
                                                        strategy=self.param.target_strategy)
        
        # Initialize data structures for both strategies
        benchmark_data = []
        target_data = []
        
        # Initialize starting positions
        benchmark_position = {'cash': self.param.benchmark_initial_capital, 'shares': 0.0}
        target_position = {'cash': self.param.target_initial_capital, 'shares': 0.0}
        
        # Initialize previous returns
        prev_benchmark_return = 0.0
        prev_target_return = 0.0

        for daily_data in datafeed:
            today_date = daily_data.report_date
            today_price = daily_data.close

            # Process benchmark strategy
            # Step 1-3: Calculate asset value, cumulative return and return change
            benchmark_asset_value = (benchmark_position['cash'] + 
                                benchmark_position['shares'] * today_price)
            benchmark_cum_return = ((benchmark_asset_value - 
                                self.param.benchmark_initial_capital) / 
                                self.param.benchmark_initial_capital)
            benchmark_return_change = benchmark_cum_return - prev_benchmark_return
            prev_benchmark_return = benchmark_cum_return

            # Step 4-5: Process trades and adjust positions
            benchmark_trades = [trade for trade in benchmark_tradelog 
                            if trade.report_date == today_date]
            for trade in benchmark_trades:
                if trade.action == 'BUY':
                    trade_value = trade.price * trade.shares
                    benchmark_position['cash'] -= trade_value
                    benchmark_position['shares'] += trade.shares
                elif trade.action == 'SELL':
                    trade_value = trade.price * trade.shares
                    benchmark_position['cash'] += trade_value
                    benchmark_position['shares'] -= trade.shares

            # Store benchmark data
            benchmark_data.append(CumRetDatum(
                report_date=today_date,
                cash=benchmark_position['cash'],
                shares=benchmark_position['shares'],
                cumulative_return=benchmark_cum_return,
                cumulative_return_change=benchmark_return_change
            ))

            # Process target strategy (similar to benchmark)
            target_asset_value = target_position['cash'] + target_position['shares'] * today_price
            target_cum_return = ((target_asset_value - 
                                self.param.target_initial_capital) / 
                                self.param.target_initial_capital)
            target_return_change = target_cum_return - prev_target_return
            prev_target_return = target_cum_return

            target_trades = [trade for trade in target_tradelog 
                            if trade.report_date == today_date]
            for trade in target_trades:
                if trade.action == 'BUY':
                    trade_value = trade.price * trade.shares
                    target_position['cash'] -= trade_value
                    target_position['shares'] += trade.shares
                elif trade.action == 'SELL':
                    trade_value = trade.price * trade.shares
                    target_position['cash'] += trade_value
                    target_position['shares'] -= trade.shares

            target_data.append(CumRetDatum(
                report_date=today_date,
                cash=target_position['cash'],
                shares=target_position['shares'],
                cumulative_return=target_cum_return,
                cumulative_return_change=target_return_change
            ))

        # Convert results to DataFrame
        benchmark_df = pd.DataFrame([vars(d) for d in benchmark_data])
        target_df = pd.DataFrame([vars(d) for d in target_data])

        # Combine the results
        result_df = pd.DataFrame({
            'date': benchmark_df['report_date'],
            'benchmark_cum_return_pct': benchmark_df['cumulative_return'] * 100,
            'benchmark_return_change_pct': benchmark_df['cumulative_return_change'] * 100,
            'benchmark_cash': benchmark_df['cash'],
            'benchmark_shares': benchmark_df['shares'],
            'target_cum_return_pct': target_df['cumulative_return'] * 100,
            'target_return_change_pct': target_df['cumulative_return_change'] * 100,
            'target_cash': target_df['cash'],
            'target_shares': target_df['shares'],
            'target_performance_pct': (target_df['cumulative_return_change'] - benchmark_df['cumulative_return_change']) * 100
        })

        return result_df