from app.models.TradeBotDataFeed import TradeBotDataFeed
from app.models.ClassifierResult import ClassifierResult
from app.models.MarketData import MarketData
from app.models.EquityIndicators import EquityIndicators

from sqlalchemy import select, label
from sqlalchemy.orm import aliased


class DataFeeder:
    def __init__(self, session):
        self.session=session

    def pullData(self, ticker: str, classifier_model: str, feature_set: str)->list[TradeBotDataFeed]:
        with self.session() as db:
            # Create an alias for the subquery
            classifier_subq = (
                select(
                    ClassifierResult.report_date.label('report_date'),
                    ClassifierResult.ticker.label('ticker'),
                    ClassifierResult.model.label('model'),
                    ClassifierResult.feature_set.label('feature_set'),
                    ClassifierResult.uptrend_prob.label('uptrend_prob'),
                    ClassifierResult.side_prob.label('side_prob'),
                    ClassifierResult.downtrend_prob.label('downtrend_prob'),
                    ClassifierResult.predicted_label.label('predicted_label')
                )
                .where(
                    (ClassifierResult.ticker == ticker) &
                    (ClassifierResult.model == classifier_model) &
                    (ClassifierResult.feature_set == feature_set)
                )
                .alias('classifier_subq')
            )

            # Perform the main query with a left join
            query = (
                select( 
                    classifier_subq.c.report_date,
                    classifier_subq.c.ticker,
                    classifier_subq.c.model,
                    classifier_subq.c.feature_set,
                    classifier_subq.c.uptrend_prob,
                    classifier_subq.c.side_prob,
                    classifier_subq.c.downtrend_prob,
                    classifier_subq.c.predicted_label,
                    MarketData.open,
                    MarketData.close,
                    EquityIndicators.rsi_1,
                    EquityIndicators.rsi_2,
                    EquityIndicators.rsi_3,
                    EquityIndicators.rsi_4,
                    EquityIndicators.rsi_5,
                    EquityIndicators.rsi_6,
                    EquityIndicators.rsi_7,
                    EquityIndicators.rsi_8,
                    EquityIndicators.rsi_9,
                    EquityIndicators.rsi_10,
                    EquityIndicators.rsi_11,
                    EquityIndicators.rsi_12,
                    EquityIndicators.rsi_13,
                    EquityIndicators.rsi_14,
                    EquityIndicators.rsi_15,
                    EquityIndicators.rsi_16,
                    EquityIndicators.rsi_17,
                    EquityIndicators.rsi_18,
                    EquityIndicators.rsi_19,
                    EquityIndicators.rsi_20,
                )
                .select_from(
                    classifier_subq.join(
                        MarketData,
                        (classifier_subq.c.report_date == MarketData.report_date)
                        & (classifier_subq.c.ticker == MarketData.ticker)
                    ).join(
                        EquityIndicators,
                        (classifier_subq.c.report_date == EquityIndicators.report_date)
                        & (classifier_subq.c.ticker == EquityIndicators.ticker)
                    )
                )
            ).order_by(MarketData.report_date)

            query_result = db.execute(query).all()
            return [TradeBotDataFeed(*result) for result in query_result]