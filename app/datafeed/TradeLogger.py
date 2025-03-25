from app.models.TradeLog import TradeLog
from sqlalchemy import select


class TradeLogger:
    def __init__(self, session):
        self.session = session

    def getTradeLogs(self, ticker: str, strategy: str) -> list[TradeLog]:
        with self.session() as db:
            # Perform the main query with a left join
            query = (
                select(TradeLog)
                .where(
                    TradeLog.ticker == ticker,
                    TradeLog.strategy == strategy
                )
                .order_by(TradeLog.report_date)
            )

            query_result = db.execute(query).scalars().all()  # Use scalars() to get TradeLog objects directly
            
            return query_result