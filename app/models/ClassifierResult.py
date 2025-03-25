from sqlalchemy import Column, Integer, String, Date, Float
from sqlalchemy.ext.declarative import declarative_base

from datetime import date

Base = declarative_base()

class ClassifierResult(Base):
    __tablename__ = 'classifier_result'
    __table_args__ = {'schema': 'fyp'}

    report_date = Column(Date, nullable=False)
    
    ### Primary keys
    ticker = Column(String, primary_key=True)
    model = Column(String, primary_key=True)
    feature_set = Column(String, primary_key=True)
    
    uptrend_prob = Column(Float(4))
    side_prob = Column(Float(4))
    downtrend_prob = Column(Float(4))
    predicted_label = Column(Integer, nullable=False)
    actual_label = Column(Integer, nullable=False)

    def __init__(self, 
             report_date: date, 
             ticker: str, 
             model: str, 
             feature_set: str, 
             uptrend_prob: float, 
             side_prob: float, 
             downtrend_prob: float, 
             predicted_label: int, 
             actual_label: int):
        self.report_date = report_date
        self.ticker = ticker
        self.model = model
        self.feature_set = feature_set
        self.uptrend_prob = uptrend_prob
        self.side_prob = side_prob
        self.downtrend_prob = downtrend_prob
        self.predicted_label = predicted_label
        self.actual_label = actual_label

    def __repr__(self):
            return f"<ClassifierResult(date={self.report_date}, ticker={self.ticker}, model={self.model}, feature_set={self.feature_set})>"