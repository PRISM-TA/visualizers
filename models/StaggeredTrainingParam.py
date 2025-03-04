class StaggeredTrainingParam:
    training_day_count: str # number of days to train on
    prediction_day_count: str # number of days to predict for
    ticker_list: list[str] # list of tickers to train on
    model_param: dict

    def __init__(self, 
                 training_day_count: str, 
                 prediction_day_count: str, 
                 ticker: str
                 ):
        self.training_day_count = training_day_count
        self.prediction_day_count = prediction_day_count
        self.ticker = ticker
    
    def __repr__(self):
        return f"<StaggeredTrainingParam(training_day_count={self.training_day_count}, prediction_day_count={self.prediction_day_count}, ticker_list={self.ticker_list})>"