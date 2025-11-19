class SequenceBuilder:
    def __init__(self, df):
        self.df = df

    def add_lags(self, columns, lags=24):
        for col in columns:
            for lag in range(1, lags+1):
                self.df[f"{col}_lag_{lag}"] = self.df[col].shift(lag)
        return self.df