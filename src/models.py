from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor

class BaselineModels:
    @staticmethod
    def train_arima(train_data, order=(0, 1, 0)):
        """Fits a classical statistical baseline model."""
        model = ARIMA(train_data, order=order)
        return model.fit()

    @staticmethod
    def train_nn(X_train, y_train):
        """Fits the high-performing deep multi-layer neural network."""
        nn_model = MLPRegressor(
            hidden_layer_sizes=(50, 25, 10), 
            activation='relu', 
            solver='adam', 
            max_iter=500,
            random_state=42
        )
        nn_model.fit(X_train, y_train)
        return nn_model
    