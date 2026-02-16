"""
Forecasting Module
LSTM Neural Network and Prophet time-series forecasting for stock prices.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


# ─── LSTM Forecasting ───────────────────────────────────────────────────────

def prepare_lstm_data(data: pd.Series, lookback: int = 60, train_split: float = 0.8):
    """
    Prepare data for LSTM model.

    Parameters
    ----------
    data : pd.Series
        Close price series.
    lookback : int
        Number of past days to use for prediction.
    train_split : float
        Fraction of data for training.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, scaler, train_size)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * train_split)

    X_train = X[:train_size].reshape(-1, lookback, 1)
    y_train = y[:train_size]
    X_test = X[train_size:].reshape(-1, lookback, 1)
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test, scaler, train_size


def build_lstm_model(lookback: int = 60):
    """
    Build an LSTM neural network model.

    Architecture:
    - LSTM(50, return_sequences=True) → Dropout(0.2)
    - LSTM(50, return_sequences=True) → Dropout(0.2)
    - LSTM(50) → Dropout(0.2)
    - Dense(25) → Dense(1)
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model
    except ImportError:
        return None


def forecast_lstm(
    df: pd.DataFrame,
    column: str = "Close",
    lookback: int = 60,
    epochs: int = 20,
    batch_size: int = 32,
    forecast_days: int = 30,
) -> dict:
    """
    Run LSTM forecasting on stock data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    column : str
        Column to forecast.
    lookback : int
        Number of past days for input sequence.
    epochs : int
        Training epochs.
    batch_size : int
        Batch size for training.
    forecast_days : int
        Number of days to forecast into the future.

    Returns
    -------
    dict
        Contains 'train_pred', 'test_pred', 'future_pred', 'metrics', 'dates'.
    """
    data = df[column].dropna()

    X_train, y_train, X_test, y_test, scaler, train_size = prepare_lstm_data(
        data, lookback
    )

    model = build_lstm_model(lookback)
    if model is None:
        return {"error": "TensorFlow not installed. LSTM forecasting unavailable."}

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predictions
    train_pred = scaler.inverse_transform(model.predict(X_train, verbose=0))
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))

    # Actual values for comparison
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
    test_mae = mean_absolute_error(y_test_actual, test_pred)
    test_mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100

    # Future forecast
    last_sequence = scaler.transform(data.values[-lookback:].reshape(-1, 1))
    future_predictions = []

    current_sequence = last_sequence.reshape(1, lookback, 1)
    for _ in range(forecast_days):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]

    future_pred = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    # Generate future dates
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    return {
        "train_predictions": pd.Series(
            train_pred.flatten(),
            index=df.index[lookback : lookback + train_size],
        ),
        "test_predictions": pd.Series(
            test_pred.flatten(),
            index=df.index[lookback + train_size :],
        ),
        "future_predictions": pd.Series(future_pred.flatten(), index=future_dates),
        "metrics": {
            "RMSE": round(test_rmse, 2),
            "MAE": round(test_mae, 2),
            "MAPE": round(test_mape, 2),
        },
        "train_size": train_size,
    }


# ─── Prophet Forecasting ────────────────────────────────────────────────────

def forecast_prophet(
    df: pd.DataFrame,
    column: str = "Close",
    forecast_days: int = 30,
    changepoint_prior_scale: float = 0.05,
) -> dict:
    """
    Run Prophet forecasting on stock data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DateTimeIndex.
    column : str
        Column to forecast.
    forecast_days : int
        Number of days to forecast.
    changepoint_prior_scale : float
        Flexibility of the trend (higher = more flexible).

    Returns
    -------
    dict
        Contains 'forecast_df', 'components', 'metrics', 'model'.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {"error": "Prophet not installed. Prophet forecasting unavailable."}

    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = pd.DataFrame({
        "ds": df.index.tz_localize(None) if df.index.tz else df.index,
        "y": df[column].values,
    })

    # Train/test split
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:train_size]
    test_df = prophet_df[train_size:]

    # Build and fit model
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    model.fit(train_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=len(test_df) + forecast_days)
    forecast = model.predict(future)

    # Calculate metrics on test set
    test_forecast = forecast[forecast["ds"].isin(test_df["ds"])]
    if len(test_forecast) > 0 and len(test_df) > 0:
        merged = test_df.merge(
            test_forecast[["ds", "yhat"]], on="ds", how="inner"
        )
        if len(merged) > 0:
            test_rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
            test_mae = mean_absolute_error(merged["y"], merged["yhat"])
            test_mape = np.mean(np.abs((merged["y"] - merged["yhat"]) / merged["y"])) * 100
        else:
            test_rmse = test_mae = test_mape = float("nan")
    else:
        test_rmse = test_mae = test_mape = float("nan")

    return {
        "forecast": forecast,
        "model": model,
        "metrics": {
            "RMSE": round(test_rmse, 2),
            "MAE": round(test_mae, 2),
            "MAPE": round(test_mape, 2),
        },
        "train_size": train_size,
    }
