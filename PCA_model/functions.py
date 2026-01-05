import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def predict_temperature(model, X, temp_scaler):
    """
    Predict temperature using trained regression model
    
    Args:
        model: Trained regression model
        X: Spectra to predict
        temp_scaler: Fitted StandardScaler
        w_len: Window length
        dw: Window step
        
    Returns:
        Predicted temperatures in original scale
    """
    # Segment spectra

    
    # Predict (scaled values)
    predictions_scaled = model.predict(X)
    temps_scaled = predictions_scaled.flatten()
    # Convert back to original scale
    predictions_original = temp_scaler.inverse_transform(
        temps_scaled.reshape(-1, 1)
    ).flatten()
    
    return predictions_original




def evaluate_regression_model(model, X_test, y_test, temp_scaler):
    """
    Evaluate regression model performance
    
    Args:
        model: Trained regression model
        X_test: Test spectra
        y_test: True temperature values
        temp_scaler: Fitted StandardScaler
        w_len: Window length
        dw: Window step
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = predict_temperature(model, X_test, temp_scaler)
    print(y_pred)
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Calculate prediction intervals (95%)
    residuals = y_test - y_pred
    std_residual = np.std(residuals)
    prediction_interval = 1.96 * std_residual
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Std_Residual': std_residual,
        'Prediction_Interval_95': prediction_interval,
        'Predictions': y_pred,
        'Residuals': residuals
    }
    
    # Plot results
    # plot_regression_results(y_test, y_pred, metrics)
    
    return metrics
