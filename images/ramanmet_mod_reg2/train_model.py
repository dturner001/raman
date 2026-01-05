"""
train the RamanNet regression model
"""

from data_processing import segment_spectrum_batch
from model import RamanNetRegression  # Our modified model
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
import keras
import joblib
def train_regression_model(X_train, y_train, X_val, y_val, w_len, dw, epochs, model_path, plot=True):
    """
    Train RamanNet for regression (temperature prediction)
    
    Args:
        X_train: Training spectra
        y_train: Training temperature values (continuous)
        X_val: Validation spectra  
        y_val: Validation temperature values
        w_len: Window length
        dw: Window step
        epochs: Training epochs
        model_path: Path to save model
        plot: Whether to plot training history
        
    Returns:
        Trained model and training history
    """
    
    # 1. Standardize temperature values
    print("Standardizing temperature values...")
    spectra_scaler = StandardScaler()
    X_train_scaled = spectra_scaler.fit_transform(X_train)
    X_val_scaled = spectra_scaler.transform(X_val)

    temp_scaler = StandardScaler()
    y_train_scaled = temp_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = temp_scaler.transform(y_val.reshape(-1, 1)).flatten()


    
    # Save scaler for later use
    scaler_path = model_path.replace('.h5', '_scaler.joblib').replace('.keras', '_scaler.joblib')
    joblib.dump(temp_scaler, scaler_path)

    joblib.dump(spectra_scaler, model_path.replace('.keras', '_spectra_scaler.joblib').replace('.h5', '_spectra_scaler.joblib'))

    # 2. Segment spectra into windows
    print("Segmenting spectra...")
    X_train_segmented = segment_spectrum_batch(X_train_scaled, w_len, dw)
    X_val_segmented = segment_spectrum_batch(X_val_scaled, w_len, dw)
    
    # 3. Create regression model
    print("Creating regression model...")
    mdl = RamanNetRegression(w_len=X_train_segmented[0].shape[1], 
                            n_windows=len(X_train_segmented))
    
    
    # 4. Define custom metrics for regression

    
    def mean_absolute_percentage_error(y_true, y_pred):
        """MAPE metric"""
        return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100
    
    # 5. Compile model with regression-appropriate settings
    # Option A: MSE loss (standard)
    # mdl.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for regression
    #     loss='mse',
    #     metrics=['mae', keras.metrics.R2Score()]
    # )
    
    # Option B: Huber loss (more robust to outliers)
    def huber_loss(y_true, y_pred, delta=4.0):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    mdl.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'temperature': tf.keras.losses.Huber(delta=1.0),

    },
    loss_weights={
        'temperature': 1.0,

    },
    metrics={
        'temperature': ['mae', 'mse'],

    }
    )
    
    # 6. Set up callbacks
    checkpoint = ModelCheckpoint(
        model_path, 
        verbose=1, 
        monitor='val_loss',
        save_best_only=True, 
        mode='min'
        
    )  
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,  # More aggressive reduction
        patience=5, 
        min_lr=1e-7, 
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # 7. Train the model
    print(f"Training regression model for {epochs} epochs...")
    print(f"Input shape: {[x.shape for x in X_train_segmented]}")
    print(f"Output shape: {y_train_scaled.shape}")
    
    training_history = mdl.fit(
    x=X_train_segmented,
    y=  y_train_scaled,
    batch_size=64,
    epochs=epochs,
    validation_data=(
        X_val_segmented,
        y_val_scaled,
    ),
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1
    )
    
    # 8. Plot training history
    if plot:
        plot_training_history(training_history, temp_scaler)
    
    return mdl, training_history


def plot_training_history(history, temp_scaler):
    """Plot training metrics for main (temperature) and aux (rpet) heads"""
    hist = history.history
    print("Available history keys:", hist.keys())
    # Extract series (use .get to avoid KeyErrors)
    reg_loss      = hist.get('loss', [])
    val_reg_loss  = hist.get('val_loss', [])
    reg_mae       = hist.get('mae', [])
    val_reg_mae   = hist.get('val_mae', [])
    reg_r2        = hist.get('r2_score', [])
    val_reg_r2    = hist.get('val_r2_score', [])


    lrs           = hist.get('lr', [])

    # Rescale temperature MAE back to original units
    reg_mae_orig     = np.array(reg_mae) * temp_scaler.scale_[0] if len(reg_mae) else []
    val_reg_mae_orig = np.array(val_reg_mae) * temp_scaler.scale_[0] if len(val_reg_mae) else []

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Temp loss
    axes[0, 0].plot(reg_loss, 'b-', label='Train temp loss')
    axes[0, 0].plot(val_reg_loss, 'r-', label='Val temp loss')
    axes[0, 0].set_title('Temperature loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Temp MAE (orig units)
    axes[0, 1].plot(reg_mae_orig, 'b-', label='Train temp MAE')
    axes[0, 1].plot(val_reg_mae_orig, 'r-', label='Val temp MAE')
    axes[0, 1].set_title('Temperature MAE (orig)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Temp R²
    if len(reg_r2):
        axes[0, 2].plot(reg_r2, 'b-', label='Train temp R²')
        axes[0, 2].plot(val_reg_r2, 'r-', label='Val temp R²')
        axes[0, 2].set_title('Temperature R²'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].axis('off')


    # Learning rate
    if len(lrs):
        axes[1, 2].plot(lrs, 'g-')
        axes[1, 2].set_title('Learning rate'); axes[1, 2].set_yscale('log'); axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


def predict_temperature(model, X, temp_scaler, w_len, dw):
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

    
    X_segmented = segment_spectrum_batch(X, w_len, dw)
    
    # Predict (scaled values)
    predictions_scaled = model.predict(X_segmented)
    temps_scaled = predictions_scaled['temperature'].flatten()
    # Convert back to original scale
    predictions_original = temp_scaler.inverse_transform(
        temps_scaled.reshape(-1, 1)
    ).flatten()
    
    return predictions_original


def evaluate_regression_model(model, X_test, y_test, temp_scaler,spectra_scaler, w_len, dw):
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
    X_test = spectra_scaler.fit_transform(X_test)
    y_pred = predict_temperature(model, X_test, temp_scaler, w_len, dw)
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
    plot_regression_results(y_test, y_pred, metrics)
    
    return metrics


def plot_regression_results(y_true, y_pred, metrics):
    """Plot regression evaluation results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Predicted vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                   'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Temperature')
    axes[0, 0].set_ylabel('Predicted Temperature')
    axes[0, 0].set_title(f'Predicted vs Actual (R² = {metrics["R2"]:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = metrics['Residuals']
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Temperature')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'Residuals (MAE = {metrics["MAE"]:.2f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(x=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Residual Distribution (σ = {metrics["Std_Residual"]:.2f})')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1, 0].hist(np.abs(residuals), bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=metrics['MAE'], color='r', linestyle='--', 
                      label=f'MAE = {metrics["MAE"]:.2f}')
    axes[1, 0].set_xlabel('Absolute Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Absolute Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Metrics summary
    axes[1, 2].axis('off')
    metrics_text = (
        f'Regression Metrics:\n\n'
        f'MAE: {metrics["MAE"]:.2f}\n'
        f'RMSE: {metrics["RMSE"]:.2f}\n'
        f'R²: {metrics["R2"]:.3f}\n'
        f'MAPE: {metrics["MAPE"]:.2f}%\n'
        f'95% Prediction Interval: ±{metrics["Prediction_Interval_95"]:.2f}'
    )
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, 
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()