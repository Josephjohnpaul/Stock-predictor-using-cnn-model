# =============================================================
#  Pattern Recognition for Financial Time Series Forecasting
#  Complete Pipeline — Single File
# =============================================================
#
#  PIPELINE:
#  Raw Stock Data → Normalize → STFT Spectrogram → CNN → Predict
#
#  HOW TO RUN:
#  1. Install dependencies:
#     pip install yfinance numpy pandas matplotlib scipy tensorflow scikit-learn
#  2. Run this file:
#     python stock_prediction.py
#  3. All plots are saved automatically in the same folder
# =============================================================


# ── IMPORTS ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models


# ── CONFIGURATION ─────────────────────────────────────────────
# Change these values if you want different stocks or date range

TICKERS    = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]  # NSE stocks
START_DATE = "2018-01-01"
END_DATE   = "2023-12-31"
WINDOW_SIZE = 64    # Number of days per training sample
NPERSEG     = 32    # STFT window length (frequency resolution)
NOVERLAP    = 24    # Overlap between STFT windows
EPOCHS      = 30    # Max training epochs
BATCH_SIZE  = 32    # Samples per training step
FORECAST_DAYS = 10  # How many days ahead to predict


# =============================================================
#  CHAPTER 1 — DATA PREPARATION
# =============================================================

def load_and_prepare_data(tickers, start, end):
    """
    Downloads stock data + macroeconomic signals,
    combines into X(t) multivariate signal, normalizes it.

    Returns:
        df_scaled  — normalized DataFrame (all signals)
        df_raw     — original prices in rupees (for inverse transform)
        scaler     — fitted MinMaxScaler (to reverse normalization later)
    """

    print("\n" + "="*50)
    print(" CHAPTER 1: Data Preparation")
    print("="*50)

    # ── Download stock prices ──────────────────────────────────
    print("\nDownloading stock prices...")
    stocks = yf.download(tickers, start=start, end=end)["Close"]
    # Returns a DataFrame: rows=dates, columns=ticker symbols

    # ── Download Sensex (market index) ────────────────────────
    print("Downloading Sensex...")
    sensex = yf.download("^BSESN", start=start, end=end)["Close"]
    # ^BSESN = BSE Sensex ticker on Yahoo Finance

    # ── Download USD-INR exchange rate ────────────────────────
    print("Downloading USD-INR rate...")
    usdinr = yf.download("INR=X", start=start, end=end)["Close"]
    # INR=X = USD to Indian Rupee exchange rate

    # ── Build X(t) = [p(t), p2(t), p3(t), s(t), d(t)] ────────
    # This is the multivariate signal defined in the assignment
    df_raw = pd.DataFrame({
        "RELIANCE": stocks["RELIANCE.NS"].squeeze(),
        "TCS":      stocks["TCS.NS"].squeeze(),
        "INFY":     stocks["INFY.NS"].squeeze(),
        "SENSEX":   sensex.squeeze(),
        "USDINR":   usdinr.squeeze(),
    }).dropna()
    # dropna() removes any day where ANY signal has missing data
    # (different markets have different holidays)

    print(f"\nX(t) shape: {df_raw.shape}")
    print(f"Signals: {list(df_raw.columns)}")
    print(f"Date range: {df_raw.index[0].date()} to {df_raw.index[-1].date()}")

    # ── Normalize all signals to [0, 1] ───────────────────────
    # Formula: scaled = (value - min) / (max - min)
    # Prevents large-valued signals from dominating the model
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_raw)

    df_scaled = pd.DataFrame(
        scaled_values,
        columns=df_raw.columns,
        index=df_raw.index
    )

    # ── Plot: Time Series (Required Figure 1) ─────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Stock prices
    df_scaled[["RELIANCE", "TCS", "INFY"]].plot(
        ax=axes[0],
        title="Normalized Stock Prices — RELIANCE, TCS, INFY"
    )
    axes[0].set_ylabel("Normalized Price (0 to 1)")
    axes[0].set_xlabel("Date")

    # Macro signals
    df_scaled[["SENSEX", "USDINR"]].plot(
        ax=axes[1],
        title="Normalized Macro Signals — Sensex & USD-INR"
    )
    axes[1].set_ylabel("Normalized Value (0 to 1)")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig("figure1_time_series.png", dpi=150)
    plt.show()
    print("\nSaved: figure1_time_series.png")

    return df_scaled, df_raw, scaler


# =============================================================
#  CHAPTER 2 — SIGNAL PROCESSING
# =============================================================

def compute_spectrogram(signal, nperseg=32, noverlap=24):
    """
    Converts a 1D price signal into a 2D spectrogram image.

    How it works:
    1. Slides a window of length nperseg across the signal
    2. Applies Fourier Transform to each window segment
    3. Computes energy = |STFT|^2 for each time-frequency point
    4. Returns a 2D grid: rows=frequencies, cols=time windows

    Args:
        signal  : 1D numpy array of prices
        nperseg : window length L (frequency resolution)
        noverlap: overlap between windows (L - hop_size)

    Returns:
        f : frequency axis values
        t : time axis values
        S : 2D spectrogram array S(t,f) = |STFT|^2
    """
    f, t, Zxx = stft(
        signal,
        fs=1,           # 1 sample per day
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann'   # smooth window to avoid edge artifacts
    )
    S = np.abs(Zxx) ** 2  # Power spectrogram: S(t,f) = |STFT|^2
    return f, t, S


def plot_signal_analysis(df_scaled, nperseg=32, noverlap=24):
    """
    Generates required figures:
    - Figure 2: Frequency Spectrum (plain FFT)
    - Figure 3: Spectrogram (STFT)
    """

    print("\n" + "="*50)
    print(" CHAPTER 2: Signal Processing")
    print("="*50)

    signal = df_scaled["RELIANCE"].values

    # ── Plot: Frequency Spectrum (Required Figure 2) ──────────
    print("\nComputing frequency spectrum...")

    fft_values    = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_values)
    freqs         = np.fft.rfftfreq(len(signal))
    # rfftfreq returns frequency values in cycles/day

    plt.figure(figsize=(12, 4))
    plt.plot(freqs, fft_magnitude, color='steelblue')
    plt.title("Frequency Spectrum — RELIANCE.NS")
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Amplitude (signal strength)")
    plt.tight_layout()
    plt.savefig("figure2_frequency_spectrum.png", dpi=150)
    plt.show()
    print("Saved: figure2_frequency_spectrum.png")

    # ── Plot: Spectrogram (Required Figure 3) ─────────────────
    print("\nComputing spectrogram...")

    f, t, S = compute_spectrogram(signal, nperseg=nperseg, noverlap=noverlap)

    plt.figure(figsize=(14, 5))
    plt.pcolormesh(
        t, f,
        10 * np.log10(S + 1e-10),  # convert to dB scale
        shading='gouraud',           # smooth color blending
        cmap='inferno'               # dark=low energy, bright=high
    )
    plt.colorbar(label='Energy (dB)')
    plt.title("Spectrogram — RELIANCE.NS (STFT)")
    plt.xlabel("Time (trading days)")
    plt.ylabel("Frequency (cycles/day)")
    plt.tight_layout()
    plt.savefig("figure3_spectrogram.png", dpi=150)
    plt.show()
    print("Saved: figure3_spectrogram.png")

    print("\nSpectrogram shape:", S.shape)
    print(f"  → {S.shape[0]} frequency bins × {S.shape[1]} time windows")


# =============================================================
#  CHAPTER 3 — MODEL DEVELOPMENT
# =============================================================

def create_dataset(df_scaled, window_size=64, nperseg=32, noverlap=24):
    """
    Builds the training dataset from the multivariate signal X(t).

    For each sliding window of `window_size` days:
    - Computes spectrogram for EACH signal component separately
    - Stacks them as channels → one multi-channel image
    - Label = next day's RELIANCE price

    Returns:
        X : array of shape (samples, freq_bins, time_steps, n_signals)
        y : array of shape (samples,)
    """

    print("\n" + "="*50)
    print(" CHAPTER 3: Building Dataset")
    print("="*50)

    values = df_scaled.values
    # Shape: (num_days, num_signals)
    # Each row = one day's X(t) = [RELIANCE, TCS, INFY, SENSEX, USDINR]

    X, y = [], []

    total = len(values) - window_size - 1
    print(f"\nCreating {total} training samples...")

    for i in range(total):
        window = values[i : i + window_size]
        # Shape: (64, 5) — 64 days of all 5 signals

        channels = []
        for col in range(window.shape[1]):
            # Compute spectrogram for each signal independently
            _, _, S = compute_spectrogram(
                window[:, col],
                nperseg=nperseg,
                noverlap=noverlap
            )
            channels.append(S)
            # S shape: (freq_bins, time_steps)

        # Stack all signal spectrograms into one image
        # Like stacking R,G,B channels but with 5 signals
        Xt_spec = np.stack(channels, axis=-1)
        # Shape: (freq_bins, time_steps, 5)

        X.append(Xt_spec)
        y.append(values[i + window_size, 0])
        # Label = next day's RELIANCE price (column 0)

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset ready:")
    print(f"  X shape: {X.shape}  (samples, freq_bins, time_steps, signals)")
    print(f"  y shape: {y.shape}  (samples,)")

    return X, y


def build_cnn(input_shape):
    """
    Builds a CNN model for spectrogram-based price regression.

    Architecture:
    - 3 Convolutional blocks (find patterns at different scales)
    - GlobalAveragePooling (summarize patterns)
    - Dense layers (combine patterns → make prediction)
    - Single output neuron (predicted price)

    Args:
        input_shape: shape of one input image e.g. (17, 7, 5)

    Returns:
        compiled Keras model
    """

    model = models.Sequential([

        # ── Block 1: Basic pattern detection ──────────────────
        layers.Conv2D(32, (3,3), activation='relu',
                      padding='same', input_shape=input_shape),
        # 32 filters, each looking at 3×3 patches
        # relu: keep positive values, set negatives to 0
        layers.BatchNormalization(),
        # Normalize outputs for stable training
        layers.MaxPooling2D((2,2)),
        # Shrink image by half, keep strongest patterns

        # ── Block 2: Intermediate pattern detection ────────────
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        # 64 filters — finds more complex combinations
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # ── Block 3: High-level pattern detection ─────────────
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        # 128 filters — finds high-level patterns
        layers.GlobalAveragePooling2D(),
        # Average across entire remaining image → 1D vector of 128

        # ── Decision layers ───────────────────────────────────
        layers.Dense(64, activation='relu'),
        # Combine all 128 pattern scores into 64 decisions
        layers.Dropout(0.3),
        # Randomly disable 30% of neurons → prevents memorization
        layers.Dense(1)
        # Single output = predicted next price (no activation)
    ])

    model.compile(
        optimizer='adam',   # smart adaptive learning rate
        loss='mse',         # minimize mean squared error
        metrics=['mae']     # also track mean absolute error
    )

    return model


def train_model(X_train, y_train, input_shape, epochs=30, batch_size=32):
    """
    Builds and trains the CNN model.

    Returns:
        model   : trained CNN
        history : training history (loss per epoch)
    """

    print("\n" + "="*50)
    print(" CHAPTER 3: Training CNN Model")
    print("="*50)

    model = build_cnn(input_shape)
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        # Use 10% of training data to monitor overfitting
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                # Stop if no improvement for 5 epochs
                restore_best_weights=True
                # Revert to best weights after stopping
            )
        ]
    )

    # ── Plot: Training Progress ────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'],     label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    plt.show()
    print("Saved: training_progress.png")

    # ── Save CNN Architecture Diagram (Required Figure 4) ─────
    try:
        tf.keras.utils.plot_model(
            model,
            to_file='figure4_cnn_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            dpi=96
        )
        print("Saved: figure4_cnn_architecture.png")
    except Exception:
        print("Note: Install pydot & graphviz to save architecture diagram")

    return model, history


# =============================================================
#  CHAPTER 4 — EVALUATION & ANALYSIS
# =============================================================

def evaluate_model(model, X_test, y_test, scaler, df_raw):
    """
    Makes predictions, converts back to rupees,
    plots actual vs predicted, computes error metrics.
    """

    print("\n" + "="*50)
    print(" CHAPTER 4: Evaluation")
    print("="*50)

    # ── Make predictions ──────────────────────────────────────
    y_pred = model.predict(X_test).flatten()

    # ── Convert normalized values back to rupees ──────────────
    def to_rupees(normalized_values):
        dummy = np.zeros((len(normalized_values), scaler.n_features_in_))
        dummy[:, 0] = normalized_values
        # Column 0 = RELIANCE
        return scaler.inverse_transform(dummy)[:, 0]

    y_test_real = to_rupees(y_test)
    y_pred_real = to_rupees(y_pred)

    # ── Compute error metrics ─────────────────────────────────
    mse_norm = mean_squared_error(y_test, y_pred)
    mse_real = mean_squared_error(y_test_real, y_pred_real)
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mse_real)

    print(f"\nResults:")
    print(f"  MSE  (normalized): {mse_norm:.6f}")
    print(f"  MSE  (₹):          {mse_real:.2f}")
    print(f"  RMSE (₹):          {rmse_real:.2f}")
    print(f"  MAE  (₹):          {mae_real:.2f}")
    print(f"\n  → On average, predictions are off by ₹{mae_real:.2f}")

    # ── Plot: Actual vs Predicted ──────────────────────────────
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_real, color='blue',
             label='Actual Price', linewidth=1.5)
    plt.plot(y_pred_real, color='red', linestyle='--',
             label='Predicted Price', linewidth=1.5)
    plt.title("Actual vs Predicted — RELIANCE.NS")
    plt.xlabel("Trading Days (test period)")
    plt.ylabel("Stock Price (₹)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_plot.png", dpi=150)
    plt.show()
    print("Saved: prediction_plot.png")

    return mse_norm, y_pred


def forecast_future(model, df_scaled, scaler,
                    window_size=64, nperseg=32,
                    noverlap=24, n_days=10):
    """
    Predicts the next n_days prices using recursive forecasting.

    How it works:
    - Start with the last `window_size` days of real data
    - Predict day+1, add it to the window
    - Predict day+2 using updated window
    - Repeat n_days times

    Note: accuracy decreases as n_days increases
    because errors compound at each step.
    """

    print("\n" + "="*50)
    print(f" CHAPTER 4: Forecasting Next {n_days} Days")
    print("="*50)

    values = df_scaled.values
    # Use the last window_size days as starting point
    current_window = list(values[-window_size:])
    # Shape: (64, 5) — last 64 days of all 5 signals

    predictions = []

    for day in range(n_days):
        # Build spectrogram from current window
        window = np.array(current_window[-window_size:])
        channels = []
        for col in range(window.shape[1]):
            _, _, S = compute_spectrogram(
                window[:, col],
                nperseg=nperseg,
                noverlap=noverlap
            )
            channels.append(S)
        Xt_spec = np.stack(channels, axis=-1)

        # Predict next price
        S_input = Xt_spec[np.newaxis, ...]
        # Add batch dimension: (1, freq, time, 5)
        next_price = model.predict(S_input, verbose=0)[0][0]

        predictions.append(next_price)

        # Add prediction to window for next iteration
        # Use same macro values as last known day (simplified)
        new_row = current_window[-1].copy()
        new_row[0] = next_price  # Update only RELIANCE price
        current_window.append(new_row)

    # Convert to rupees
    def to_rupees(normalized_values):
        dummy = np.zeros((len(normalized_values), scaler.n_features_in_))
        dummy[:, 0] = normalized_values
        return scaler.inverse_transform(dummy)[:, 0]

    future_real = to_rupees(np.array(predictions))

    # Print forecast table
    print(f"\n📈 RELIANCE.NS Price Forecast:")
    print("-" * 30)
    for i, price in enumerate(future_real):
        print(f"  Day +{i+1:2d}: ₹{price:,.2f}")

    # Plot forecast
    last_60_real = to_rupees(values[-60:, 0])

    plt.figure(figsize=(12, 5))
    plt.plot(range(60), last_60_real,
             color='blue', linewidth=2,
             label='Last 60 days (actual)')
    plt.plot(range(59, 59 + len(future_real) + 1),
             np.append(last_60_real[-1], future_real),
             color='red', linestyle='--',
             linewidth=2, marker='o', markersize=5,
             label=f'Forecast (next {n_days} days)')
    plt.axvline(x=59, color='gray', linestyle=':',
                linewidth=1.5, label='Prediction starts')
    plt.title(f"RELIANCE.NS — {n_days}-Day Price Forecast")
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price (₹)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("future_forecast.png", dpi=150)
    plt.show()
    print("Saved: future_forecast.png")

    return future_real


def compare_approaches(df_scaled, scaler, window_size=64,
                       nperseg=32, noverlap=24,
                       epochs=30, batch_size=32):
    """
    Compares two approaches:
    1. Single stock only (RELIANCE price)
    2. Full X(t) multivariate signal (all 5 signals)

    This directly answers the assignment's Task 4:
    'Analyze effect of different features'
    """

    print("\n" + "="*50)
    print(" CHAPTER 4: Feature Comparison")
    print("="*50)

    results = {}

    # ── Approach 1: Single stock ───────────────────────────────
    print("\nTraining: Single stock (RELIANCE only)...")
    signal_only = df_scaled[["RELIANCE"]].values

    X1, y1 = [], []
    for i in range(len(signal_only) - window_size - 1):
        seg = signal_only[i:i+window_size, 0]
        _, _, S = compute_spectrogram(seg, nperseg=nperseg, noverlap=noverlap)
        X1.append(S[..., np.newaxis])
        y1.append(signal_only[i+window_size, 0])
    X1, y1 = np.array(X1), np.array(y1)

    split1 = int(0.8 * len(X1))
    m1 = build_cnn(X1.shape[1:])
    m1.fit(X1[:split1], y1[:split1], epochs=epochs,
           batch_size=batch_size, validation_split=0.1,
           callbacks=[tf.keras.callbacks.EarlyStopping(
               patience=5, restore_best_weights=True)],
           verbose=0)
    p1 = m1.predict(X1[split1:], verbose=0).flatten()
    results["Single Stock\n(RELIANCE only)"] = \
        mean_squared_error(y1[split1:], p1)

    # ── Approach 2: Full X(t) ─────────────────────────────────
    print("Training: Full X(t) — all 5 signals...")
    X2, y2 = [], []
    values = df_scaled.values
    for i in range(len(values) - window_size - 1):
        window = values[i:i+window_size]
        channels = []
        for col in range(window.shape[1]):
            _, _, S = compute_spectrogram(
                window[:, col], nperseg=nperseg, noverlap=noverlap)
            channels.append(S)
        X2.append(np.stack(channels, axis=-1))
        y2.append(values[i+window_size, 0])
    X2, y2 = np.array(X2), np.array(y2)

    split2 = int(0.8 * len(X2))
    m2 = build_cnn(X2.shape[1:])
    m2.fit(X2[:split2], y2[:split2], epochs=epochs,
           batch_size=batch_size, validation_split=0.1,
           callbacks=[tf.keras.callbacks.EarlyStopping(
               patience=5, restore_best_weights=True)],
           verbose=0)
    p2 = m2.predict(X2[split2:], verbose=0).flatten()
    results["Full X(t)\n(All 5 signals)"] = \
        mean_squared_error(y2[split2:], p2)

    # ── Plot comparison ────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results.keys(), results.values(),
                   color=['steelblue', 'green'], width=0.4)
    for bar, val in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.00005,
                 f'{val:.5f}', ha='center', fontsize=11)
    plt.title("MSE Comparison: Feature Analysis")
    plt.ylabel("MSE — Normalized (lower is better)")
    plt.tight_layout()
    plt.savefig("feature_comparison.png", dpi=150)
    plt.show()
    print("Saved: feature_comparison.png")

    print("\nResults:")
    for name, mse in results.items():
        print(f"  {name.replace(chr(10), ' ')}: MSE = {mse:.6f}")

    return results


# =============================================================
#  MAIN — Run everything in order
# =============================================================

if __name__ == "__main__":

    print("\n" + "="*50)
    print("  Stock Price Prediction — Full Pipeline")
    print("="*50)

    # ── Chapter 1: Data ────────────────────────────────────────
    df_scaled, df_raw, scaler = load_and_prepare_data(
        TICKERS, START_DATE, END_DATE
    )

    # ── Chapter 2: Signal Analysis ────────────────────────────
    plot_signal_analysis(df_scaled, NPERSEG, NOVERLAP)

    # ── Chapter 3: Build Dataset & Train ─────────────────────
    X, y = create_dataset(df_scaled, WINDOW_SIZE, NPERSEG, NOVERLAP)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model, history = train_model(
        X_train, y_train,
        input_shape=X_train.shape[1:],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # ── Chapter 4: Evaluate ───────────────────────────────────
    mse, y_pred = evaluate_model(model, X_test, y_test, scaler, df_raw)

    # ── Chapter 4: Forecast future prices ────────────────────
    future_prices = forecast_future(
        model, df_scaled, scaler,
        window_size=WINDOW_SIZE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        n_days=FORECAST_DAYS
    )

    # ── Chapter 4: Feature comparison ─────────────────────────
    comparison = compare_approaches(
        df_scaled, scaler,
        window_size=WINDOW_SIZE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "="*50)
    print("  ALL DONE — Files Saved:")
    print("="*50)
    print("  figure1_time_series.png       ← Required Figure 1")
    print("  figure2_frequency_spectrum.png ← Required Figure 2")
    print("  figure3_spectrogram.png        ← Required Figure 3")
    print("  figure4_cnn_architecture.png   ← Required Figure 4")
    print("  training_progress.png")
    print("  prediction_plot.png")
    print("  future_forecast.png")
    print("  feature_comparison.png")
    print("="*50)
