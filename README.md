# Stock Price Prediction using LSTM

## Overview
This project utilizes a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical stock data. The model is trained using stock market data, processes multiple features (Open, High, Low, Volume), and predicts future closing prices. The model is implemented using TensorFlow/Keras and utilizes data preprocessing, normalization, and sequence modeling.

## Features
- Preprocesses stock data (handling missing values, data formatting, and normalization).
- Uses LSTM neural networks for time-series forecasting.
- Implements early stopping and model checkpointing to optimize training.
- Evaluates model performance using RMSE and MAPE.
- Visualizes training history, predictions, and future forecasts.
- Saves predictions and forecasts to CSV files.

---

## Dataset
The dataset used for training is a stock market dataset in CSV format. The required columns include:
- `Date` (Timestamp of stock trading days)
- `Open` (Opening price of the stock)
- `High` (Highest price of the stock on that day)
- `Low` (Lowest price of the stock on that day)
- `Close` (Closing price of the stock)
- `Volume` (Trading volume of the stock)

Ensure that the dataset follows this structure before running the model.

---

## Model Architecture
The model consists of:
1. **LSTM Layers:** Three LSTM layers with dropout for regularization.
2. **Dense Layers:** Two fully connected layers to map the output.
3. **Loss Function:** Mean Squared Error (MSE) for regression.
4. **Optimizer:** Adam optimizer for efficient training.
5. **Early Stopping & Checkpointing:** To prevent overfitting and save the best model.

---

## Setup & Installation
### Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Running the Project
1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-prediction-lstm.git
cd stock-prediction-lstm
```

2. Place the stock dataset (`stock_data.csv`) in the project folder.

3. Run the script:

```bash
python stock_prediction.py
```

This will:
- Load and preprocess the data.
- Train the LSTM model on historical stock prices.
- Evaluate and visualize results.
- Save the predictions and forecasted prices to CSV files.

---

## Results & Output
- **Training Metrics:** Training and validation loss plotted over epochs.
- **Predictions:** Model predictions vs. actual prices plotted.
- **Forecast:** Future stock price predictions plotted and saved to `price_forecast.csv`.
- **Performance Metrics:**
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

---

## Future Improvements
- Experiment with different LSTM architectures.
- Incorporate additional market indicators (e.g., moving averages, RSI, MACD).
- Implement real-time stock price prediction.
- Deploy the model as a web service using Flask/Django.

---

## License
This project is licensed under the MIT License.

---


