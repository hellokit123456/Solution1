# Stock Price Prediction with LSTM

## Overview

This project demonstrates how to use a Long Short-Term Memory (LSTM) neural network for predicting stock prices, specifically focusing on Reliance Industries Limited (RELIANCE.NS) using daily historical data. The workflow is implemented in Python and designed to be run in a Jupyter Notebook, making it easy to follow and adapt for other stocks or time series forecasting tasks.

The main objective is to forecast future stock prices based on past price movements using deep learning. LSTM networks are particularly well-suited for this task due to their ability to capture long-term dependencies in sequential data, which is essential for financial time series analysis.

## Project Workflow

### 1. Data Acquisition

The project utilizes the `yfinance` Python library to download historical stock data for Reliance Industries from Yahoo Finance. The data spans from January 1, 2012, to January 1, 2025, at a daily interval. This dataset includes columns such as Open, High, Low, Close, Adj Close, and Volume.

### 2. Data Preprocessing

Preprocessing is a crucial step in any machine learning task, especially for time series data. In this project:

- **Sorting and Cleaning:** The raw data is sorted by date to ensure chronological order, which is vital for time series modeling.
- **Mid Price Calculation:** The mid price for each day is calculated as the average of the 'High' and 'Low' prices. This value is used as the target for prediction, providing a smoother and more representative price series than using just the closing price.
- **Normalization:** The mid price data is normalized using a MinMaxScaler. Normalization is essential for neural network training, as it ensures that all input values are on a similar scale, which speeds up convergence and improves model performance.
- **Sequence Creation:** The data is structured into sequences, where each input sequence consists of a fixed number of consecutive days (e.g., 60 days), and the target is the mid price of the next day. This sliding window approach allows the LSTM to learn from historical patterns.

### 3. Model Building

An LSTM-based neural network is constructed using TensorFlow and Keras. The model architecture typically includes:

- Two LSTM layers to capture temporal dependencies in the data.
- Dropout layers for regularization and to prevent overfitting.
- A Dense output layer to predict the mid price.

The model is compiled with the Adam optimizer and mean squared error (MSE) as the loss function, which is standard for regression tasks.

### 4. Model Training

The model is trained on the prepared sequences for a set number of epochs (e.g., 50), using a batch size (e.g., 32) and a validation split to monitor performance on unseen data. Training and validation loss curves are plotted to visualize the learning process and check for overfitting or underfitting.

### 5. Prediction and Visualization

After training, the model is used to predict mid prices on a test set (e.g., the last 200 days of data). The predicted values are inverse-transformed to the original scale and plotted alongside the actual mid prices. This visual comparison helps assess the model's predictive accuracy and its ability to follow real price trends.

## How to Run This Project

1. **Clone or Download the Repository:**  
   Ensure you have all the required files, including the Jupyter Notebook.

2. **Install Dependencies:**  
   Install the necessary Python libraries using pip:
   ```
   pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
   ```

3. **Run the Notebook:**  
   Open the notebook in Jupyter and execute each cell in order. The notebook is organized into logical sections for data loading, preprocessing, model building, training, and evaluation.

4. **Review Results:**  
   Examine the plots to compare predicted and actual mid prices. You can experiment with different window sizes, model architectures, or hyperparameters to improve performance.

## Customization

- **Change the Stock:**  
  Modify the ticker symbol in the `yfinance.download()` function to analyze a different stock.
- **Adjust Window Size:**  
  Change the sequence length (number of days) used for prediction to see how it affects accuracy.
- **Add Features:**  
  Incorporate other columns (Open, Close, Volume) as additional features for the LSTM.
- **Experiment with the Model:**  
  Try different numbers of LSTM layers, units, or dropout rates.

## Limitations & Considerations

- **Financial data is noisy and non-stationary.** LSTM models can capture patterns but may not always generalize well to future unseen data, especially in volatile markets.
- **This project is for educational and research purposes only.** It is not intended for real-world trading or investment decisions.
- **Further improvements:**  
  You can enhance the model by including more features, using more advanced architectures (e.g., bidirectional LSTM, attention mechanisms), or incorporating external data (news sentiment, macroeconomic indicators).

## Conclusion

This project provides a practical introduction to time series forecasting with LSTM networks for stock price prediction. By following the notebook, you will learn how to preprocess financial data, build and train an LSTM model, and evaluate its predictive performance. The approach can be extended to other stocks, indices, or time series forecasting problems with minimal adjustments.
