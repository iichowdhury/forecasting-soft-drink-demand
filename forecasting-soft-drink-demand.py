import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load the dataset
bicupDataFrame = pd.read_csv('/Users/imadulislamchowdhury/Downloads/ml2/final_exam/bicup2006.csv')

#Part A: Data Preparation & Exploration
# Exploratory Data Analysis (EDA)
# Display a concise summary of the DataFrame, including the number of non-null entries and data types of each column
print(bicupDataFrame.info())

# Display the first 10 rows of the DataFrame to get a quick overview of the data
print(bicupDataFrame.head(10))

# Generate descriptive statistics that summarize the central tendency, dispersion, and shape of the dataset’s distribution, excluding NaN values
print(bicupDataFrame.describe())

# Display the data types of each column in the DataFrame
print(bicupDataFrame.dtypes)

# Display the dimensions of the DataFrame (number of rows, number of columns)
print(bicupDataFrame.shape)

#Transformation
# Convert DATE column to datetime format
bicupDataFrame['DATE'] = pd.to_datetime(bicupDataFrame['DATE'], format='%d-%b-%y')

# Aggregate the demand data by day
bicupDailyDemand = bicupDataFrame.groupby('DATE')['DEMAND'].sum().reset_index()

# Set DATE column as index and set frequency to daily
bicupDailyDemand.set_index('DATE', inplace=True)
bicupDailyDemand = bicupDailyDemand.asfreq('D')

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(bicupDailyDemand, label='Weekly Demand Chart')
plt.title('Weekly Demand for Soft Drink')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.xticks([bicupDailyDemand.index[0], bicupDailyDemand.index[6], bicupDailyDemand.index[13], bicupDailyDemand.index[20]])
plt.legend()
plt.show()

# Comment on visible trend, seasonality, or irregularity
print()
print("The time series graph presents a detailed depiction of fluctuations in weekly demand over a specified period. \nUpon careful examination, it becomes evident that the plot does not indicate any clear trends or seasonal patterns that could suggest a consistent increase or decrease in demand. \nInstead, the data appears to exhibit a level of randomness, with variations that do not conform to predictable cycles or trends, highlighting the complexity of consumer behavior during the observed timeframe.")

print()
print("Technical Checkpoint 1: ")
print()
print("Q1. What kind of problems can arise if we do not handle missing time steps in time series data?")
print("Addressing missing time steps is very important for time series analysis. \nFailed to address missing time steps can mislead the analysis of trends and patterns, as well as making it too difficult to recognize the seasonality or cyclical behavior. \nTime series models like ARIMA, SARIMA very much depends on evenly spaced data points, because of this, missing steps can lead to data inaccuracies in model fitting and forecasting. \nFurthermore, statistical measures like mean and variance can also be be impacted, and stationarity tests which includes the Augmented Dickey-Fuller (ADF) test, \ncan produce misleading results, ultimately resulting in unreliable predictions and poor model performance.")
print()
print("Q2. How would differencing help here, and how do you detect the correct level of differencing needed?")
print("Differencing is crucial for many forecasting models to remove the trends and seasonality, it helps to stabilize the mean and ensure that the time series is stationary. \nTo detect the proper level of differencing, we need to visually inspect the data for trends, also use the ADF test to evaluate stationarity, and analyze the autocorrelation function (ACF) plot. \nIt is suggested to start with the first difference and check for stationarity; if needed, a second difference can be applied, but needs to be careful avoid over-differencing, which can introduce unnecessary noise.")
print()



#Part B: Time Series Analysis
#Stationarity Check
# Perform Augmented Dickey-Fuller test
result = adfuller(bicupDailyDemand['DEMAND'])
print()
print('ADF Statistic for Soft Drink:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])
print()
if result[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is not stationary.")

# Plot rolling mean and standard deviation
bicupRollingMean = bicupDailyDemand.rolling(window=3).mean()
bicupRollingSTD = bicupDailyDemand.rolling(window=3).std()

plt.figure(figsize=(12, 6))
plt.plot(bicupDailyDemand, label='Weekly Demand')
plt.plot(bicupRollingMean, label='Rolling Mean', color='red')
plt.plot(bicupRollingSTD, label='Rolling Std', color='green')
plt.title('Rolling Mean and Standard Deviation for Soft Drink')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.xticks([bicupDailyDemand.index[0], bicupDailyDemand.index[6], bicupDailyDemand.index[13], bicupDailyDemand.index[20]])
plt.legend()
plt.show()

print()
print("The graphical representation showcasing the rolling mean and standard deviation provides a comprehensive view of the fluctuations in weekly demand. \nThis analysis highlights the inherent variations that occur over time, reflecting the dynamic nature of consumer behavior. \nNevertheless, it is important to note that the data does not reveal any clear trends or seasonal patterns, \nsuggesting that the demand remains relatively unpredictable and lacks consistent cyclical behavior.")

#Decompose the Series

# Decompose the time series
bicupDecomposition = seasonal_decompose(bicupDailyDemand, model='additive')

# Extract trend, seasonality, and residuals
bicupTrend = bicupDecomposition.trend
bicupSeasonal = bicupDecomposition.seasonal
bicupResidual = bicupDecomposition.resid

# Plot the decomposed components
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(bicupDailyDemand, label='Original')
plt.xticks([bicupDailyDemand.index[0], bicupDailyDemand.index[6], bicupDailyDemand.index[13], bicupDailyDemand.index[20]])
plt.legend(loc='best')

plt.subplot(412)
plt.plot(bicupTrend, label='Trend')
plt.xticks([bicupDailyDemand.index[0], bicupDailyDemand.index[6], bicupDailyDemand.index[13], bicupDailyDemand.index[20]])
plt.legend(loc='best')

plt.subplot(413)
plt.plot(bicupSeasonal, label='Seasonality')
plt.xticks([bicupDailyDemand.index[0], bicupDailyDemand.index[6], bicupDailyDemand.index[13], bicupDailyDemand.index[20]])
plt.legend(loc='best')

plt.subplot(414)
plt.plot(bicupResidual, label='Residuals')
plt.xticks([bicupDailyDemand.index[0], bicupDailyDemand.index[6], bicupDailyDemand.index[13], bicupDailyDemand.index[20]])
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Explain the type of seasonality observed
print()
print("The seasonal decomposition plot shows fluctuations in weekly demand. The seasonality component indicates a weekly pattern in the data.")

print()
print("Technical Checkpoint 2: ")
print()
print("Q3. Why is stationarity important for ARIMA-type models?")
print("Stationarity is critical for time series models like ARIMA and SARIMA as it influences the statistical properties, which includes the mean and variance, remain constant over time. \nIf you have a non-stationary data, it can have an impact on the model's ability to accurately capture the underlying patterns, leading to unreliable forecasts. \nBy ensuring stationarity confirms that the important relationships between the time steps are stable, allowing the ARIMA to effectively model the autocorrelations and trends in the data. \nBecause of this, transformations like differencing, are often necessary to achieve stationarity before applying ARIMA.")
print()
print("Q4. When would you use additive decomposition over multiplicative decomposition?")
print("Additive decomposition is appropriate when the seasonal variations of the data in a time series are stable, and they are independent of the trend. \nOn the other hand, multiplicative decomposition is best when seasonal variations can vary in proportion to the trend, showing that the seasonality adjustments with the overall level of the series. \nFor example, if the holiday sales increase along an overall upward trend, a multiplicative model would be more appropriate.")
print()



#Part C: Forecasting Models

# Assume your bicupDailyDemand is a time series with daily frequency but only 21 days
# We'll use 80% for training and 20% for testing (approx. 16 train, 5 test)
bicupTrainSize = int(len(bicupDailyDemand) * 0.8)
train, test = bicupDailyDemand[:bicupTrainSize], bicupDailyDemand[bicupTrainSize:]


# Implement SARIMA model
bicupSarimaModel = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
bicupSarimaResult = bicupSarimaModel.fit(disp=False)

# Forecast next 2 weeks (14 days)
bicupSarimaForecast = bicupSarimaResult.get_forecast(steps=14)
bicupSarimaPred = bicupSarimaForecast.predicted_mean

# Implement LSTM model
# Prepare data
bicupScaler = MinMaxScaler()
bicupScaledData = bicupScaler.fit_transform(bicupDailyDemand.values.reshape(-1, 1))

X, y = [], []
bicupLookBack = 3
for i in range(bicupLookBack, len(bicupScaledData)):
    X.append(bicupScaledData[i - bicupLookBack:i])
    y.append(bicupScaledData[i])

X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test = X[:bicupTrainSize - bicupLookBack], X[bicupTrainSize - bicupLookBack:]
y_train, y_test = y[:bicupTrainSize - bicupLookBack], y[bicupTrainSize - bicupLookBack:]

# Build model
bicupLSTM = Sequential()
bicupLSTM.add(LSTM(50, activation='relu', input_shape=(bicupLookBack, 1)))
bicupLSTM.add(Dense(1))
bicupLSTM.compile(optimizer='adam', loss='mse')

# Train
bicupLSTM.fit(X_train, y_train, epochs=200, verbose=0)

# Forecast 2 weeks
bicupLastSequence = bicupScaledData[-bicupLookBack:]
bicupLstmPredictions = []
for _ in range(14):
    pred = bicupLSTM.predict(bicupLastSequence.reshape(1, bicupLookBack, 1), verbose=0)
    bicupLstmPredictions.append(pred[0][0])
    bicupLastSequence = np.append(bicupLastSequence[1:], pred, axis=0)

bicupLstmForecast = bicupScaler.inverse_transform(np.array(bicupLstmPredictions).reshape(-1, 1))


# Implement Prophet model
# Prepare dataframe
bicupDataFrame_prophet = bicupDailyDemand.reset_index()
bicupDataFrame_prophet.columns = ['ds', 'y']

# Train/Test split
bicupDataFrame_train = bicupDataFrame_prophet.iloc[:bicupTrainSize]

# Fit Prophet
bicupProphet = Prophet()
bicupProphet.fit(bicupDataFrame_train)

# Future dataframe
bicupProphetFuture = bicupProphet.make_future_dataframe(periods=19)
bicupProphetForecast = bicupProphet.predict(bicupProphetFuture)

# Ensure future index is consistent
future_index = pd.date_range(start=bicupDailyDemand.index[-1] + pd.Timedelta(days=1), periods=14)

# Extract Prophet forecast for the last 14 days only
bicupProphetForecast = bicupProphetForecast.set_index('ds').loc[future_index, 'yhat']

# Plot all
plt.figure(figsize=(14, 6))
plt.plot(bicupDailyDemand, label='Original Demand', color='black')

# Plot each model's forecast
plt.plot(future_index, bicupSarimaPred[:14], label='SARIMA Forecast', color='green', linestyle='--', marker='o')
plt.plot(future_index, bicupLstmForecast.flatten(), label='LSTM Forecast', color='orange', linestyle='--', marker='x')
plt.plot(future_index, bicupProphetForecast.values, label='Prophet Forecast', color='blue', linestyle='--', marker='s')

# Styling
plt.title('Forecast Comparison for Soft Drinks: SARIMA vs LSTM vs Prophet (Next 2 Weeks)')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.axvline(bicupDailyDemand.index[-1], color='gray', linestyle='--', label='Forecast Start')
plt.legend()
#plt.xticks([bicupDailyDemand.index[0], bicupDailyDemand.index[6], bicupDailyDemand.index[13], bicupDailyDemand.index[20]])
plt.grid(True)
plt.tight_layout()
plt.show()



# Actual 5 days of demand (from test set)
bicupActual = test[:5].values.flatten()

# Slice forecasts to match
bicupSarimaPreds = bicupSarimaPred[:5].values
bicupLstmPreds = bicupLstmForecast[:5].flatten()
bicupProphetPreds = bicupProphetForecast[:5].values


print()
print("Technical Checkpoint 3: ")
print()
print("Q5. How would you decide the values for SARIMA’s (p, d, q)(P, D, Q)s parameters?")
print("To define the values of the (p, d, q)(P, D, Q)s parameters for SARIMA, first need to identify d which is the number of times to difference the non-seasonal part of the series, \nand on the other side D is the number of times to variance of the seasonal part of the series. \nThis is naturally done by using a stationarity test like the Augmented Dickey-Fuller (ADF) test, which tells the user whether a series is stationary or non-stationary. \nIf the series is non-stationary, apply differencing until it becomes stationary. \nNext, identify the p and q using the ACF and PACF plots while observing where (and how) the autocorrelations and partial autocorrelations are cut off. \nAlso use these plots to identify the seasonal components (P and Q). \nFinally, as for our dataset it has daily observations with weekly seasonality, we set s to 7 to capture the weekly cycles.")
print()


#Part D: Model Evaluation & Business Recommendation
def evaluate_model(name, bicupActual, bicupPredicted):
    bicupRMSE = round(np.sqrt(mean_squared_error(bicupActual, bicupPredicted)), 2)
    bicupMAE = round(mean_absolute_error(bicupActual, bicupPredicted), 2)
    bicupMSE = round(mean_squared_error(bicupActual, bicupPredicted), 2)
    bicupMAPE = round(np.mean(np.abs((bicupActual, bicupPredicted) / bicupActual)), 2)

    print(f"{name} Evaluation:")
    print(f"  MAE : {bicupMAE:.2f}")
    print(f"  RMSE: {bicupRMSE:.2f}")
    print(f"  MSE : {bicupMSE:.2f}")
    print(f"  MAPE: {bicupMAPE:.2f}%")
    print("-" * 30)

evaluate_model("SARIMA", bicupActual, bicupSarimaPreds)
evaluate_model("LSTM", bicupActual, bicupLstmPreds)
evaluate_model("Prophet", bicupActual, bicupProphetPreds)

print()
print("As par the model performances, the SARIMA model beats both LSTM and Prophet, achieving the lowest MAE (121.55), RMSE (165.86), and MAPE (1.07%), thus providing the most accurate forecasts. \nLSTM performs moderately with higher error rates (MAE: 692.00, RMSE: 785.08, MAPE: 3.23%) but still outperforms Prophet, \nwhich has the highest error rates (MAE: 1251.02, RMSE: 1517.39, MAPE: 5.26%), making it the least reliable model. \nTherefore, SARIMA is the most appropriate model for forecasting in this context.")
print()
print("With an average error of only 1.07% (MAPE), the SARIMA model shows strong predictive confidence. \nThe business should consider boosting production next two weeks. \nHowever, it is crucial to monitor external factors like market trends and unexpected disruptions, as they may impact forecast accuracy.")
print()


print()
print("Technical Checkpoint 4: ")
print()
print("Q6. How would you explain residuals and confidence intervals to a non-technical manager?")
print("Residuals are defined as the differences between the actual values and the model output predicted values. \nIdeally, residuals should be small and randomly scattered, showing that the model is working. \nOn the other hand, confidence intervals represent the range of values within which we expect the true value of a forecast or parameter to lie, with a certain level of certainty (like 95%). \nFor example, if we forecast the sales to be 100 units with a 95% confidence interval of 90 to 110, it means we are 95% confident the actual sales will lie somewhere in that range.")
print()
print("Q7. What limitations does your forecast have, and how might external factors affect it?")
print("Predictions depends on the trends from the historical data and assume that the past data is a good introduction to the future. \nThey are consequently well-suited to projecting the effects of changes we know about (like new products) or the kinds of changes we tend to see over time (like technological ones) and that are very hard to forecast otherwise. \nHowever, forecasts often have trouble with the big, unexpected changes. \nThat is because their accuracy relies heavily on the quality of the data used to make them and the assumptions built into the model.")
print()
