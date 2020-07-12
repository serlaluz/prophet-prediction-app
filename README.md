# prophet-prediction-app

![alt text](https://github.com/serlaluz/prophet-prediction-app/blob/master/graph.png?raw=true)
Simple web app that uses Facebook's Prophet model to predict trends and stock prices

## General 
This is a simple app that utilizes facebook's open source project Prophet. This machine learning model is good at interpreting and making predictions based on time series data. With this app you can do the following things

1. Predict Stock Price
2. Predict Google Trends

## Machine Model Used

I used facebook's opensource project Prophet. 

```Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.```

Project URL - https://prophet-prediction-app.herokuapp.com/

## Steps

### Cryptocurrency App
1. Select Activity : "Stock Prediction" 
2. Input target cryptocurrency, eg. bitcoin, ethereum, ripple
3. Enter currency
4. Enter Start and End Date
5. Download the stock data(if needed)
6. Choose days you want to predict ahead
7. Press **Predict**
8. Read Chart

### Stock App
1. Select Activity : "Stock Prediction" 
2. Input target stock's ticker, eg. facebook as **FB** google as **GOOG**
3. Enter Start and End Date of stock data
4. Download the stock data(if needed)
5. Choose days you want to predict ahead
6. Press **Predict**
7. Read Chart

### Google Trends
1. Select Activity : "Google Trends"
2. visit https://trends.google.com/trends, enter keyword and download 5 years of data
3. Upload csv file
4. Choose days you want to predict ahead
5. Press **Predict**
6. Read Chart
7. Download forecasted datatable

## Upcoming Features
* Add Cryptocurrency Predictions(July 12th)

## UI tool
* Streamlit, https://www.streamlit.io/

## Deployed
* Heroku, https://www.heroku.com

## Required Files
1. setup.sh
2. Procfile
3. requirements.txt

### References
1. JCharis Tech's video on How to deploy Streamlit to Heroku, https://www.youtube.com/watch?v=skpiLtEN3yk&list=PLJ39kWiJXSixyRMcn3lrbv8xI8ZZoYNZU&index=3
2. Facebook's Prophet, https://facebook.github.io/prophet/
3. Rakshitratan, https://github.com/rakshitratan/Stock-Price-Prediction
4. Viral ML, https://www.viralml.com/video-content.html?v=AX1wKnBPhvU
5. https://github.com/Conformist101/streamlit_stock_predict_app

