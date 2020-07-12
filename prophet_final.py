import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
from time import mktime
from datetime import timedelta
import yfinance as yf
from pycoingecko import CoinGeckoAPI




def clean(df):
  '''This function will clean the Google Trends CSV so that it can be inputted nicely inside Facebook Prophet
  '''
  trends = pd.read_csv(df, header=1)
  st.write(trends)
  trends.columns = ["ds", "y"]
  trends['ds'] = pd.to_datetime(trends['ds'], errors='coerce')
  trends = trends.sort_values('ds', ascending=True)
  return trends

def get_stock(company, start, end):
  '''This function will do a scrape and fetch the pertaining stock data in Yahoo. 
  '''
  data = yf.download(company, start, end) 
  return data

def crypto_stat(crypto_name, currency, start1, end1):
  cg = CoinGeckoAPI()
  conv_from1 = time.mktime(time.strptime(start1, '%Y-%m-%d'))
  conv_to1 = time.mktime(time.strptime(end1, '%Y-%m-%d'))
  crypto = cg.get_coin_market_chart_range_by_id(crypto_name, 'usd', conv_from1, conv_to1)
  historical=crypto['prices'][:]
  return historical

def get_crypto_list():
  cg = CoinGeckoAPI()
  c_list = cg.get_coins_list()
  d = []
  for i in c_list:
    d.append(i['id'])
  return d

def get_currency():
  cg = CoinGeckoAPI()
  sc_list = cg.get_supported_vs_currencies()
  s = []
  for i in sc_list:
    s.append(i)
  return s

def clean_crypto(df):
  '''This function will clean the Google Trends CSV so that it can be inputted nicely inside Facebook Prophet
  '''
  trends = pd.read_csv(df, header=1)
  st.write(trends)
  trends.columns = ["ds", "y"]
  trends['ds'] = pd.to_datetime(trends['ds'], errors='coerce')
  trends = trends.sort_values('ds', ascending=True)
  return trends




def main():
  '''This is the main function to operate the app
  '''
  st.title('Forecast Time Series Data with Facebook Prophet')
  st.image("graph.png", use_column_width=True)  
  activity = ['Cryptocurrency Prediction', 'Stock Prediction', 'Google Trends']
  choice = st.selectbox("Select Activity", activity)  
  st.write('### General information')
  st.write('This is only an example of how you can use Facebook Prophet as a way to predict trends for cryptocurrency predictions, stock predictions, and also google trends. This will help the user to visualize and understand how Time Series Machine Learning can be applied. **Please do not take any results as financial advice.** This is for education purposes only.')
  
  if choice == 'Stock Prediction':
    '''This activity will allow the user to fetch stock data through an API and use that data to predict stock price
    '''
    st.info("Predict Stock with Facebook Prophet")

    #Allows users to type in their stock ticker
    st.write('### (1) Enter the stock ticker of the company you want to predict')
    if st.button("Show Ticker Examples"):
      st.json({
        "Facebook" : "FB",
        "Google" : "GOOGL",
        "Apple" : "AAPL",
        "Amazon" : "AMZN",
        "Microsoft" : "MSFT",
        "Tesla" : "TSLA"
      })
    company = st.text_input("Enter Name for example FB for facebook", 'FB')

    #Allows users to enter the start and end date of the historical stock data
    st.write('### (2) Enter the historical start date and end dates for your stock')
    start = st.date_input('Enter Stock Start Date', datetime.date(2011,1,1))
  
    end = st.date_input('Enter Stock End Date', datetime.date(2020,1,1))
    stock_info = get_stock(company, start, end)
    

    if st.button("Fetch Stock using Yahoo Finance"):
      st.write(stock_info)
    
    #Download stock information
    st.success('Download stock data if needed')   
    csv_exp = stock_info.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}">Download Stock Info,</a> right-click and save link as ** [ticker].csv**'
    st.markdown(href, unsafe_allow_html=True)

    #Clean Data, Keep only date and close, Remove [Open, High,  Low, Adj CLose, and Volume]
    stock_info = stock_info.reset_index()    
    stock_info[['ds','y']] = stock_info[['Date' ,'Adj Close']]
    stock_info = stock_info[['ds','y']]
    train_dataset = stock_info.copy()
    prophet_basic = Prophet()
    prophet_basic.fit(train_dataset)
    st.write('### (3) Select how many days you would like to forecast')
    st.warning('The longer the forecasted days, the less accurate the predictions are.')

    #Allows the user to choose how many days they want to forecast
    periods_inputs = st.slider('Select how many days you want to forecast in the future',
    min_value = 1, max_value = 365)

    #Allows the machine to predict
    if st.button("Predict"):
      future = prophet_basic.make_future_dataframe(periods=periods_inputs)
   
      forecast = prophet_basic.predict(future)        
      fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
      fig1=prophet_basic.plot(forecast)
      """
      ### Data Table
      The table shows the predicted value(depicted as yhat) and the upper and lower variance confidence levels.
      """     
      st.write(fcst)
      """
      ### GRAPH1
      The next visual show in black dots are the actual data points and blue line is the predicted line over the given time.
      """          
      st.write(fig1)

      """
      ### GRAPH 2
      The graphs below show a high level trend of predicted values, day of week trends, and yearly trends, depending on the data. The blue line is the prediction line and the blue area represents the confidence intervals.
      """
      fig2 = prophet_basic.plot_components(forecast)
      st.write(fig2)         

  if choice == 'Google Trends':
    '''This will activity will allow the user to upload Google Trend csv files and predict the trend
    '''    
    st.info("Predict Google Trends with Facebook Prophet")
    st.write('### Directions')
    """
    * Go to https://trends.google.com/trends/ and download the selected data(5 years worth) in csv.
    * Drag and Drop into the Uploader.
    """
    gf = st.file_uploader('Upload the Google trend time series csv file here', type='csv', encoding='auto')
    
    st.write('### (1) Select how many days you would like to forecast')
    st.warning('The longer the forecasted days, the less accurate the predictions are.')

    #Allows the user to choose how many days they want to forecast
    periods_input = st.slider('Select how many days you want to forecast in the future',
    min_value = 1, max_value = 365)   

    #call clean function    
    if st.button("Predict"):
      '''View the csv data
      '''
      ready = clean(gf)
      st.write(ready.shape)
  
      max_date = ready['ds'].max()
      min_date = ready['ds'].min()

  # #lets users select the most recent and oldest data
  #   dateselection = st.selectbox("Check Date",['Most Recent Date','Oldest Date'])
  #   if dateselection == "Most Recent Date":
  #     st.write("The most recent date", max_date)
  #   else:
  #     st.write("The oldest date", min_date)


    #Train facebook prophet
    
      train_dataset = ready.copy()
      prophet_basic = Prophet()
      prophet_basic.fit(train_dataset)
      """
      ### (2) Read the Data Table
      The table shows the predicted value(depicted as yhat) and the upper and lower variance confidence levels.
      """ 
      #Fit and forecast data to prophet
      future = prophet_basic.make_future_dataframe(periods=periods_input)
      forecast = prophet_basic.predict(future)
      
      #Plotting the predicted data
      fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
      fcst_filtered =  fcst[fcst['ds'] > max_date]    
      st.write(fcst_filtered)
      
      """
      ### GRAPH1
      TThe next visual show in black dots are the actual data points and blue line is the predicted line over the given time.
      """  
      fig1=prophet_basic.plot(forecast)
      st.write(fig1)
      """
      ### GRAPH 2
      The graphs below show a high level trend of predicted values, day of week trends, and yearly trends, depending on the data.
      """
      fig2 = prophet_basic.plot_components(forecast)
      st.write(fig2)
      """
      ### (3) Download the Forecasted Data
      You will be able to download created forecast data for your reference.
      """
      #Allows the user to download the forecasted data
      csv_exp = fcst.to_csv(index=False)
      b64 = base64.b64encode(csv_exp.encode()).decode()
      href = f'<a href="data:file/csv;base64,{b64}">Download Forecast File,</a> right-click and save link as ** [forecasted dataset].csv**'
      st.markdown(href, unsafe_allow_html=True)

  if choice == 'Cryptocurrency Prediction':
      '''This activity will allow the user to fetch stock data through an API and use that data to predict stock price
      '''
      st.info("Predict Cryptocurrency with Facebook Prophet")

      #Allows users to type in their stock ticker
      st.write('### (1) Enter the cryptocurrency you want to predict')
      
      #this lets you select
      call_id = get_crypto_list()
      crypto_avail = st.selectbox("Check available cryptocurrency in the list", call_id) 

      crypto_name = st.text_input("Enter cryptocurrency name in lowercase and no spaces", 'bitcoin')
      currency_lists = get_currency()
      currency = st.selectbox("Choose your currency", currency_lists)

      #Allows users to enter the start and end date of the historical crypto currency data
      st.write('### (2) Enter the historical start date and end dates for your cryptocurrency')
      #start1 = st.date_input('Enter Cryptocurrency Start Date', datetime.date(2016,1,1))
      start1 = st.text_input('Enter Cryptocurrency Start Date', '2011-09-11')
    
      #end1 = st.date_input('Enter Cryptocurrency End Date', datetime.date(2020,7,1))
      end1 = st.text_input('Enter Cryptocurrency End Date', '2020-07-03')

      crypto = crypto_stat(crypto_name, currency, start1, end1)
      df = pd.DataFrame(crypto)

      # #Clean Data, Keep only date and close, Remove [Open, High,  Low, Adj CLose, and Volume]
      crypto_info = df.reset_index()    
      crypto_info[['ds','y']] = crypto_info[[0 , 1]]
      crypto_info = crypto_info[['ds','y']]

      crypto_info['ds'] = pd.to_datetime(crypto_info['ds'], unit='ms')
      
      #allows users to see the data
      if st.button("View Cryptocurrency Data"):
        st.write(crypto_info)
      
      #Download stock information
      st.success('Download Cryptocurrency data if needed')   
      csv_exp = df.to_csv(index=False)
      b64 = base64.b64encode(csv_exp.encode()).decode()  
      href = f'<a href="data:file/csv;base64,{b64}">Download Stock Info,</a> right-click and save link as ** [ticker].csv**'
      st.markdown(href, unsafe_allow_html=True)



      train_dataset = crypto_info.copy()
      prophet_basic = Prophet()
      prophet_basic.fit(train_dataset)

      st.write('### (3) Select how many days you would like to forecast')
      st.warning('The longer the forecasted days, the less accurate the predictions are.')

      #Allows the user to choose how many days they want to forecast
      periods_inputs = st.slider('Select how many days you want to forecast in the future', min_value = 1, max_value = 365)

      #Allows the machine to predict
      if st.button("Predict"):
        future = prophet_basic.make_future_dataframe(periods=periods_inputs)
    
        forecast = prophet_basic.predict(future)        
        fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        fig1=prophet_basic.plot(forecast)
        """
        ### Data Table
        The table shows the predicted value(depicted as yhat) and the upper and lower variance confidence levels.
        """     
        st.write(fcst)
        """
        ### GRAPH1
        The next visual show in black dots are the actual data points and blue line is the predicted line over the given time.
        """          
        st.write(fig1)

        """
        ### GRAPH 2
        The graphs below show a high level trend of predicted values, day of week trends, and yearly trends, depending on the data. The blue line is the prediction line and the blue area represents the confidence intervals.
        """
        fig2 = prophet_basic.plot_components(forecast)
        st.write(fig2)        
  

if __name__ == '__main__':
  main()
