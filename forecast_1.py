import streamlit as st

from datetime import date
# from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
# from tqdm.auto import tqdm



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from neuralprophet import NeuralProphet
import urllib.request



urlfile = "https://raw.githubusercontent.com/marqsin/NP_2/main/pp_new1.csv"
urllib.request.urlretrieve(urlfile, 'ppri_new1.csv')

prods = ('PVC', 'PET', 'PE', 'PP', 'PS')
selected_prod = st.selectbox('Select dataset for prediction', prods)

# n_months = st.slider('Months of prediction:', 1, 12)
n_months = 6
period = n_months



# @st.cache
# def load_data():
data = pd.read_csv('ppri_new1.csv')
data['Month'] = pd.DatetimeIndex(data['Month'])
data = data.astype({"PVC":'float', "PP":'float', "PET":'float', "PE":'float', "GPPS":'float'})
data.rename(columns = {'GPPS':'PS'}, inplace = True)
    # return data

# load_data()

# df = data.copy()

def select_data(ticker):
    df_prod = data[['Month', ticker]]
    # data.drop(['PP', 'GPPS', 'PET', 'PE'], axis=1, inplace=True)
    # df_prod.rename(columns = {'Month':'ds', ticker:'y'}, inplace = True)

    # data.reset_index(inplace=True)
    return df_prod

## Summary of the dataset
#print(df1.info())

## Renaming columns
# df1.rename(columns = {'Month':'ds'}, inplace = True)
#print(df1.head())

# Convert ds to datetime type
# df1['ds'] = pd.DatetimeIndex(df1['ds'])

# df1 = df1.astype({"PVC":'float', "PP":'float', "PET":'float',}) 

# print(df1.info())

# print(df1.dtypes)

# df11 = df1.copy()

# df11.drop(['PP', 'GPPS', 'PET', 'PE'], axis=1, inplace=True) # drops all the columns not needed, by column name. axis=1 says that we are dropping columns rather than rows
# print(df11.info())

# df = df11.copy()

## Renaming columns
# df.rename(columns = {'Month':'ds', 'PVC':'y'}, inplace = True)
# print(df.head())

data_load_state = st.text('Loading data...')
df = select_data(selected_prod)
data_load_state.text('Data loaded. Preparing forecast...')
st.subheader('Raw data')
st.write(df.tail())

df.rename(columns = {'Month':'ds', selected_prod:'y'}, inplace = True)

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="y_price"))
	fig.layout.update(title_text=f"Price chart - {selected_prod}", xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()




# Visualize the dataset
# ax = df.set_index('ds').plot(figsize=(15, 12))
# ax.set_ylabel('Monthly Price')
# ax.set_xlabel('Month')
# plt.title('PVC Price')
# plt.show()



# # Predict forecast with Prophet.
# df_train = data[['Date','Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)



# forecast_state = st.text('Loading data...')

m = NeuralProphet()
df_train, df_val = m.split_df(df, freq='M', valid_p = 0.02)
metrics = m.fit(df_train, freq='M', validation_df=df_val) # plot_live_loss=True taken out as it wd give error here

# print(metrics)

future = m.make_future_dataframe(df, periods=period, n_historic_predictions=len(df))
forecast = m.predict(future)
# print(forecast)

data_load_state.text('Forecast ready!')
# forecast_state.text('Preparing forecast... done!')

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_months} months')
# fig1 = plot_plotly(m, forecast)
fig1 = m.plot(forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)






# Visualize the forecast from the model
# fig_forecast = m.plot(forecast)

# Visualize the model parameters
fig_model = m.plot_parameters()
st.write(fig_model )

# Visualize Training and Validation Loss
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(metrics["MAE"], '-o', label="Training Loss")  
ax.plot(metrics["MAE_val"], '-r', label="Validation Loss")
ax.legend(loc='center right', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("Epoch", fontsize=28)
ax.set_ylabel("Loss", fontsize=28)
ax.set_title("Model Loss (MAE)", fontsize=28)
