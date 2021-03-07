import streamlit as st
import numpy as np
import pandas as pd 
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.plot import add_changepoints_to_plot
from statsmodels.tools.eval_measures import rmse

st.title("Haglofs Digital Trend Analysis")

st.write("""

## Comparing worldwide and UK future search traffic for the brand using the FB Prophet Library

""")

dataset_name = st.selectbox("Select Dataset", ("UK - 5 year search traffic", "Worldwide - 5 year search traffic"))

def get_dataset(dataset_name):
    if dataset_name == "UK - 5 year search traffic":
        df = pd.read_csv("haglofs_uk.csv")
    else:
        df = pd.read_csv("haglofs_ww.csv")
    return df

df = get_dataset(dataset_name)


df = df.drop(df.index[0])
df['ds'] = df.index
df = df.reset_index(drop=True)
df = df.rename(columns={"Category: All categories": "y"})
df = df[['ds', 'y']]
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y'].astype(float)


data_load_state = st.text('Loading data...')

st.subheader(f'Graph Showing {dataset_name}')
st.write('x-axis - Weeks starting March 2016')
st.write('y-axis - Scaled search traffic from 0 to 100')

st.line_chart(df['y'])

weeks = st.slider("Weeks", 26, 104)

st.subheader(f'Predicting {weeks} weeks into the future')

m = Prophet(seasonality_mode='multiplicative')
m.fit(df)
future = m.make_future_dataframe(periods=weeks,freq='W')
forecast = m.predict(future)

st.pyplot(m.plot(forecast))
   


st.subheader(f'Exploring the changepoints in the data')
st.write('Red vertical lines indicate a change in the trend')

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast)

st.pyplot(fig)