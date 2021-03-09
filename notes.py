ax = df.plot(x='ds',y='y',figsize=(18,6))
        ax.set_xlabel("Weeks (Starting March 2016)")
        ax.set_ylabel("Google trend index")
        ax.set_title("Haglofs UK Search Traffic")

def plot_data(dataset_name):
    if dataset_name == "UK - 5 year search traffic":
        ax = df.plot(x='ds',y='y',figsize=(18,6))
        ax.set_xlabel("Weeks (Starting March 2016)")
        ax.set_ylabel("Google trend index")
        ax.set_title("Haglofs UK Search Traffic")
    else:
        ax = df.plot(x='ds',y='y',figsize=(18,6))
        ax.set_xlabel("Weeks (Starting March 2016)")
        ax.set_ylabel("Google trend index")
        ax.set_title("Haglofs Worldwide Search Traffic")
    return plt.show()

plt = plot_data(dataset_name)

plt.show()


train = df.iloc[:209]
test = df.iloc[209:]

m = Prophet(seasonality_mode='multiplicative')
m.fit(train)
future = m.make_future_dataframe(periods=52,freq='W')
forecast = m.predict(future)
predictions = forecast.iloc[-52:]['yhat']
rmse(predictions,test['y'])
st.write("")
