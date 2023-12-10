import pandas as pd
import streamlit as st 
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

def load_data():
    # Function for loading data
    dataf_apu = pd.read_csv("Apu.csv", index_col="date")
    dataf_suu = pd.read_csv("Suu.csv", index_col="date")
    dataf = pd.concat([dataf_apu, dataf_suu], axis=0)

    numeric_dataf = dataf.select_dtypes(['float', 'int'])
    numeric_cols = numeric_dataf.columns
    
    text_dataf = dataf.select_dtypes(['object'])
    text_cols = text_dataf.columns
    
    stock_column = dataf['name']
    
    unique_stocks = stock_column.unique()
    
    return dataf, numeric_cols, text_cols, unique_stocks

dataf, numeric_cols, text_cols, unique_stocks = load_data()

# Title of dashboard
st.title("Ratio Analysis & Prediction of Stocks Dashboard")

# Link for the news
company_links = {
    'APU': 'https://mse.mn/mn/company/90',
    'SUU': 'https://mse.mn/mn/company/135'
}

# Dictionary to store file paths for ratio data
ratio_files = {'APU': 'ratioapu.csv', 'SUU': 'ratiosuu.csv'}
valuation_files = {'APU': 'valuationapu.csv', 'SUU': 'valuationsuu.csv'}

#Sidebar a title
st.sidebar.title("Settings")
st.sidebar.subheader("Timeseries settings")
feature_selection = st.sidebar.multiselect(label="Features to plot",
                                           options=numeric_cols)

stock_dropdown = st.sidebar.selectbox(label="Stock Ticker",
                                      options=unique_stocks)

print(feature_selection)

dataf_selected = dataf[dataf['name']==stock_dropdown]
dataf_features = dataf_selected[feature_selection]

plotly_figure = px.line(data_frame=dataf_features,
                        x=dataf_features.index, y=feature_selection,
                        title=(str(stock_dropdown) + ' ' +'timeline')
                       )

st.plotly_chart(plotly_figure)

ratio_data, valuation_data, news = st.tabs(["Ratio Analysis", "Valuation", "News"])

with ratio_data:
    st.header('Ratio Analysis')
    ratio_file = ratio_files.get(stock_dropdown)
    ratio_df = pd.read_csv(ratio_file)
    st.write(ratio_df)
    
with valuation_data:
    st.header("Valuation")
    valuation_file = valuation_files.get(stock_dropdown)
    valuation_df = pd.read_csv(valuation_file)
    st.write(valuation_df)
    
with news:
    st.header("News")
    company_link = company_links.get(stock_dropdown)
    embed_code = f'<iframe src="{company_link}" width="800" height="600"></iframe>'
    st.markdown(embed_code, unsafe_allow_html=True)
    
st.subheader("Prediction")

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

data2 = dataf[dataf['name'] == stock_dropdown]

st.subheader(f'{stock_dropdown} Raw Data')
st.write(data2.tail())


# Forecasting
df_train = data2.reset_index()[['date', 'close']]  # Reset index to make 'date' a regular column
df_train = df_train.rename(columns={"date": "ds", "close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

