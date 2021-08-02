import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from data_cleaner import clean_data

import joblib

def main():
    st.title("Rossmann Pharmaceuticals Sales Prediction Dashboard")

    st.sidebar.write("Navigation")
    app_mode = st.sidebar.selectbox("Choose Here", ("Home", "Model Performance", "Predict Sales"))
    if app_mode == 'Home':
        st.write('''
        ## Introduction
        Rossmann Pharmaceuticals is a pharmaceutical chain that has 1115 stores. 
        Rossmann Pharmaceuticals’ finance team wants to forecast sales in all their 1115 stores across several cities six weeks ahead of time.
        The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.
        In this project, we are building an end-to-end product that delivers this prediction to analysts in the finance team.
        ''')

    elif app_mode == 'Model Performance':
        st.write('''
        ## Here are a few plots of the variables
        ''')
        model_file = 'forest-2021-07-30-05-45-41-547423.sav'
        model = pickle.load(open(model_file, 'rb'))

        df = pd.read_csv('./train_final.csv', na_values=['?', None, 'undefined'])
        df = df.reindex(index=df.index[::-1])


        #Select the Sales data of a specific store
        store_id = 16
        df_store = df[df['Store'] == store_id].head(100)

        st.subheader('Some Statistics on the data')
        st.table(df_store.describe().T)

        st.subheader('Number of days where the store ran Promotions')
        hist_values = np.histogram(df_store['Promo'], bins=2)[0]
        st.bar_chart(hist_values)

        st.subheader('Days the stores were Open')
        hist_values = np.histogram(df_store['Open'], bins=2)[0]
        st.bar_chart(hist_values)

        st.subheader('Distribution of data by day of week')
        hist_values = np.histogram(df_store['DayOfWeek'], bins=7)[0]
        st.bar_chart(hist_values)

        store_sales = df_store[['Sales']].values

        scaler = joblib.load('std_scaler.pkl')
        y = scaler.fit_transform(store_sales)

        X = df_store.drop('Sales', axis = 1).values

        score = model.score(X, y)
        mse = mean_squared_error(y, model.predict(X))

        st.subheader('Real Sales vs Predicted Sales')
        plot_df = pd.DataFrame(data = {'Sales': y.reshape((100,)), 'Predicted' : model.predict(X)})
        st.line_chart(plot_df)

        st.subheader(f'Model performance on the sales data for store: {store_id}')

        st.write(f'Model r2 Score: {score}')
        st.write(f'Model mean squared error: {mse}')

    elif app_mode == 'Predict Sales':

        st.subheader('Upload the data you want to perform predictions on')

        csv_file =  st.file_uploader("Upload Data", type = ['csv'])
        if csv_file is not None:
            data = pd.read_csv(csv_file, na_values=['?', None, 'undefined'])
            data = clean_data(data)
            data = data.reindex(index=data.index[::-1])
            st.subheader('First 5 rows')
            st.table(data.head(5))

            st.subheader('Last 5 rows')
            st.table(data.tail(5))

            st.subheader('Some Statistics on the data')
            st.table(data.describe().T)

            model_file = 'forest-2021-07-30-05-45-41-547423.sav'
            model = pickle.load(open(model_file, 'rb'))
            
            x = data[data['Store'] == min(data['Store'])].values
            predicted_sales = model.predict(x)

            scaler = joblib.load('std_scaler.pkl')

            st.subheader('Predicted Sales Plot for the first Store')
            plot_df = pd.DataFrame(data = {'Sales': scaler.inverse_transform(predicted_sales)})
            st.line_chart(plot_df)

            st.subheader('Download Predicted Sales')
            full_prediction = model.predict(data.values)

            

            result_df = pd.DataFrame(data = {'Id': range(1,len(full_prediction)+1), 'Sales': scaler.inverse_transform(full_prediction)})
            csv  = result_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}" download="PredictedSales.csv">Download csv file</a>'
            st.markdown(href, unsafe_allow_html=True)

    
if __name__ == "__main__":
    main()