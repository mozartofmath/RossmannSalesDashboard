import streamlit as st
import pandas as pd
import numpy as np
#from database_ops import *
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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
        df_store = df[df['Store'] == store_id].sample(300)

        st.subheader('Some Statistics on the data')
        st.table(df_store.describe())

        st.subheader('Sales Figures')
        hist_values = np.histogram(df_store['Sales'], bins=100)[0]
        st.bar_chart(hist_values)

        st.subheader('Promotions')
        hist_values = np.histogram(df_store['Promo'], bins=2)[0]
        st.bar_chart(hist_values)

        st.subheader('Store Type')
        hist_values = np.histogram(df_store['StoreType'], bins=4)[0]
        st.bar_chart(hist_values)

        st.subheader('Assortment Level of Stores')
        hist_values = np.histogram(df_store['Assortment'], bins=3)[0]
        st.bar_chart(hist_values)

        store_sales = df_store[['Sales']].values

        scaler = StandardScaler()
        y = scaler.fit_transform(store_sales)

        X = df_store.drop('Sales', axis = 1).values

        score = model.score(X, y)
        mse = mean_squared_error(y, model.predict(X))
        st.subheader(f'Model performance on the sales data for store: {store_id}')

        st.write(f'Model r2 Score: {score}')
        st.write(f'Model mean squared error: {mse}')

    elif app_mode == 'Predict Sales':

        st.subheader('Upload the data you want to perform predictions on')

        csv_file =  st.file_uploader("Upload Data", type = ['csv'])
        if csv_file is not None:
            data = pd.read_csv(csv_file, na_values=['?', None, 'undefined'])

            data = data.reindex(index=data.index[::-1])
            st.subheader('First 5 rows')
            st.table(data.head(5))

            st.subheader('Last 5 rows')
            st.table(data.tail(5))

            st.subheader('Some Statistics on the data')
            st.table(data.describe())

            model_file = 'forest-2021-07-30-05-45-41-547423.sav'
            model = pickle.load(open(model_file, 'rb'))
            
            store_id = 16
            x = data[data['Store'] == store_id].values
            predicted_sales = model.predict(x)

            st.subheader('Sales Plot')
            hist_values = np.histogram(predicted_sales, bins=50)[0]
            st.bar_chart(hist_values)
    
if __name__ == "__main__":
    main()