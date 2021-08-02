import pandas as pd
import numpy as np

from sklearn import preprocessing

from preprocessing_functions import weekends, time_of_month, label_holidays, days_from_holiday

#df_tes = pd.read_csv('./test.csv', na_values=['?', None, 'undefined'])

df_store = pd.read_csv('./store.csv', na_values=['?', None, 'undefined'])

class Preprocessor:
    
    def holiday(self,x):
        if x in ['a','b','c']:
            return 1
        return 0
    
    def day_month_year(self, df, col):
        try:
            df['Day'] = pd.DatetimeIndex(df[col]).day
            df['Month'] = pd.DatetimeIndex(df[col]).month
            df['Year'] = pd.DatetimeIndex(df[col]).year
        except KeyError:
            print("Unknown Column Index")
    
    
    def percent_missing(self, df):
        # how many missing values exist or better still what is the % of missing values in the dataset?
        # Calculate total number of cells in dataframe
        totalCells = np.product(df.shape)

        # Count number of missing values per column
        missingCount = df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        print("The dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")
    
    def missing_per_column(self, df):
        return df.isna().sum()
    
    def columns_info(self, df):
        return df.info()

def clean_data(df_test):
#Instantiate the Preprocessor class
    preprocessor = Preprocessor()

    df_test["Date"]=pd.to_datetime(df_test["Date"], format='%Y/%m/%d', errors='coerce')
        
    df_test['Holiday'] = df_test['StateHoliday'].apply(preprocessor.holiday)

    df_test['Holiday'] = df_test['Holiday'] | df_test['SchoolHoliday']

    # Split the Date into Day, Month, and Year columns
    preprocessor.day_month_year(df_test, 'Date')

    df_test['Weekend'] = df_test['DayOfWeek'].apply(weekends)

    df_test['TimeOfMonth'] = df_test['Day'].apply(time_of_month)

    df_test['Holiday'] = df_test['StateHoliday'].apply(label_holidays)

    df_test['Date'] = pd.DatetimeIndex(df_test['Date'])

    df_test['Open'] = df_test['Open'].fillna(1)

    df_weekends = df_test[['Store', 'DayOfWeek','Open']]
    df_weekends = df_weekends[df_weekends['Open'] == 1]
    weekend_stores = df_weekends[['Store', 'DayOfWeek']].groupby('Store').nunique()
    weekend_stores = weekend_stores[weekend_stores['DayOfWeek'] == 7].reset_index()

    df_weekends = df_weekends[df_weekends['Store'].isin(set(weekend_stores['Store']))]
    weekendstores = set(df_weekends['Store'])

    def isallweekstore(x):
        if x in weekendstores:
            return 1
        return 0

    df_test['7DayStore'] = df_test['Store'].apply(isallweekstore)

    df_store['CompetitionDistance'] = df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].max())

    df_test = df_test.merge(df_store, on='Store', how='left')

    holidays = np.array(df_test[df_test["Holiday"] > 1]["Date"].unique())
    holidays = np.sort(holidays)
    df_test["TillHday"], df_test["AfterHday"] = days_from_holiday(df_test["Date"], holidays)

    le = preprocessing.LabelEncoder()
    
    df_test['StoreType'] = le.fit_transform(df_test['StoreType'])
    df_test['Assortment'] = le.fit_transform(df_test['Assortment'])

    test = df_test[['Store', 'DayOfWeek', 'Open', 'Promo', 'Holiday', 'SchoolHoliday', 'Day',
                'Month', 'Year', 'Weekend' ,'TimeOfMonth', '7DayStore','StoreType',
                'Assortment','CompetitionDistance', 'Promo2', 'TillHday', 'AfterHday']]
    #print(test.isna().sum())
    return test

#clean_data(df_tes)