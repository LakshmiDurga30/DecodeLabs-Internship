# ------------------------------------------------------
# PROJECT 1: ADVANCED EDA & FEATURE ENGINEERING
# ------------------------------------------------------


import pandas as pd
import numpy as np
df = pd.read_excel("Dataset for Data Analytics.xlsx")

#Basic Dataset Information
print("\nFirst 5 Rows")
print(df.head())
print("\nLast 5 Rows")
print(df.tail())
print("\nDataset Shape")
print(df.shape)
print("\nColumn Names")
print(df.columns)
print("\nDataset Information")
print(df.info())
print("\nStatistical Summary")
print(df.describe())

#Finding Missing Values
print("\nMissing Values")
print(df.isnull().sum())

#Filling Missing Values
df['CouponCode'] = df['CouponCode'].fillna('No Coupon')
print("\nMissing Values After Handling")
print(df.isnull().sum())
df['Date'] = pd.to_datetime(df['Date'])

#Identifying and neutralizing outliers using IQR method
numerical_columns = ['Quantity','UnitPrice','ItemsInCart','TotalPrice']
for col in numerical_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
print("\nDataset Shape After Removing Outliers")
print(df.shape)
# Adding Features
#  1
df['OrderMonth'] = df['Date'].dt.month
# 2
df['OrderYear'] = df['Date'].dt.year
# 3
df['PricePerItem'] = df['TotalPrice'] / df['Quantity']
# 4
df['CartValue'] = df['ItemsInCart'] * df['UnitPrice']
print("\nNew Features Added Successfully!")

# Final dataset

print("\nFinal Dataset")
print(df.head())

print("\nFinal Dataset Shape")
print(df.shape)

print("\nFinal Dataset Info")
print(df.info())

#Saving Dataset
df.to_csv("Cleaned_Ecommerce_Dataset.csv", index=False)

print("\nProject Completed Successfully!")
print("Cleaned Dataset Saved as 'Cleaned_Ecommerce_Dataset.csv'")