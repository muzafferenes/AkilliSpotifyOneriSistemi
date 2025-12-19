import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset yükle
df = pd.read_csv('data/spotify_data.csv')

# İlk bakış
print("Dataset Boyutu:", df.shape)
print("\nİlk 5 Satır:")
print(df.head())

print("\nSütun İsimleri:")
print(df.columns.tolist())

print("\nEksik Değerler:")
print(df.isnull().sum())

print("\nTemel İstatistikler:")
print(df.describe())

# Veri tiplerine bak
print("\nVeri Tipleri:")
print(df.dtypes)