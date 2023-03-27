
from ML_Pipeline.EDA import EDA
from ML_Pipeline.MovingAverage import MovingAverage
from ML_Pipeline.Resampling import Resampling
import pandas as pd


# Loading the dataset
df = pd.read_csv("input/Data-Chillers.csv")

# Convert the time from string to date format
df.time = pd.to_datetime(df.time, format='%d-%m-%Y %H:%M')

# Set time column as the index
df.set_index("time", inplace=True)

# Perfrom Exploratory Data Analysis(EDA)
eda = EDA(df)
eda.RunEDA()

# Perform Resampling onn data
resampling = Resampling(df)
resampling.RunResampling()

# Perform moving average smoothing
movingaverage = MovingAverage(df)
movingaverage.RunMovingAverage()

print("Completed Moving Average")