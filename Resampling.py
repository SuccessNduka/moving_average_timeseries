import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas import Grouper
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
from .Config import  IOConfig


# create a class for resampling the data
class Resampling(IOConfig):

    def __init__(self, series):
        self.series = series

    def RunResampling(self):

        data = pd.DataFrame(columns=["Sensor_Value"],data=self.series["Sensor_Value"])

        upsampled = data.resample('D').mean()
        print("No of Days : \t",upsampled["Sensor_Value"].isna().sum())

        # Our data is per hour basis, so apart from the hours in 256 days, lets get the number of hours missing from the dataset
        upsampled = data.resample('H').mean()
        print("No of hours missing : \t",upsampled["Sensor_Value"].isna().sum())

        # there are two kinds of resampling, they aree upsampling and downsampling
        # Since this data has missing record we would upsample and interpolate the values for those data points
        upsampled = data.resample('D').mean()
        print("Overall data points : \t",upsampled.shape)

        print(upsampled.head(5)) # Now using the resample method from pandas we are able to identify the missing days and we are able to fill it up with "NAN"

        # Lets fill the values and plot it
        interpolated = upsampled.interpolate(method='linear') # we have used linear which draws a straight line between available data
        print(interpolated.head(5))
        interpolated.plot()
        plt.savefig(f"{self.output_folder}+Interploted Plot (linear).png")
        # plt.show() #use this in notebook files
        plt.close()

        interpolated = upsampled.interpolate(method="spline", order=1) # we have used a polynomial function with order 1
        print(interpolated.head(5))
        interpolated.plot()
        plt.savefig(f"{self.output_folder}+Interploted Plot (spline).png")
        # plt.show() #use this in notebook files
        plt.close()

        #### Before we do our transformations, we will look at the data one more time through line and histogram
        plt.figure(1)
        # line plot
        plt.subplot(211)
        plt.plot(data)

        # histogram
        plt.subplot(212)
        plt.hist(data["Sensor_Value"])
        # plt.plot(data)
        plt.savefig(self.output_folder + "/Before transformations+.png")
        plt.close()

        #### Square Root Transform
        '''It is possible that our dataset shows a quadratic growth. In that case, then we could expect a square root transform to reduce the growth trend to be linear
        and change the distribution of observations to nearly Gaussian type'''

        transform = np.sqrt(data)
        plt.figure(1)
        # line plot
        plt.subplot(211)
        plt.plot(transform)
        # histogram
        plt.subplot(212)
        plt.hist(data["Sensor_Value"])
        plt.savefig(self.output_folder + "/Square Root Transformation Histogram+Line.png")
        plt.close()

        #### Log Transform
        '''
        This method is popular among time series data as they are effective at removing exponential variance. It  assumes values are positive and non-zero.
        It is common to transform observations by adding a fixed constant to ensure all input values meet this requirement.
                transform = log(constant + x)
        '''
        data1 = data.copy()

        data1['log_Sensor_Value'] = np.log(data1['Sensor_Value'])
        plt.figure(1)
        # line plot
        plt.subplot(211)
        plt.plot(data1['Sensor_Value'])
        # histogram
        plt.subplot(212)
        plt.hist(data1['Sensor_Value'])
        plt.savefig(self.output_folder + "/Log Transformation Histogram+Line.png")
        plt.close()

        # BOX COX Transform
        '''
        The square root transform and log transform belong to a class of transforms called power transforms. 
        The Box-Cox transform2 is a configurable data transform method that supports both square root and log transform, as well as a suite of related transforms.
        Box-Cox transform only takes predictor variable which is positive integers.
        Since our data has negative value we would scale our data and apply it to BOX-COX Transform
        '''
        data2 = data.copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit([data2["Sensor_Value"].tolist()])
        data2["Scaled_Sensor_Value"] = scaler.transform([data2["Sensor_Value"].tolist()])[0]
        set(scaler.transform([data2["Sensor_Value"].tolist()]).tolist()[0]) # from this we can infer that if we scale our data then we will get 0.0 for all the values

        # hence we cannot perform Box-Cox Transform on this data
        ## data2['box_cox_Sensor_Value'], lam = boxcox(data2['Scaled_Sensor_Value'])
        ## pyplot.figure(1)
        ## #line plot
        ## pyplot.subplot(211)
        ## pyplot.plot(data2['box_cox_Sensor_Value'])
        ## # histogram
        ## pyplot.subplot(212)
        ## pyplot.hist(data2['box_cox_Sensor_Value'])
        ## pyplot.show()

        #### Moving Average Smoothing
        '''Centered Moving Average
        At time (t), it is calculated as the average of actual observations at, before, and after (t). 
        For example, a center moving average with a window of 3 would be calculated as:
        center_ma(t) = mean(obs(t − 1), obs(t), obs(t + 1))'''
        '''
        Trailing Moving Average
        At time (t), it is calculated as the average of  actual observations at and before the time (t). 
        For example, a trailing moving average with a window of 3 would be calculated as:
        trail_ma(t) = mean(obs(t − 2), obs(t − 1), obs(t))
        '''
        '''
        Data Expectations
        Calculating a moving average of a time series takes some assumptions on data. 
        It is assumed that both trend and seasonal components have been removed from your data.
        This means that your time series is stationary, or does not show obvious trends (long-term increasing or decreasing movement) or seasonality (consistent periodic structure).
        There are many methods to remove trends and seasonality from a time series dataset when forecasting.
        Two methods for each are to use the differencing method and to model the behavior and explicitly subtract it from the series.
        '''

        print("This completes Re-Sampling")