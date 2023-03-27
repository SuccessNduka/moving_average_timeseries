import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from .Config import IOConfig

# create a class for moving average smoothing
class MovingAverage(IOConfig):

    def __init__(self, series):
        self.series = series

    def RunMovingAverage(self):

        data = pd.DataFrame(columns=["Sensor_Value"],data=self.series["Sensor_Value"])
        rolling = data.rolling(window=3)
        rolling_mean = rolling.mean()
        print(rolling_mean.head(10))

        # plot original and transformed dataset
        data.plot()
        rolling_mean.plot(color='red')
        plt.savefig(f"{self.output_folder}+Resampled.png")
        # plt.show() #use this in notebook files
        plt.close()

        # zoomed plot original and transformed dataset
        data[:100].plot()
        rolling_mean[:100].plot(color='red')
        plt.savefig(f"{self.output_folder}+Resampled (Close Look).png")
        # plt.show() #use this in notebook files
        plt.close()


        ### this show how to extract the window and time stamp from the data for Moving Average of window 3
        df = data.copy()
        width = 3
        lag1 = df.shift(1)
        lag3 = df.shift(width - 1)
        window = lag3.rolling(window=width)
        means = window.mean()
        dataframe = pd.concat([means, lag1, df], axis=1)
        dataframe.columns = ['mean', 't', 't+1']
        print(dataframe.head(10))

        # Now lets write a mathematical function to make prediction based on Moving average
        X = data.values
        window = 3
        history = [X[i] for i in range(window)]
        test = [X[i] for i in range(window, len(X))]
        predictions = list()
        # walk forward over time steps in test
        for t in range(len(test)):
            length = len(history)
            yhat = np.mean([history[i] for i in range(length-window,length)])
            obs = test[t]
            predictions.append(yhat)
            history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
        rmse = np.sqrt(mean_squared_error(test, predictions))
        print('Test RMSE: %.3f' % rmse)

        # plotting
        plt.plot(test)
        plt.plot(predictions, color='red')
        plt.savefig(f"{self.output_folder}+Moving Average.png")
        # plt.show() #use this in notebook files
        plt.close()

        # zoom plot
        plt.plot(test[:100])
        plt.plot(predictions[:100], color='red')
        plt.savefig(f"{self.output_folder}+Moving Average (Close Look).png")
        plt.close()

        print("This completes Moving Average")
