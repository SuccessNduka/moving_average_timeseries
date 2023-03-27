import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from .Config import IOConfig

# create a class for performing EDA
class EDA(IOConfig):


    def __init__(self, series):


        self.series = series

        print(self.series.head())

        #### Number of Observations
        print(self.series.shape)

        ####  Querying By Time
        print(self.series["2017-08-30"])

        #### Descriptive Statistic
        print(self.series.describe())

    def RunEDA(self):

        #### Feature Engineering + EDA
        #### Feature engineering is to provide strong and ideally simple relationships between new input features and the output feature for the supervised learning algorithm to model.
        dataframe1 = self.date_refactor()

        print(dataframe1.columns)
        for col in dataframe1.columns[:-1]: # Visualizing the distribution of all the columns
            sns.countplot(dataframe1[col])
            plt.title(str(col).upper()+" Distribution")
            plt.xticks(rotation=45)
            plt.savefig(f"{self.output_folder}+{col}.png")
            # plt.show()  #use this in notebook files
            plt.close()

        ## making the data one dimensional for GROUPER
        data = pd.DataFrame(columns=["Sensor_Value"],data=self.series["Sensor_Value"])
        print(data.head(3))


        # Getting the months distribution of sensory data
        groups = data.groupby(pd.Grouper(freq='M'))
        weeks = pd.DataFrame()
        for name, group in groups:
            print(name.month, "-",group.Sensor_Value.nunique())
            weeks[name.month] = list(group.Sensor_Value)[:53] # The above print shows the number of data points for each months differs, so standarding it to 53 data points (lowest from all months considered)

        weeks.plot(subplots=True,legend=True, figsize=(25, 25) )
        plt.savefig(f"{self.output_folder}+Each Month Distribution.png")
        # plt.show() #use this in notebook files
        plt.close()

        # Histogram Plot For Distribution
        self.series.hist(figsize=(20, 10))
        plt.savefig(f"{self.output_folder}+Histogram of all columns.png")
        # plt.show() #use this in notebook files
        plt.close()

        # Density Plots
        self.series.plot(kind='kde', figsize=(20, 10))
        plt.savefig(f"{self.output_folder}+Density Plots.png")
        # plt.show() #use this in notebook files
        plt.close()

        pd.plotting.lag_plot(data)
        plt.savefig(f"{self.output_folder}+Lag Plot.png")
        # plt.show() #use this in notebook files
        plt.close()

        #### QQPlot
        sm.qqplot(data["Sensor_Value"])
        plt.savefig(f"{self.output_folder}+QQPlot.png")
        # plt.show() #use this in notebook files
        plt.close()

        #### checking for missing values in predictor variable ( Sensor_Value)
        print(data["Sensor_Value"].isna().sum()) #there are no missing values

        print("This completes EDA")

    def date_refactor(self):
        dataframe1 = pd.DataFrame()
        dataframe1['month'] = [self.series.index[i].month for i in
                               range(len(self.series))]  # Extracting month attribute from TimeStamp
        dataframe1['day'] = [self.series.index[i].day for i in
                             range(len(self.series))]  # Extracting day attribute from TimeStamp
        dataframe1['hour'] = [self.series.index[i].hour for i in
                              range(len(self.series))]  # Extracting hour attribute from TimeStamp
        dataframe1["sensoryVale"] = [self.series.Sensor_Value[i] for i in range(len(self.series))]
        print("Years present: \t", set([self.series.index[i].year for i in range(len(self.series))]))
        # Since there is only one year of data, we  would discard the year
        print(dataframe1.head())
        return dataframe1