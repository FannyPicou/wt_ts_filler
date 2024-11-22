import matplotlib.pyplot as plt
from random import *

def plot_timeseries(data_original, data):
    plt.figure(figsize=(15, 5))
    plt.plot(data_original, ".", color="red")
    plt.plot(data, ".", color="green")
    plt.title(f"{data.name}")
    plt.ylabel('Groundwater level (mNGF)')
    plt.show()


def plot_dataframes(cleaned_dataframe,estimated_dataframe):
    for i in range(len(estimated_dataframe.columns)):
        plt.figure(figsize=(15, 5))
        plt.plot(estimated_dataframe.index, estimated_dataframe.iloc[:, i], lw=0, marker='.', color='orchid',label='Missing data estimation')
        plt.plot(cleaned_dataframe.index, cleaned_dataframe.iloc[:, i], lw=0, marker='.',
                 label='Measurement', color='darkblue')
        plt.ylabel('Groundwater level (mNGF)')
        plt.legend()
        plt.title(estimated_dataframe.columns[i])
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    
