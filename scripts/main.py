from tscleaner.cleaning import SpikeCleaner, FlatPeriodCleaner
from tscleaner.plotting import plot_timeseries

import pandas as pd

# Import data
dataframe = pd.read_csv('C:/Users/picourlat/Documents/040724_Data_recap/DATA/Hydrologic_data/Groundwater_lvls/Analyse_data_drought/Data/wt_ts.csv')
dataframe.iloc[:,0] = pd.to_datetime(dataframe.iloc[:,0], format='%Y-%m-%d')
data_series = []
for i in range(1, len(dataframe.columns)):
    data = pd.Series(dataframe.iloc[:,i].values, index=dataframe.iloc[:,0], name="data"+str(i))
    data_series.append(data)

cleaners = [
    SpikeCleaner(max_jump=10),
    FlatPeriodCleaner(flat_period=10)
]

# Clean
for data in data_series :
    data_original = data.copy()
    for cleaner in cleaners:
        data = cleaner.clean(data)
    # plot_timeseries(data_original, data)
cleaned_dataframe = pd.concat(data_series, axis=1)

# Fill gaps
corr_matrix = cleaned_dataframe.corr()
# print(corr_matrix)
# import seaborn as sn
import matplotlib.pyplot as plt
# sn.heatmap(corr_matrix, annot=True)
# plt.show()
corr_matrix = corr_matrix.replace(1, 0)
max_corr = corr_matrix.idxmax()
for i in range(len(max_corr)):
    x=cleaned_dataframe.iloc[:,i]
    y=cleaned_dataframe[max_corr[i]]
    plt.plot(x,y,lw=0,marker='.')
    plt.title(max_corr[i]+' = f(data'+str(i+1)+')')
    plt.grid()
    plt.show()