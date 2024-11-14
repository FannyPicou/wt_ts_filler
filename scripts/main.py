from tscleaner.cleaning import SpikeCleaner, FlatPeriodCleaner
from tscleaner.plotting import plot_timeseries

import pandas as pd

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

for data in data_series :
    data_original = data.copy()
    for cleaner in cleaners:
        data = cleaner.clean(data)
    plot_timeseries(data_original, data)
