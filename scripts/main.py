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

from scipy import stats
corr_matrix = cleaned_dataframe.corr()
# print(corr_matrix)
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
# sn.heatmap(corr_matrix, annot=True)
# plt.show()
corr_matrix = corr_matrix.replace(1, 0)
max_corr = corr_matrix.idxmax()
max_corr_values = corr_matrix.max()
data_series_estimates = []

estimated_dataframe = cleaned_dataframe
# for i in range(len(cleaned_dataframe.columns)) : # on parcourt les datasets
for i in range(3) : # on parcourt les datasets
    print("Filling gaps in data"+str(i+1))
    for j in range(len(cleaned_dataframe.index)) : # pour chaque dataset on parcourt les temps
        if np.isnan(cleaned_dataframe.iloc[j,i]) : # si la valeur au temps j est nan :
            col_corr_matrix = corr_matrix.iloc[:,i] # on regarde la col de la matrice de correlation correspondante à data_i
            col_corr_matrix = col_corr_matrix.dropna(axis=0) # on supprime les rows with nan values
            Nb_datasets_corr = len(col_corr_matrix.index)
            # print(col_corr_matrix)
            col_max_corr = col_corr_matrix.idxmax() # On cherche le dataset le plus corrélé
            n=0
            while np.isnan(cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]) and n<Nb_datasets_corr-1: # tant que la valeur au temps j du dataset le + corrélé est nan, et que n<31
                # print(int(col_max_corr[4:]))
                # print(n)
                col_corr_matrix = col_corr_matrix.drop(labels=[col_max_corr]) # on supprime la ligne de la colonne de correlation
                col_max_corr = col_corr_matrix.idxmax() # on recherche le dataset le plus corrélé
                n = n+1
            # print('dataset le + corrélé et dont la valeur est dispo = '+ col_max_corr)
            if col_corr_matrix.max() >= 0.75:
                # print('Pcoeff >= 0.75')
                x = cleaned_dataframe.iloc[:, i]
                y = cleaned_dataframe[col_max_corr]
                mask = ~np.isnan(x) & ~np.isnan(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                y_value_pred = slope * x.iloc[j] + intercept
                estimated_dataframe.iloc[j,i]=y_value_pred
            else:
                # print('Pcoeff insuffisant')
                estimated_dataframe.iloc[j, i] = np.nan

# Pb : estimated values = nan, everytime

for i in estimated_dataframe.columns:
    plt.plot(estimated_dataframe.index,estimated_dataframe[i],lw=0,marker='.',label=i, color ='red')
    plt.plot(cleaned_dataframe.index, cleaned_dataframe[i], lw=0, marker='.', label=i, color='green')
plt.ylabel('Estimated groundwater level (mNGF)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

