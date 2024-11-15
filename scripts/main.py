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
# import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
# sn.heatmap(corr_matrix, annot=True)
# plt.show()
corr_matrix = corr_matrix.replace(1, 0)
max_corr = corr_matrix.idxmax()
max_corr_values = corr_matrix.max()
data_series_estimates = []

# for i in range(len(max_corr)):
#     print('Pearson coeff betwenn data'+str(i+1)+' and '+max_corr[i]+': '+str(max_corr_values[i]))
#     x=cleaned_dataframe.iloc[:,i]
#     y=cleaned_dataframe[max_corr[i]]
#     # plt.plot(x,y,lw=0,marker='.')
#     # plt.title(max_corr[i]+' = f(data'+str(i+1)+')')
#     # plt.grid()
#     # plt.show()
#     mask = ~np.isnan(x) & ~np.isnan(y)
#     slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
#     y_pred = pd.Series([slope * xi + intercept for xi in x], index=dataframe.iloc[:,0], name="data"+str(i+1))
#     data_series_estimates.append(y_pred)

# for i in range(len(cleaned_dataframe.columns)) :
estimated_dataframe = cleaned_dataframe
for i in range(1) : # on parcourt les datasets
    for j in range(len(cleaned_dataframe.index)) : # pour chaque dataset on parcourt les temps
        if np.isnan(cleaned_dataframe.iloc[j,i]) : # si la valeur au temps j est nan :
            col_corr_matrix = corr_matrix.iloc[:,i] # on regarde la col de la matrice de correlation correspondante à data_i
            col_max_corr = col_corr_matrix.idxmax() # On cherche le dataset le plus corrélé
            n=0
            while np.isnan(cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]) and n<len(cleaned_dataframe.columns)-1: # tant que la valeur au temps j du dataset le + corrélé est nan
                print(int(col_max_corr[4:]))
                col_corr_matrix = col_corr_matrix.drop(labels=[col_max_corr]) # on supprime la ligne de la colonne de correlation
                col_max_corr = col_corr_matrix.idxmax() # on recalcule le dataset le plus corrélé
                n = n+1
            print('dataset le + corrélé et dont la valeur est dispo = '+ col_max_corr)
            if col_corr_matrix.max() >= 0.75:
                print('Pcoeff >= 0.75')
                x = cleaned_dataframe.iloc[:, i]
                y = cleaned_dataframe[col_max_corr]
                mask = ~np.isnan(x) & ~np.isnan(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                y_value_pred = slope * x[j] + intercept
                estimated_dataframe.iloc[j,i]=y_value_pred
            else:
                print('Pcoeff insuffisant')
                estimated_dataframe.iloc[j, i] = np.nan

a=0


            # if ~np.isnan(cleaned_dataframe.iloc[j,col_max_corr[-1]-1]):
            #     x = cleaned_dataframe.iloc[:, i]
            #     y = cleaned_dataframe[col_max_corr[i]]
            #     mask = ~np.isnan(x) & ~np.isnan(y)
            #     slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            #     y_value_pred = slope * x[j] + intercept
            #     estimated_dataframe.iloc[j,i]=y_value_pred

a=0
# estimated_dataframe = pd.concat(data_series_estimates, axis=1)

# for i in estimated_dataframe.columns:
#     plt.plot(estimated_dataframe.index,estimated_dataframe[i],lw=0,marker='.',label=i)
# plt.ylabel('Estimated groundwater level (mNGF)')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.show()
#
# a=0