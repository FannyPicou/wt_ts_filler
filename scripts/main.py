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
# max_corr = corr_matrix.idxmax()
# max_corr_values = corr_matrix.max()

# # Technique 1 : linear interpolation
# estimated_dataframe = cleaned_dataframe.copy()
# # for i in range(len(cleaned_dataframe.columns)) : # on parcourt les datasets
# for i in range(3) : # on parcourt les datasets
#     print("Filling gaps in data"+str(i+1))
#     for j in range(12*30,len(cleaned_dataframe.index)) : # pour chaque dataset on parcourt les temps, en commençant au 6*30eme jour
#         if np.isnan(cleaned_dataframe.iloc[j,i]) : # si la valeur au temps j est nan :
#             jour_lim = max([0,j-12*30])
#             corr_matrix = cleaned_dataframe.iloc[jour_lim:j,:].corr() # on calcule la table de corrélation sur les données des 6 mois précedant j
#             col_corr_matrix = corr_matrix.iloc[:,i] # on regarde la col de la matrice de correlation correspondante à data_i
#             if col_corr_matrix.isna().sum() == len(col_corr_matrix.index):
#                 estimated_dataframe.iloc[j, i] = np.nan
#             else :
#                 col_corr_matrix = col_corr_matrix.dropna(axis=0) # on supprime les rows with nan values
#                 Nb_datasets_corr = len(col_corr_matrix.index)
#                 # print(col_corr_matrix)
#                 col_max_corr = col_corr_matrix.idxmax() # On cherche le dataset le plus corrélé
#                 n=0
#                 while np.isnan(cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]) and n<Nb_datasets_corr-1: # tant que la valeur au temps j du dataset le + corrélé est nan, et que n<31
#                     # print(int(col_max_corr[4:]))
#                     # print(n)
#                     col_corr_matrix = col_corr_matrix.drop(labels=[col_max_corr]) # on supprime la ligne de la colonne de correlation
#                     col_max_corr = col_corr_matrix.idxmax() # on recherche le dataset le plus corrélé
#                     n = n+1
#                 print('dataset le + corrélé et dont la valeur est dispo = '+ col_max_corr)
#                 if col_corr_matrix.max() >= 0.85:
#                     # print('Pcoeff >= 0.75')
#                     x = cleaned_dataframe[col_max_corr]
#                     y = cleaned_dataframe.iloc[:, i]
#                     mask = ~np.isnan(x) & ~np.isnan(y)
#                     slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
#                     y_value_pred = slope * x.iloc[j] + intercept
#                     estimated_dataframe.iloc[j,i]=y_value_pred
#                 else:
#                     # print('Pcoeff insuffisant')
#                     estimated_dataframe.iloc[j, i] = np.nan

# # Technique 2 : apply same variation
#
# estimated_dataframe = cleaned_dataframe.copy()
#
# # Step 1 : Interpolate for gaps inf or equal to N days
# N = 5
# print("Interpolate for gaps inf or equal to "+str(N)+' days')
# estimated_df_interpolated = estimated_dataframe.interpolate()
# for c in estimated_dataframe:
#     mask = estimated_dataframe[c].isna()
#     x = (mask.groupby((mask != mask.shift()).cumsum()).transform(lambda x: len(x) > N)* mask)
#     estimated_df_interpolated[c] = estimated_df_interpolated.loc[~x, c]
# estimated_dataframe = estimated_df_interpolated
#
# # Step 2 : Search the more correlated and apply same variation FORWARD
# print("Forward estimations")
# estimated_dataframe_forward = estimated_dataframe.copy()
# for i in range(len(cleaned_dataframe.columns)) : # on parcourt les datasets
#     for j in range(1,len(estimated_dataframe_forward.index)) : # on parcourt les dates
#         if ~np.isnan(estimated_dataframe_forward.iloc[j-1,i]) and np.isnan(estimated_dataframe_forward.iloc[j,i]) : # si value = Nan and prec_value isnot Nan
#             col_corr_matrix = corr_matrix.iloc[:,i]  # on regarde la col de la matrice de correlation correspondante à data_i
#             col_corr_matrix = col_corr_matrix.dropna(axis=0)  # on supprime les rows with nan values
#             Nb_datasets_corr = len(col_corr_matrix.index)
#             col_max_corr = col_corr_matrix.idxmax() # On cherche le dataset le plus corrélé
#             n=0
#             while np.isnan(cleaned_dataframe.iloc[j-1,int(col_max_corr[4:])-1]) and np.isnan(cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]) and n<Nb_datasets_corr-1: # tant que la valeur aux temps j-1 et j du dataset le + corrélé est nan, et que n<31
#                 col_corr_matrix = col_corr_matrix.drop(labels=[col_max_corr]) # on supprime la ligne de la colonne de correlation
#                 col_max_corr = col_corr_matrix.idxmax() # on recherche le dataset le plus corrélé
#                 n = n+1
#             # print('dataset le + corrélé et dont la valeur est dispo = '+ col_max_corr)
#             if col_corr_matrix.max() >= 0.75:
#                 # print('Pcoeff >= 0.75')
#                 y_value_pred = estimated_dataframe_forward.iloc[j-1,i] + (cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]-cleaned_dataframe.iloc[j-1,int(col_max_corr[4:])-1])
#                 estimated_dataframe_forward.iloc[j,i]=y_value_pred
#             else:
#                 # print('Pcoeff insuffisant')
#                 estimated_dataframe_forward.iloc[j, i] = np.nan
#
# # Step 3 : Search the more correlated and apply same variation BACKWARD
# print("Backward estimations")
# estimated_dataframe_backward = estimated_dataframe.copy()
# for i in range(len(cleaned_dataframe.columns)) : # on parcourt les datasets
#     for j in range(len(estimated_dataframe_backward.index)-2,1,-1) : # on parcourt les dates à l'envers
#         if ~np.isnan(estimated_dataframe_backward.iloc[j+1,i]) and np.isnan(estimated_dataframe_backward.iloc[j,i]) : # si value = Nan and prec_value isnot Nan
#             col_corr_matrix = corr_matrix.iloc[:,i]  # on regarde la col de la matrice de correlation correspondante à data_i
#             col_corr_matrix = col_corr_matrix.dropna(axis=0)  # on supprime les rows with nan values
#             Nb_datasets_corr = len(col_corr_matrix.index)
#             col_max_corr = col_corr_matrix.idxmax() # On cherche le dataset le plus corrélé
#             n=0
#             while np.isnan(cleaned_dataframe.iloc[j+1,int(col_max_corr[4:])-1]) and np.isnan(cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]) and n<Nb_datasets_corr-1: # tant que la valeur aux temps j-1 et j du dataset le + corrélé est nan, et que n<31
#                 col_corr_matrix = col_corr_matrix.drop(labels=[col_max_corr]) # on supprime la ligne de la colonne de correlation
#                 col_max_corr = col_corr_matrix.idxmax() # on recherche le dataset le plus corrélé
#                 n = n+1
#             # print('dataset le + corrélé et dont la valeur est dispo = '+ col_max_corr)
#             if col_corr_matrix.max() >= 0.75:
#                 # print('Pcoeff >= 0.75')
#                 y_value_pred = estimated_dataframe_backward.iloc[j+1,i] + (cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]-cleaned_dataframe.iloc[j+1,int(col_max_corr[4:])-1])
#                 estimated_dataframe_backward.iloc[j,i]=y_value_pred
#             else:
#                 # print('Pcoeff insuffisant')
#                 estimated_dataframe_backward.iloc[j, i] = np.nan
#
# # Step 4 : Interpolation between forward and backward
# print("Interpolation between forward and backward estimations")
# for i in range(len(estimated_dataframe.columns)) : # on parcourt les datasets
#     for j in range(len(estimated_dataframe.index)) : # on parcourt les dates
#         if ~np.isnan(estimated_dataframe_forward.iloc[j, i]) and np.isnan(estimated_dataframe_backward.iloc[j, i]):
#             estimated_dataframe.iloc[j, i] = estimated_dataframe_forward.iloc[j, i]
#         if np.isnan(estimated_dataframe_forward.iloc[j, i]) and ~np.isnan(estimated_dataframe_backward.iloc[j, i]):
#             estimated_dataframe.iloc[j, i] = estimated_dataframe_backward.iloc[j, i]
# print("Compute estimation length")
# df_predict_lengths = estimated_dataframe.copy()
# for i in range(len(df_predict_lengths.columns)) : # on parcourt les datasets
#     j=0
#     while j < len(df_predict_lengths.index)-1 : # on parcourt les dates
#         if ~np.isnan(df_predict_lengths.iloc[j,i]):
#             df_predict_lengths.iloc[j, i] = np.nan
#             j=j+1
#         else:
#             L = 0
#             k = j
#             while np.isnan(df_predict_lengths.iloc[k,i]) and k <len(df_predict_lengths.index)-1:
#                 k=k+1
#                 L=L+1
#             df_predict_lengths.iloc[j:k, i] = L
#             j=k
# print("Compute interpolation")
# for i in range(len(estimated_dataframe.columns)) :
#     for j in range(len(estimated_dataframe.index)):
#         if np.isnan(estimated_dataframe.iloc[j, i]) and ~np.isnan(df_predict_lengths.iloc[j, i]) and ~np.isnan(estimated_dataframe_forward.iloc[j, i]) and ~np.isnan(estimated_dataframe_backward.iloc[j, i]) :
#             L = int(df_predict_lengths.iloc[j, i])
#             print([(estimated_dataframe_backward.iloc[j, i]*l + estimated_dataframe_forward.iloc[j, i]+(L-l))/L for l in range(L)])
#             print(estimated_dataframe.iloc[j:j+L, i])
#             estimated_dataframe.iloc[j:j+L, i] = [(estimated_dataframe_backward.iloc[j+l, i]*l + estimated_dataframe_forward.iloc[j+l, i]*(L-l))/L for l in range(int(L))]

# Technique 3 : apply linear regression + correction epsilon + interpolation

estimated_dataframe = cleaned_dataframe.copy()

# Step 1 : Interpolate for gaps inf or equal to N days
N = 5
print("Interpolate for gaps inf or equal to "+str(N)+' days')
estimated_df_interpolated = estimated_dataframe.interpolate()
for c in estimated_dataframe:
    mask = estimated_dataframe[c].isna()
    x = (mask.groupby((mask != mask.shift()).cumsum()).transform(lambda x: len(x) > N)* mask)
    estimated_df_interpolated[c] = estimated_df_interpolated.loc[~x, c]
estimated_dataframe = estimated_df_interpolated

# Step 2 : Search the more correlated and apply linear regression FORWARD
print("Forward estimations")
estimated_dataframe_forward = estimated_dataframe.copy()
for i in range(len(cleaned_dataframe.columns)) : # on parcourt les datasets
    for j in range(1,len(estimated_dataframe_forward.index)) : # on parcourt les dates
        if ~np.isnan(estimated_dataframe_forward.iloc[j-1,i]) and np.isnan(estimated_dataframe_forward.iloc[j,i]) : # si value = Nan and prec_value isnot Nan
            col_corr_matrix = corr_matrix.iloc[:,i]  # on regarde la col de la matrice de correlation correspondante à data_i
            col_corr_matrix = col_corr_matrix.dropna(axis=0)  # on supprime les rows with nan values
            Nb_datasets_corr = len(col_corr_matrix.index)
            col_max_corr = col_corr_matrix.idxmax() # On cherche le dataset le plus corrélé
            n=0
            while np.isnan(cleaned_dataframe.iloc[j-1,int(col_max_corr[4:])-1]) and np.isnan(cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]) and n<Nb_datasets_corr-1: # tant que la valeur aux temps j-1 et j du dataset le + corrélé est nan, et que n<31
                col_corr_matrix = col_corr_matrix.drop(labels=[col_max_corr]) # on supprime la ligne de la colonne de correlation
                col_max_corr = col_corr_matrix.idxmax() # on recherche le dataset le plus corrélé
                n = n+1
            # print('dataset le + corrélé et dont la valeur est dispo = '+ col_max_corr)
            if col_corr_matrix.max() >= 0.75:
                # print('Pcoeff >= 0.75')
                x = cleaned_dataframe[col_max_corr]
                y = cleaned_dataframe.iloc[:, i]
                mask = ~np.isnan(x) & ~np.isnan(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                y_value_pred = slope * x.iloc[j] + intercept
                print(y_value_pred)
                # PB ICI : quand on passe à j suivant, calcul de epsilon décalé. Epsilon doit rester identique tant que valeur suivante = nan
                y_value_prec_pred = slope * x.iloc[j-1] + intercept
                epsilon = cleaned_dataframe.iloc[j-1, i] - y_value_prec_pred
                print(epsilon)
                estimated_dataframe_forward.iloc[j,i]=y_value_pred + epsilon
            else:
                # print('Pcoeff insuffisant')
                estimated_dataframe_forward.iloc[j, i] = np.nan

# # Step 3 : Search the more correlated and apply same variation BACKWARD
# print("Backward estimations")
# estimated_dataframe_backward = estimated_dataframe.copy()
# for i in range(len(cleaned_dataframe.columns)) : # on parcourt les datasets
#     for j in range(len(estimated_dataframe_backward.index)-2,1,-1) : # on parcourt les dates à l'envers
#         if ~np.isnan(estimated_dataframe_backward.iloc[j+1,i]) and np.isnan(estimated_dataframe_backward.iloc[j,i]) : # si value = Nan and prec_value isnot Nan
#             col_corr_matrix = corr_matrix.iloc[:,i]  # on regarde la col de la matrice de correlation correspondante à data_i
#             col_corr_matrix = col_corr_matrix.dropna(axis=0)  # on supprime les rows with nan values
#             Nb_datasets_corr = len(col_corr_matrix.index)
#             col_max_corr = col_corr_matrix.idxmax() # On cherche le dataset le plus corrélé
#             n=0
#             while np.isnan(cleaned_dataframe.iloc[j+1,int(col_max_corr[4:])-1]) and np.isnan(cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]) and n<Nb_datasets_corr-1: # tant que la valeur aux temps j-1 et j du dataset le + corrélé est nan, et que n<31
#                 col_corr_matrix = col_corr_matrix.drop(labels=[col_max_corr]) # on supprime la ligne de la colonne de correlation
#                 col_max_corr = col_corr_matrix.idxmax() # on recherche le dataset le plus corrélé
#                 n = n+1
#             # print('dataset le + corrélé et dont la valeur est dispo = '+ col_max_corr)
#             if col_corr_matrix.max() >= 0.75:
#                 # print('Pcoeff >= 0.75')
#                 y_value_pred = estimated_dataframe_backward.iloc[j+1,i] + (cleaned_dataframe.iloc[j,int(col_max_corr[4:])-1]-cleaned_dataframe.iloc[j+1,int(col_max_corr[4:])-1])
#                 estimated_dataframe_backward.iloc[j,i]=y_value_pred
#             else:
#                 # print('Pcoeff insuffisant')
#                 estimated_dataframe_backward.iloc[j, i] = np.nan
#
# # Step 4 : Interpolation between forward and backward
# print("Interpolation between forward and backward estimations")
# for i in range(len(estimated_dataframe.columns)) : # on parcourt les datasets
#     for j in range(len(estimated_dataframe.index)) : # on parcourt les dates
#         if ~np.isnan(estimated_dataframe_forward.iloc[j, i]) and np.isnan(estimated_dataframe_backward.iloc[j, i]):
#             estimated_dataframe.iloc[j, i] = estimated_dataframe_forward.iloc[j, i]
#         if np.isnan(estimated_dataframe_forward.iloc[j, i]) and ~np.isnan(estimated_dataframe_backward.iloc[j, i]):
#             estimated_dataframe.iloc[j, i] = estimated_dataframe_backward.iloc[j, i]
# print("Compute estimation length")
# df_predict_lengths = estimated_dataframe.copy()
# for i in range(len(df_predict_lengths.columns)) : # on parcourt les datasets
#     j=0
#     while j < len(df_predict_lengths.index)-1 : # on parcourt les dates
#         if ~np.isnan(df_predict_lengths.iloc[j,i]):
#             df_predict_lengths.iloc[j, i] = np.nan
#             j=j+1
#         else:
#             L = 0
#             k = j
#             while np.isnan(df_predict_lengths.iloc[k,i]) and k <len(df_predict_lengths.index)-1:
#                 k=k+1
#                 L=L+1
#             df_predict_lengths.iloc[j:k, i] = L
#             j=k
# print("Compute interpolation")
# for i in range(len(estimated_dataframe.columns)) :
#     for j in range(len(estimated_dataframe.index)):
#         if np.isnan(estimated_dataframe.iloc[j, i]) and ~np.isnan(df_predict_lengths.iloc[j, i]) and ~np.isnan(estimated_dataframe_forward.iloc[j, i]) and ~np.isnan(estimated_dataframe_backward.iloc[j, i]) :
#             L = int(df_predict_lengths.iloc[j, i])
#             print([(estimated_dataframe_backward.iloc[j, i]*l + estimated_dataframe_forward.iloc[j, i]+(L-l))/L for l in range(L)])
#             print(estimated_dataframe.iloc[j:j+L, i])
#             estimated_dataframe.iloc[j:j+L, i] = [(estimated_dataframe_backward.iloc[j+l, i]*l + estimated_dataframe_forward.iloc[j+l, i]*(L-l))/L for l in range(int(L))]

# for i in estimated_dataframe.columns:
#     plt.plot(estimated_dataframe.index,estimated_dataframe[i],lw=0,marker='.',label=i)
# plt.ylabel('Estimated groundwater level (mNGF)')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.show()

for i in estimated_dataframe.columns:
    plt.plot(estimated_dataframe_forward.index, estimated_dataframe_forward[i], lw=0, marker='.', label=i, color='darkorange')
    # plt.plot(estimated_dataframe_backward.index, estimated_dataframe_backward[i], lw=0, marker='.', label=i,color='orchid')
    # plt.plot(estimated_dataframe.index, estimated_dataframe[i], lw=0, marker='.', label=i, color='red')
    plt.plot(cleaned_dataframe.index, cleaned_dataframe[i], lw=0, marker='.', label=i, color='green')
plt.ylabel('Cleanec groundwater level (mNGF)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

