from scipy import stats
import numpy as np


class GapsFiller:
    def __init__(self, max_gap_lin_interp, Corr_min):
        self.max_gap_lin_interp = max_gap_lin_interp
        self.Corr_min = Corr_min

    def fill(self, dataframe):
        corr_matrix = dataframe.corr()
        corr_matrix = corr_matrix.replace(1, 0)
        estimated_dataframe = dataframe.copy()

        # Step 1 : Interpolate for gaps inf or equal to N days
        print("Linear interpolation for gaps inf or equal to " + str(self.max_gap_lin_interp) + ' days.')
        estimated_df_interpolated = estimated_dataframe.interpolate()
        for c in estimated_dataframe:
            mask = estimated_dataframe[c].isna()
            x = (mask.groupby((mask != mask.shift()).cumsum()).transform(
                lambda x: len(x) > self.max_gap_lin_interp) * mask)
            estimated_df_interpolated[c] = estimated_df_interpolated.loc[~x, c]
        estimated_dataframe = estimated_df_interpolated

        # Step 2 : Search the more correlated and apply linear regression + compute epsilon left
        # print("Apply linear regression and compute epsilon left")
        print(
            "Estimation of missing data from a data set with a correlation coefficient greater than or equal to " + str(
                self.Corr_min) + '.')
        estimated_dataframe_w_lin_reg = estimated_dataframe.copy()
        df_epsilon = estimated_dataframe.copy()
        correlation_df = estimated_dataframe.copy()
        for i in range(len(dataframe.columns)):  # on parcourt les datasets
            j = 0
            for j in range(len(estimated_dataframe_w_lin_reg.index)):  # on parcourt les dates
                if ~np.isnan(df_epsilon.iloc[j, i]):
                    df_epsilon.iloc[j, i] = np.nan
                if ~np.isnan(correlation_df.iloc[j, i]):
                    correlation_df.iloc[j, i] = np.nan
                if ~np.isnan(estimated_dataframe_w_lin_reg.iloc[j - 1, i]) and np.isnan(
                        estimated_dataframe_w_lin_reg.iloc[j, i]):  # si value = Nan and prec_value isnot Nan
                    col_corr_matrix = corr_matrix.iloc[:,
                                      i]  # on regarde la col de la matrice de correlation correspondante à data_i
                    col_corr_matrix = col_corr_matrix.dropna(axis=0)  # on supprime les rows with nan values
                    Nb_datasets_corr = len(col_corr_matrix.index)
                    col_max_corr = col_corr_matrix.idxmax()  # On cherche le dataset le plus corrélé
                    n = 0
                    while np.isnan(dataframe.iloc[j - 1, int(col_max_corr[4:]) - 1]) and np.isnan(
                            dataframe.iloc[j, int(col_max_corr[
                                                  4:]) - 1]) and n < Nb_datasets_corr - 1:  # tant que la valeur aux temps j-1 et j du dataset le + corrélé est nan, et que n<31
                        col_corr_matrix = col_corr_matrix.drop(
                            labels=[col_max_corr])  # on supprime la ligne de la colonne de correlation
                        col_max_corr = col_corr_matrix.idxmax()  # on recherche le dataset le plus corrélé
                        n = n + 1
                    # print('dataset le + corrélé et dont la valeur est dispo = '+ col_max_corr)
                    if col_corr_matrix.max() >= self.Corr_min:
                        x = dataframe[col_max_corr]
                        y = dataframe.iloc[:, i]
                        mask = ~np.isnan(x) & ~np.isnan(y)
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                        y_value_pred = slope * x.iloc[j] + intercept
                        y_value_prec_pred = slope * x.iloc[j - 1] + intercept
                        epsilon = dataframe.iloc[j - 1, i] - y_value_prec_pred
                        df_epsilon.iloc[j, i] = epsilon
                        #                estimated_dataframe_w_lin_reg.iloc[j,i]=y_value_pred + epsilon
                        estimated_dataframe_w_lin_reg.iloc[j, i] = y_value_pred
                        correlation_df.iloc[j, i] = float(col_max_corr[4:])
                    else:
                        estimated_dataframe_w_lin_reg.iloc[j, i] = np.nan

        # Step 3 : Compute estimation length and add values where there is a change in selected correlated dataset
        # print("Compute estimation length")
        df_predict_lengths = estimated_dataframe.copy()
        for i in range(len(df_predict_lengths.columns)):  # on parcourt les datasets
            # for i in [8]:  # on parcourt les datasets
            j = 0
            while j < len(df_predict_lengths.index) - 2:  # on parcourt les dates
                if ~np.isnan(df_predict_lengths.iloc[j, i]):
                    df_predict_lengths.iloc[j, i] = np.nan
                    j = j + 1
                else:
                    L = 0
                    k = j
                    while np.isnan(df_predict_lengths.iloc[k, i]) and k < len(df_predict_lengths.index) - 1 and \
                            correlation_df.iloc[k, i] == correlation_df.iloc[k + 1, i]:
                        k = k + 1
                        L = L + 1
                    df_predict_lengths.iloc[j:k, i] = L + 1
                    j = k
                    if j < len(df_predict_lengths.index) - 2:
                        if correlation_df.iloc[j, i] != correlation_df.iloc[j + 1, i] and np.isnan(
                                correlation_df.iloc[j + 1, i]):
                            j = j + 1
                        if correlation_df.iloc[j, i] != correlation_df.iloc[j + 1, i] and ~np.isnan(
                                correlation_df.iloc[j + 1, i]):
                            df_predict_lengths.iloc[j, i] = np.nan
                            df_predict_lengths.iloc[j + 1, i] = np.nan
                            estimated_dataframe.iloc[j, i] = estimated_dataframe_w_lin_reg.iloc[j, i] + df_epsilon.iloc[
                                j - L, i]
                            estimated_dataframe.iloc[j + 1, i] = estimated_dataframe_w_lin_reg.iloc[j, i] + \
                                                                 df_epsilon.iloc[j - L, i]
                            df_predict_lengths.iloc[j - L:j, i] = [L] * L
                            j = j + 2

        # Step 4 : Search the more correlated and apply linear regression + compute epsilon left again
        # print("Apply linear regression and compute epsilon left again")
        estimated_dataframe_w_lin_reg = estimated_dataframe.copy()
        df_epsilon = estimated_dataframe.copy()
        correlation_df = estimated_dataframe.copy()
        for i in range(len(dataframe.columns)):  # on parcourt les datasets
            # for i in [8] : # on parcourt les datasets
            j = 0
            for j in range(len(estimated_dataframe_w_lin_reg.index)):  # on parcourt les dates
                if ~np.isnan(df_epsilon.iloc[j, i]):
                    df_epsilon.iloc[j, i] = np.nan
                if ~np.isnan(correlation_df.iloc[j, i]):
                    correlation_df.iloc[j, i] = np.nan
                if ~np.isnan(estimated_dataframe_w_lin_reg.iloc[j - 1, i]) and np.isnan(
                        estimated_dataframe_w_lin_reg.iloc[j, i]):  # si value = Nan and prec_value isnot Nan
                    col_corr_matrix = corr_matrix.iloc[:,
                                      i]  # on regarde la col de la matrice de correlation correspondante à data_i
                    col_corr_matrix = col_corr_matrix.dropna(axis=0)  # on supprime les rows with nan values
                    Nb_datasets_corr = len(col_corr_matrix.index)
                    col_max_corr = col_corr_matrix.idxmax()  # On cherche le dataset le plus corrélé
                    n = 0
                    while np.isnan(dataframe.iloc[j - 1, int(col_max_corr[4:]) - 1]) and np.isnan(
                            dataframe.iloc[j, int(col_max_corr[
                                                  4:]) - 1]) and n < Nb_datasets_corr - 1:  # tant que la valeur aux temps j-1 et j du dataset le + corrélé est nan, et que n<31
                        col_corr_matrix = col_corr_matrix.drop(
                            labels=[col_max_corr])  # on supprime la ligne de la colonne de correlation
                        col_max_corr = col_corr_matrix.idxmax()  # on recherche le dataset le plus corrélé
                        n = n + 1
                    if col_corr_matrix.max() >= self.Corr_min:
                        # print('Pcoeff >= 0.75')
                        x = dataframe[col_max_corr]
                        y = dataframe.iloc[:, i]
                        mask = ~np.isnan(x) & ~np.isnan(y)
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                        y_value_pred = slope * x.iloc[j] + intercept
                        y_value_prec_pred = slope * x.iloc[j - 1] + intercept
                        epsilon = estimated_dataframe.iloc[j - 1, i] - y_value_prec_pred
                        df_epsilon.iloc[j, i] = epsilon
                        #                estimated_dataframe_w_lin_reg.iloc[j,i]=y_value_pred + epsilon
                        estimated_dataframe_w_lin_reg.iloc[j, i] = y_value_pred
                        correlation_df.iloc[j, i] = float(col_max_corr[4:])
                    else:
                        estimated_dataframe_w_lin_reg.iloc[j, i] = np.nan

        # Step 3 : Search the more correlated and apply linear regression + compute epsilon right
        # print("Compute epsilon right")
        for i in range(len(dataframe.columns)):  # on parcourt les datasets
            for j in range(len(estimated_dataframe.index) - 2, 1, -1):  # on parcourt les dates à l'envers
                if ~np.isnan(estimated_dataframe.iloc[j + 1, i]) and np.isnan(
                        estimated_dataframe.iloc[j, i]):  # si value = Nan and prec_value isnot Nan
                    col_corr_matrix = corr_matrix.iloc[:,
                                      i]  # on regarde la col de la matrice de correlation correspondante à data_i
                    col_corr_matrix = col_corr_matrix.dropna(axis=0)  # on supprime les rows with nan values
                    Nb_datasets_corr = len(col_corr_matrix.index)
                    col_max_corr = col_corr_matrix.idxmax()  # On cherche le dataset le plus corrélé
                    n = 0
                    while np.isnan(dataframe.iloc[j + 1, int(col_max_corr[4:]) - 1]) and np.isnan(
                            dataframe.iloc[j, int(col_max_corr[
                                                  4:]) - 1]) and n < Nb_datasets_corr - 1:  # tant que la valeur aux temps j-1 et j du dataset le + corrélé est nan, et que n<31
                        col_corr_matrix = col_corr_matrix.drop(
                            labels=[col_max_corr])  # on supprime la ligne de la colonne de correlation
                        col_max_corr = col_corr_matrix.idxmax()  # on recherche le dataset le plus corrélé
                        n = n + 1
                    if col_corr_matrix.max() >= self.Corr_min:
                        x = dataframe[col_max_corr]
                        y = dataframe.iloc[:, i]
                        mask = ~np.isnan(x) & ~np.isnan(y)
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                        y_value_prec_pred = slope * x.iloc[j + 1] + intercept
                        epsilon = dataframe.iloc[j + 1, i] - y_value_prec_pred
                        df_epsilon.iloc[j, i] = epsilon

        # Step 4 : Apply ponderated epsilon
        # print("Compute interpolation")
        for i in range(len(estimated_dataframe.columns)):
            j = 0
            while j < len(df_predict_lengths.index) - 1:  # on parcourt les dates
                if ~np.isnan(estimated_dataframe.iloc[j, i]) or np.isnan(df_predict_lengths.iloc[j, i]) or np.isnan(
                        df_epsilon.iloc[j, i]) or np.isnan(estimated_dataframe_w_lin_reg.iloc[j, i]):
                    j = j + 1
                else:
                    k = j
                    while np.isnan(estimated_dataframe.iloc[k, i]) and ~np.isnan(
                            df_predict_lengths.iloc[k, i]) and ~np.isnan(df_epsilon.iloc[k, i]) and ~np.isnan(
                        estimated_dataframe_w_lin_reg.iloc[k, i]):
                        L = int(df_predict_lengths.iloc[k, i])
                        epsilon_left = df_epsilon.iloc[j, i]
                        if ~np.isnan(df_epsilon.iloc[j + L - 1, i]):
                            epsilon_right = df_epsilon.iloc[j + L - 1, i]
                            estimated_dataframe.iloc[j:j + L, i] = [estimated_dataframe_w_lin_reg.iloc[j + l, i] + (
                                    l * epsilon_right + (L - l) * epsilon_left) / L for l in range(L)]
                        else:
                            estimated_dataframe.iloc[j:j + L, i] = [
                                estimated_dataframe_w_lin_reg.iloc[j + l, i] + epsilon_left for l in range(L)]
                        k = k + 1
                    j = k

        return estimated_dataframe

# # Other technique : apply same variation
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
