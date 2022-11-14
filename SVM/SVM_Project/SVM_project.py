# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from SVM.SVM_Project import utils

sns.set_style('whitegrid')
# %%
# datos totales
total_data = utils.query_data_xlsx('/Users/erickcejafuentes/DataspellProjects/Optimizacion '
                                   'convexa/ProyectoFinal/SVM/SVM_Project/data_source.xlsx')
# %%
total_data = total_data.dropna()
# data to apply PCA
data = total_data.iloc[:, 2:-5]
# data description
df_desc = total_data.iloc[:, :2].join(total_data.iloc[:, -5:-1])

# %% Estandarizar los datos media 0 y desviación estándar 1
# Normales y sin outliers
df_TF = StandardScaler().fit_transform(data)
pca, pca_score = utils.pca(df_TF, data)


# %% Mapa de calor para visualizar in influencia de las variables
utils.print_heatmap_variable_influence(data, pca)
# Gráfica del aporte a cada componente principal (los primeros n)
utils.max_added(data, pca, 5)
# Obtener las primeras n variables con mayor aporte para cada componente
utils.obtain_n_number(data, pca, 5)
pca_data = utils.get_n_principal_components(df_desc, df_TF, total_data, 5)

##############################################################################################################
##############################################################################################################
# %%
# Pipeline Steps are StandardScaler, PCA and SVM
pipe_steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]

check_params = {
    'pca__n_components': [2],
    'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
    'SupVM__gamma': [0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
}

pipeline = Pipeline(pipe_steps)

