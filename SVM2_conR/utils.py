import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def query_data_csv(route):
    total_data = pd.read_csv(route)
    total_data = total_data.dropna()
    return total_data


def query_data_xlsx(route):
    total_data = pd.read_excel(route)
    total_data = total_data.dropna()
    return total_data


def pca(df_TF, data):
    pca = PCA()
    pca.fit(df_TF)
    pca_score = pd.DataFrame(data=pca.components_, columns=data.columns)
    return pca, pca_score


def print_heatmap_variable_influence(data, pca):
    plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
    componentes = pca.components_
    plt.imshow(componentes.T, cmap='plasma', aspect='auto')
    plt.yticks(range(len(data.columns)), data.columns)
    plt.xticks(range(len(data.columns)), np.arange(pca.n_components_) + 1)
    plt.grid(False)
    plt.colorbar()
    plt.show()


def max_added(data, pca, pca_range):
    matrix_transform = pca.components_.T
    for n in range(pca_range):
        plt.bar(np.arange(len(pca.components_)), matrix_transform[:, n])
        plt.xticks(range(len(data.columns)), data.columns, rotation=90)
        plt.ylabel('Loading Score')
        plt.show()


def obtain_n_number(data, pca, n):
    top_n_var_arr = []
    for component in pca.components_:
        loading_scores = pd.DataFrame(component)
        # Nombre de las columnas
        loading_scores.index = data.columns
        # Ordena de mayor a menor los pesos
        sorted_loading_scores = loading_scores[0].abs().sort_values(ascending=False)
        # Selección de las 10 variables que más aportan a cada componente principal
        top_n_variables = sorted_loading_scores[0:n].index.values
        top_n_var_arr.append(top_n_variables)
    top_5_var_arr = pd.DataFrame(top_n_var_arr).transpose()
    print(top_5_var_arr)
    # Porcentaje de varianza explicada por cada componente principal proporciona
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # Porcentaje de varianza acumulado de los componentes
    print(np.cumsum(per_var))


def get_n_principal_components(df_desc, df_TF, pca, total_data, n):
    # Nuevas variables,componentes principales
    pca_data = pca.transform(df_TF)
    # escogemos sólo los primeros n componentes
    pca_data = pd.DataFrame(pca_data[:, :n], index=total_data.index)
    # agregamos las variables que describen cada componente
    pca_data = pca_data.join(df_desc)

    return pca_data
