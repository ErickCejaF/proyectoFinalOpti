#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# datos totales
total_data = pd.read_csv("nndb_flat.csv")
# datos con los que se va a trabajar
data = total_data.iloc[:, 7:]
# descripción de los datos
df_desc = total_data.iloc[:, :6]
cor = data.corr()
# Varianza de las variables
print(data.var())
# eliminar las columnas _USRDA, ya que son redundantes
data.drop(data.columns[data.columns.str.contains('_USRDA')].values,
          inplace=True, axis=1)
# %% Estandarizar los datos media 0 y desviación estándar 1
# Normales y sin outliers
df_TF = StandardScaler().fit_transform(data)
print("media: ", np.round(df_TF.mean(), 2))
print("desviacion estandar: ", np.round(df_TF.std(), 2))
# %% Algoritmo pca
pca = PCA()
pca.fit(df_TF)

# Ponderación de los componentes principales (vectores propios)
pca_score = pd.DataFrame(data=pca.components_, columns=data.columns)
# %% Mapa de calor para visualizar in influencia de las variables
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
componentes = pca.components_
plt.imshow(componentes.T, cmap='plasma', aspect='auto')
plt.yticks(range(len(data.columns)), data.columns)
plt.xticks(range(len(data.columns)), np.arange(pca.n_components_) + 1)
plt.grid(False)
plt.colorbar()
plt.show()
# %% Gráfica del aporte a cada componente principal (los primeros 5)
matrix_transform = pca.components_.T
for n in range(5):
    plt.bar(np.arange(len(pca.components_)), matrix_transform[:, n])
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.ylabel('Loading Score')
    plt.show()
# %%Obtener las primeras 5 variables con mayor aporte para cada componente
top_5_var_arr = []
for component in pca.components_:
    loading_scores = pd.DataFrame(component)
    # Nombre de las columnas
    loading_scores.index = data.columns
    # Ordena de mayor a menor los pesos
    sorted_loading_scores = loading_scores[0].abs().sort_values(ascending=False)
    # Selección de las 10 variables que más aportan a cada componente principal
    top_5_variables = sorted_loading_scores[0:5].index.values
    top_5_var_arr.append(top_5_variables)
top_5_var_arr = pd.DataFrame(top_5_var_arr).transpose()
print(top_5_var_arr)
# Porcentaje de varianza explicada por cada componente principal proporciona
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
# Porcentaje de varianza acumulado de los componentes
porcent_acum = np.cumsum(per_var)
# Nuevas variables,componentes principales
pca_data = pca.transform(df_TF)
# escogemos sólo los primeros 5 componentes
pca_data = pd.DataFrame(pca_data[:, :5], index=total_data.index)
# agregamos las variables que describen cada componente
pca_data = pca_data.join(df_desc)
# no nos sirven estos datos entonces los quitamos
pca_data.drop(['CommonName', 'MfgName'], axis=1, inplace=True)
# %%Graficar los primeros 0-5 componentes principales definiendo cuanta varianza pueden explicar
labels = {
    str(i): f"PC {i + 1} ({var:.1f}%)"
    for i, var in enumerate(per_var[0:5])
}
fig = px.scatter_matrix(
    pca_data.iloc[:, :5],
    labels=labels,
    dimensions=range(5),
    color=pca_data["FoodGroup"]
)
fig.update_traces(diagonal_visible=False)
fig.show()
# %%resultados
for n in range(5):
    print("componente:", n + 1)
    print(pca_data.sort_values(by=n)['label'][:500].value_counts())
