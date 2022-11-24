# %%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

from SVM2_conR import utils

COMPONENT_NUMBER = 7

sns.set_style('whitegrid')
# %%
# datos totales
total_data = utils.query_data_csv('/Users/erickcejafuentes/DataspellProjects/Optimizacion '
                                  'convexa/ProyectoFinal/SVM/SVM_Project/datasets/music_data.csv')
# %%
total_data = total_data.dropna()
# data to apply PCA
working_data = total_data.iloc[:, 1:-1]
# data description
description_data = total_data.iloc[:, :1].join(total_data.iloc[:, -1:])

# %% Estandarizar los datos media 0 y desviación estándar 1
# Normales y sin outliers
df_TF = StandardScaler().fit_transform(working_data)
pca, pca_score = utils.pca(df_TF, working_data)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

# %% Mapa de calor para visualizar in influencia de las variables
utils.print_heatmap_variable_influence(working_data, pca)
# Gráfica del aporte a cada componente principal (los primeros n)
utils.max_added(working_data, pca, COMPONENT_NUMBER)
# Obtener las primeras n variables con mayor aporte para cada componente
utils.obtain_n_number(working_data, pca, COMPONENT_NUMBER)
pca_data = utils.get_n_principal_components(description_data, df_TF, pca, total_data, 7)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# %%Graficar los primeros n componentes principales definiendo cuanta varianza pueden explicar
labels = {
    str(i): f"PC {i + 1} ({var:.1f}%)"
    for i, var in enumerate(per_var[0:COMPONENT_NUMBER])
}
fig = px.scatter_matrix(
    pca_data.iloc[:, :COMPONENT_NUMBER],
    labels=labels,
    dimensions=range(COMPONENT_NUMBER),
    color=pca_data["label"]
)
fig.update_traces(diagonal_visible=False)
fig.show()
# %%resultados
for n in range(COMPONENT_NUMBER):
    print("componente:", n + 1)
    print(pca_data.sort_values(by=n)['label'][:50].value_counts())

pca_data.to_csv(
    '/Users/erickcejafuentes/DataspellProjects/Optimizacion convexa/ProyectoFinal/SVM2_conR/outputs/components_' + str(
        COMPONENT_NUMBER) + '/out.csv')
