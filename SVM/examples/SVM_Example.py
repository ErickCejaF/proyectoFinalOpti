# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# %%
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df.head(4)

# %%
print("Shape of DataFrame: ", cancer_df.shape)
benign = len(cancer.data[cancer.target == 1])
print("number of benign samples: ", benign)
# %%
print(cancer_df.columns)
# select the columns with names mean, error and worst
feature_mean = list(cancer_df.columns[0:10])
feature_error = list(cancer_df.columns[11:20])
feature_worst = list(cancer_df.columns[20:31])

# %%
mean_corr = cancer_df[feature_mean].corr()
# fig = plt.figure()
# plt.subplot(1, 2, 1)
fig = plt.figure(figsize=(13, 11))
g1 = sns.heatmap(mean_corr, cmap='coolwarm', vmin=0, vmax=1)
g1.set_xticklabels(g1.get_xticklabels(), rotation=70, fontsize=8)
g1.set_yticklabels(g1.get_yticklabels(), rotation=15, fontsize=8)
plt.title("Correlation Plot of Mean Features")
plt.savefig("Corr_Mean_Features.png", dpi=200)
plt.show()

# %%
sns.set(font_scale=1.2)
j = sns.jointplot(x=cancer_df['mean radius'], y=cancer_df['mean area'], kind='hex', color="#4CB391")
j.set_axis_labels('Mean Radius', 'Mean Area', fontsize=15)
# plt.savefig('Mean_Area_Radius.png', dpi=200)
# %%
j2 = sns.jointplot(x=cancer_df['worst perimeter'], y=cancer_df['worst concave points'], kind='kde', color='orange')
j2.set_axis_labels('Worst Perimeter', 'Worst Concave Points', fontsize=15)
# plt.savefig('Worst_Perimeter_ConcaveP.png', dpi=200)
# %%
cancer_df_labels = cancer_df.copy()
# cancer_df_labels.head(3)
cancer_df_labels['labels'] = cancer.target
cancer_df_labels.tail(3)

# %%

replacements = {'mean radius': 'Mean Radius', 'mean texture': 'Mean Texture',
                'mean area': 'Mean Area', 'mean compactness': 'Mean Compactness', '1': 'Benign', '0': 'Malignant'}

h = sns.pairplot(cancer_df_labels,
                 x_vars=["mean radius", "mean texture"],
                 y_vars=["mean area", "mean compactness"], hue='labels', palette="husl",
                 height=5, markers=['o', '*'],
                 plot_kws=dict(s=100, alpha=0.4))

for ix in range(2):
    for jy in range(2):
        xlabel = h.axes[ix][jy].get_xlabel()
        ylabel = h.axes[ix][jy].get_ylabel()
        if xlabel in replacements.keys():
            h.axes[ix][jy].set_xlabel(replacements[xlabel], fontsize=16)
        if ylabel in replacements.keys():
            h.axes[ix][jy].set_ylabel(replacements[ylabel], fontsize=16)

for ixx in range(len(h.fig.get_children()[-1].texts)):
    label = h.fig.get_children()[-1].texts[ixx].get_text()
    if label in replacements.keys():
        h.fig.get_children()[-1].texts[ixx].set_text(replacements[label])
        # plt.savefig('Pairplots_Area_Texture.png', dpi=300)

##################################################################################################################
##################################################################################################################
# %%
# Train and Test Data Separation
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.25,
                                                    stratify=cancer.target, random_state=30)
print("train feature shape: ", X_train.shape)
print("test feature shape: ", X_test.shape)

# %%
#  Prepare the Pipeline Steps

# Normalizing the Features with StandardScaler
# Applying Principal Component Analysis
# Use SVM to Classify Malignant and Benign Samples

# %%
#  First Check the Effect of PCA
# For PCA, First Need to Scale the Data.
scaler1 = StandardScaler()
scaler1.fit(cancer.data)
feature_scaled = scaler1.transform(cancer.data)

# Now Apply PCA

pca1 = PCA(n_components=4)
pca1.fit(feature_scaled)
feature_scaled_pca = pca1.transform(feature_scaled)
print("shape of the scaled and 'PCA'ed features: ", np.shape(feature_scaled_pca))

# %%
# Let's see the variance to see out of the
# 4 components which are contributing most

feat_var = np.var(feature_scaled_pca, axis=0)
feat_var_rat = feat_var / (np.sum(feat_var))

print("Variance Ratio of the 4 Principal Components Ananlysis: ", feat_var_rat)

# As we can see sum of the first two components are contributing almost 80% of the total variance.
# So it would be interesting to plot these compnents and see how separated the sampels are.

# %%
# print (type(cancer.target))
cancer_target_list = cancer.target.tolist()
print(type(cancer_target_list))
# print (cancer_target_list)
# print (type(yl))
feature_scaled_pca_X0 = feature_scaled_pca[:, 0]
feature_scaled_pca_X1 = feature_scaled_pca[:, 1]
feature_scaled_pca_X2 = feature_scaled_pca[:, 2]
feature_scaled_pca_X3 = feature_scaled_pca[:, 3]

labels = cancer_target_list
colordict = {0: 'brown', 1: 'darkslategray'}
piclabel = {0: 'Malignant', 1: 'Benign'}
markers = {0: 'o', 1: '*'}
alphas = {0: 0.3, 1: 0.4}

fig = plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
for l in np.unique(labels):
    ix = np.where(labels == l)
    plt.scatter(feature_scaled_pca_X0[ix], feature_scaled_pca_X1[ix], c=colordict[l],
                label=piclabel[l], s=40, marker=markers[l], alpha=alphas[l])
plt.xlabel("First Principal Component", fontsize=15)
plt.ylabel("Second Principal Component", fontsize=15)

plt.legend(fontsize=15)

plt.subplot(1, 2, 2)
for l1 in np.unique(labels):
    ix1 = np.where(labels == l1)
    plt.scatter(feature_scaled_pca_X2[ix1], feature_scaled_pca_X3[ix1], c=colordict[l1],
                label=piclabel[l1], s=40, marker=markers[l1], alpha=alphas[l1])
plt.xlabel("Third Principal Component", fontsize=15)
plt.ylabel("Fourth Principal Component", fontsize=15)

plt.legend(fontsize=15)

# plt.savefig('Cancer_labels_PCAs.png', dpi=200)
plt.show()
# %%
# The figure above indeed shows that First and Second Principal Components play more
# significant roles to classify Maignant and Benign samples, compared to 3rd and 4th components.

# %%

#  Grid Search Cross-Validation and Best-Fit Paramters

# Pipeline Steps are StandardScaler, PCA and SVM
pipe_steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]

check_params = {
    'pca__n_components': [2],
    'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
    'SupVM__gamma': [0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
}

pipeline = Pipeline(pipe_steps)

# %%
# I love You So Much
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")

print("Start Fitting Training Data")
for cv in tqdm(range(4, 6)):
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv)
    create_grid.fit(X_train, Y_train)
    print("score for %d fold CV := %3.2f" % (cv, create_grid.score(X_test, Y_test)))
    print("!!!!!!!! Best-Fit Parameters From Training Data !!!!!!!!!!!!!!")
    print(create_grid.best_params_)

print("out of the loop")

# %%
print("grid best params: ", create_grid.best_params_)
# use the best one

# %%
# Prediction and Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

Y_pred = create_grid.predict(X_test)
# print (Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: \n")
print(cm)

df_cm = pd.DataFrame(cm, range(2), range(2))

sns.heatmap(df_cm, annot=True, cbar=False)
plt.title("Confusion Matrix", fontsize=14)
plt.savefig("Confusion Matrix.png", dpi=200)
# %%
scaler1.fit(X_test)
X_test_scaled = scaler1.transform(X_test)

pca2 = PCA(n_components=2)
X_test_scaled_reduced = pca2.fit_transform(X_test_scaled)

# svm_model = SVC(kernel='rbf', C=float(create_grid.best_params_['SupVM__C']),
#                 gamma=float(create_grid.best_params_['SupVM__gamma']))

svm_model = SVC(kernel='rbf', C=1., gamma=0.5)

classify = svm_model.fit(X_test_scaled_reduced, Y_test)


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))  # ,
    # np.arange(z_min, z_max, h))
    return xx, yy


X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('white')
cdict1 = {0: 'lime', 1: 'deeppink'}

Y_tar_list = Y_test.tolist()
yl1 = [int(target1) for target1 in Y_tar_list]
labels1 = yl1

labl1 = {0: 'Malignant', 1: 'Benign'}
marker1 = {0: '*', 1: 'd'}
alpha1 = {0: .8, 1: 0.5}

for l1 in np.unique(labels1):
    ix1 = np.where(labels1 == l1)
    ax.scatter(X0[ix1], X1[ix1], c=cdict1[l1], label=labl1[l1], s=70, marker=marker1[l1], alpha=alpha1[l1])

ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none',
           edgecolors='navy', label='Support Vectors')

plot_contours(ax, classify, xx, yy, cmap='seismic', alpha=0.4)
plt.legend(fontsize=15)

plt.xlabel("1st Principal Component", fontsize=14)
plt.ylabel("2nd Principal Component", fontsize=14)

# plt.savefig('ClassifyMalignant_Benign2D_Decs_FunctG10.png', dpi=300)
plt.show()
