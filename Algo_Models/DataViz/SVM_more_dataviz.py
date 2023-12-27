from ..source_dataset import dataset
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations

import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## More Data viz


SVM_more = dataset[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", "thalach", "thal", 'heart_disease']].copy()
SVM_x_more = SVM_more.iloc[:, :10].values
SVM_y_more = SVM_more.iloc[:, 10].values

scaler = StandardScaler()
SVM_x_more_standardized = scaler.fit_transform(SVM_x_more)

def SVM_more_dataviz():
    clean_data_viz_folder()

    title_names= SVM_more.columns

    all_comb_more = list(combinations(range(SVM_x_more.shape[1]), 2))
    svm = SVC(kernel='linear', C=1.0)

    fig, axs = plt.subplots(7, 7, figsize=(25, 25))
    fig.subplots_adjust(hspace=0.25, wspace=0.4)
    axs = axs.flatten()

    for i, (feature_1, feature_2) in enumerate(all_comb_more):

        pair = SVM_x_more[:, [feature_1, feature_2]]
        X_pair_std = SVM_x_more_standardized[:, [feature_1, feature_2]]
        svm.fit(X_pair_std, SVM_y_more)

        x_min, x_max = X_pair_std[:, 0].min() - 1, X_pair_std[:, 0].max() + 1
        y_min, y_max = X_pair_std[:, 1].min() - 1, X_pair_std[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))

        Z = svm.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)

        axs[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        axs[i].scatter(X_pair_std[:, 0], X_pair_std[:, 1], c=SVM_y_more, s=30, edgecolor='k', cmap=plt.cm.coolwarm)
        axs[i].set_xlabel(title_names[feature_1], fontsize=14)
        axs[i].set_ylabel(title_names[feature_2], fontsize=14)
        axs[i].tick_params(axis='both', which='major', labelsize=12)
        x_ticks = np.arange(np.floor(X_pair_std[:, 0].min()), np.ceil(X_pair_std[:, 0].max()) + 1, step=1).astype(int)
        y_ticks = np.arange(np.floor(X_pair_std[:, 1].min()), np.ceil(X_pair_std[:, 1].max()) + 1, step=1).astype(int)
        axs[i].set_xticks(x_ticks)
        axs[i].set_yticks(y_ticks)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = 'DataViz'

    img_filename = f"svm_more_dataviz_{timestamp}.png"
    img_path_more = os.path.join(folder_path, img_filename)

    plt.savefig(img_path_more)
    plt.close()

    return img_path_more

def clean_data_viz_folder():
    folder_path = 'DataViz'
    files_to_exclude = ['dataviz-dumpfile.txt']  # Add other file names to exclude as needed

    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path) and file_name not in files_to_exclude:
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Error cleaning DataViz folder: {e}")