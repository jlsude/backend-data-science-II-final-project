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

## Less Data viz

SVM_less = dataset[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", 'heart_disease']].copy()

SVM_x_less = SVM_less.iloc[:, :8].values
SVM_y_less = SVM_less.iloc[:, 8].values


scaler = StandardScaler()
SVM_x_less_standardized = scaler.fit_transform(SVM_x_less)

def SVM_less_dataviz():
    clean_data_viz_folder()
    
    title_namesl= SVM_less.columns
    all_comb_less = list(combinations(range(SVM_x_less.shape[1]), 2))
    svm = SVC(kernel='linear', C=1.0)
    fig, axs = plt.subplots(7, 7, figsize=(25, 25))
    fig.subplots_adjust(hspace=0.25, wspace=0.4)
    axs = axs.flatten()

    for i, (feature_1_less, feature_2_less) in enumerate(all_comb_less):

        X_pair_stdl = SVM_x_less_standardized[:, [feature_1_less, feature_2_less]]
        svm.fit(X_pair_stdl, SVM_y_less)

        x_min, x_max = X_pair_stdl[:, 0].min() - 1, X_pair_stdl[:, 0].max() + 1
        y_min, y_max = X_pair_stdl[:, 1].min() - 1, X_pair_stdl[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))

        Z = svm.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)

        axs[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        axs[i].scatter(X_pair_stdl[:, 0], X_pair_stdl[:, 1], c=SVM_y_less, s=30, edgecolor='k', cmap=plt.cm.coolwarm)
        axs[i].set_xlabel(title_namesl[feature_1_less], fontsize=14)
        axs[i].set_ylabel(title_namesl[feature_2_less], fontsize=14)
        axs[i].tick_params(axis='both', which='major', labelsize=12)
        x_ticks = np.arange(np.floor(X_pair_stdl[:, 0].min()), np.ceil(X_pair_stdl[:, 0].max()) + 1, step=1).astype(int)
        y_ticks = np.arange(np.floor(X_pair_stdl[:, 1].min()), np.ceil(X_pair_stdl[:, 1].max()) + 1, step=1).astype(int)
        axs[i].set_xticks(x_ticks)
        axs[i].set_yticks(y_ticks)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    # Generate a timestamp for the file suffix
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = 'DataViz'

    # Construct the image path with the timestamp
    img_filename = f"svm_less_dataviz_{timestamp}.png"
    img_path_less = os.path.join(folder_path, img_filename)

    # Save the plot with the constructed file path
    plt.savefig(img_path_less)
    plt.close()

    return img_path_less

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