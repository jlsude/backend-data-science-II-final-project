from ..source_dataset import dataset

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


KNN_less = dataset[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", 'heart_disease']].copy()

KNN_x_less = KNN_less.iloc[:, :8].values
KNN_y_less = KNN_less.iloc[:, 8].values

x_train_less, x_test_less, y_train_less, y_test_less = train_test_split(KNN_x_less, KNN_y_less, test_size = 0.2, random_state=42)

knn_less = KNeighborsClassifier(n_neighbors = 12)
knn_less.fit(x_train_less, y_train_less)

y_pred_less = knn_less.predict(x_test_less)

def KNN_less(history, age, gender, trestbps, cp, chol, fbs, restecg):
    
    KNN_prediction_less = knn_less.predict([[history, age, gender, trestbps, cp, chol, fbs, restecg]])
    KNN_accuracy_less = knn_less.score(x_test_less, y_test_less)
    KNN_probabilities_less = knn_less.predict_proba([[history, age, gender, trestbps, cp, chol, fbs, restecg]])
    KNN_cm_less = confusion_matrix(y_test_less, y_pred_less)

    return KNN_prediction_less, KNN_accuracy_less, KNN_probabilities_less, KNN_cm_less
