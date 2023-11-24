from ..source_dataset import dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


KNN_prelim = dataset[['history', 'age', 'gender', 'trestbps', 'cp', 'heart_disease']].copy()

KNN_x_prelim = KNN_prelim.iloc[:, :5].values
KNN_y_prelim = KNN_prelim.iloc[:, 5].values

x_train_prelim, x_test_prelim, y_train_prelim, y_test_prelim = train_test_split(KNN_x_prelim, KNN_y_prelim, test_size = 0.2, random_state=42)

knn_prelim = KNeighborsClassifier(n_neighbors = 12)
knn_prelim.fit(x_train_prelim, y_train_prelim)

y_pred_prelim = knn_prelim.predict(x_test_prelim)
KNN_accuracy_prelim = knn_prelim.score(x_test_prelim, y_test_prelim)


def KNN_preliminary(history, age, gender, trestbps, cp):
    
    KNN_prediction_prelim = knn_prelim.predict([[history, age, gender, trestbps, cp]])
    KNN_probabilities_prelim = knn_prelim.predict_proba([[history, age, gender, trestbps, cp]])
    KNN_cm_prelim = confusion_matrix(y_test_prelim, y_pred_prelim)

    return KNN_prediction_prelim, KNN_accuracy_prelim, KNN_probabilities_prelim, KNN_cm_prelim    
