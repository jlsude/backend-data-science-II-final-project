from ..source_dataset import dataset

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


KNN_more = dataset[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", "thalach", "thal", 'heart_disease']].copy()

KNN_x_more = KNN_more.iloc[:, :10].values
KNN_y_more = KNN_more.iloc[:, 10].values

x_train_more, x_test_more, y_train_more, y_test_more = train_test_split(KNN_x_more, KNN_y_more, test_size = 0.2, random_state=42)

knn_more = KNeighborsClassifier(n_neighbors = 12)
knn_more.fit(x_train_more, y_train_more)

y_pred_more = knn_more.predict(x_test_more)

def KNN_more(history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal):
    
    KNN_prediction_more = knn_more.predict([[history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal]])
    KNN_accuracy_more = knn_more.score(x_test_more, y_test_more)
    KNN_probabilities_more = knn_more.predict_proba([[history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal]])
    KNN_cm_more = confusion_matrix(y_test_more, y_pred_more)

    return KNN_prediction_more, KNN_accuracy_more, KNN_probabilities_more, KNN_cm_more
