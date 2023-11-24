from ..source_dataset import dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


SVM_less = dataset[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", 'heart_disease']].copy()

SVM_x_less = SVM_less.iloc[:, :8].values
SVM_y_less = SVM_less.iloc[:, 8].values


scaler = StandardScaler()
SVM_x_less_standardized = scaler.fit_transform(SVM_x_less)

x_train_less, x_test_less, y_train_less, y_test_less = train_test_split(SVM_x_less_standardized, SVM_y_less, test_size=0.2, random_state=42)

SVM_model_less = SVC(kernel='linear', probability=True)
SVM_model_less.fit(x_train_less, y_train_less)




def SVM_less(history, age, gender, trestbps, cp, chol, fbs, restecg):

    input_values_less = [history, age, gender, trestbps, cp, chol, fbs, restecg]
    input_values_less_standardized = scaler.transform([input_values_less])
    y_pred_less = SVM_model_less.predict(x_test_less)

    SVM_prediction_less = SVM_model_less.predict(input_values_less_standardized)
    SVM_accuracy_less = accuracy_score(y_test_less, y_pred_less)
    SVM_cm_less = confusion_matrix(y_test_less, y_pred_less)
    SVM_probabilities_less = SVM_model_less.predict_proba(input_values_less_standardized)

    return SVM_prediction_less, SVM_accuracy_less, SVM_probabilities_less, SVM_cm_less