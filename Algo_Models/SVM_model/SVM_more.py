from ..source_dataset import dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


SVM_more = dataset[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", "thalach", "thal", 'heart_disease']].copy()

SVM_x_more = SVM_more.iloc[:, :10].values
SVM_y_more = SVM_more.iloc[:, 10].values


scaler = StandardScaler()
SVM_x_more_standardized = scaler.fit_transform(SVM_x_more)

x_train_more, x_test_more, y_train_more, y_test_more = train_test_split(SVM_x_more_standardized, SVM_y_more, test_size=0.2, random_state=42)

SVM_model_more = SVC(kernel='linear', probability=True)
SVM_model_more.fit(x_train_more, y_train_more)




def SVM_more(history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal):

    input_values_more = [history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal]
    input_values_more_standardized = scaler.transform([input_values_more])
    y_pred_more = SVM_model_more.predict(x_test_more)

    SVM_prediction_more = SVM_model_more.predict(input_values_more_standardized)
    SVM_accuracy_more = accuracy_score(y_test_more, y_pred_more)
    SVM_cm_more = confusion_matrix(y_test_more, y_pred_more)
    SVM_probabilities_more = SVM_model_more.predict_proba(input_values_more_standardized)

    return SVM_prediction_more, SVM_accuracy_more, SVM_probabilities_more, SVM_cm_more