from ..source_dataset import dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


SVM_prelim = dataset[['history', 'age', 'gender', 'trestbps', 'cp', 'heart_disease']].copy()

SVM_x_prelim = SVM_prelim.iloc[:, :5].values
SVM_y_prelim = SVM_prelim.iloc[:, 5].values


scaler = StandardScaler()
SVM_x_prelim_standardized = scaler.fit_transform(SVM_x_prelim)

x_train_prelim, x_test_prelim, y_train_prelim, y_test_prelim = train_test_split(SVM_x_prelim_standardized, SVM_y_prelim, test_size=0.2, random_state=42)

SVM_model_prelim = SVC(kernel='linear', probability=True)
SVM_model_prelim.fit(x_train_prelim, y_train_prelim)




def SVM_preliminary(history, age, gender, trestbps, cp):

    input_values_prelim = [history, age, gender, trestbps, cp]
    input_values_prelim_standardized = scaler.transform([input_values_prelim])
    y_pred_prelim = SVM_model_prelim.predict(x_test_prelim)

    SVM_prediction_prelim = SVM_model_prelim.predict(input_values_prelim_standardized)
    SVM_accuracy_prelim = accuracy_score(y_test_prelim, y_pred_prelim)
    SVM_cm_prelim = confusion_matrix(y_test_prelim, y_pred_prelim)
    SVM_probabilities_prelim = SVM_model_prelim.predict_proba(input_values_prelim_standardized)

    return SVM_prediction_prelim, SVM_accuracy_prelim, SVM_probabilities_prelim, SVM_cm_prelim