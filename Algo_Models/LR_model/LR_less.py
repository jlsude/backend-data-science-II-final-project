from ..source_dataset import dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


#Logistic Regression LESS than 35 stage 1

LR_Dataset_less_35 = dataset[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", 'heart_disease']].copy()
LR_x_values_less_35 = LR_Dataset_less_35[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg"]].values
LR_y_values_less_35= LR_Dataset_less_35.iloc[:, 8].values

#Logistic Regression LESS than 35 stage 2

x_train_less_35, x_test_less_35, y_train_less_35, y_test_less_35 = train_test_split(LR_x_values_less_35, LR_y_values_less_35,test_size = 0.2,
                                                    random_state=42)
sc = StandardScaler()
x_train_less_35 = sc.fit_transform(x_train_less_35)
x_test_less_35 = sc.transform(x_test_less_35)

classifier_less_35 = LogisticRegression(random_state = 0)
classifier_less_35.fit(x_train_less_35, y_train_less_35)

y_pred_less_35= classifier_less_35.predict(x_test_less_35)
cm_less_35 = confusion_matrix(y_test_less_35, y_pred_less_35)

#Logistic Regression LESS than 35 stage 3

b0_less_35 = classifier_less_35.intercept_[0]

b_coefs_less_35 = classifier_less_35.coef_[0]


def LR_less(history, age, gender, trestbps, cp, chol, fbs, restecg):

    features = np.array([history, age, gender, trestbps, cp, chol, fbs, restecg])
    features_less_scaled = sc.transform([features])
    under = (b0_less_35 + np.dot(b_coefs_less_35, features_less_scaled.T))

    LR_probability_less = 1 / (1 + np.exp(-under))
    LR_cm_less = confusion_matrix(y_test_less_35, y_pred_less_35)
    LR_accuracy_less = accuracy_score(y_test_less_35, y_pred_less_35)

    return LR_probability_less, LR_cm_less, LR_accuracy_less

