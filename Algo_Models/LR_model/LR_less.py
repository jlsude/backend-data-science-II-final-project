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

x_train_less_35, x_test_less_35, y_train_less_35, y_test_less_35 = train_test_split(LR_x_values_less_35, LR_y_values_less_35,test_size = 0.25,
                                                    random_state=0)
sc = StandardScaler()
x_train_less_35 = sc.fit_transform(x_train_less_35)
x_test_less_35 = sc.transform(x_test_less_35)

classifier_less_35 = LogisticRegression(random_state = 0)
classifier_less_35.fit(x_train_less_35, y_train_less_35)

y_pred_less_35= classifier_less_35.predict(x_test_less_35)
cm_less_35 = confusion_matrix(y_test_less_35, y_pred_less_35)

#Logistic Regression LESS than 35 stage 3

b0_less_35 = classifier_less_35.intercept_[0]
b1_less_35 = classifier_less_35.coef_[0][0]
b2_less_35 = classifier_less_35.coef_[0][1]
b3_less_35 = classifier_less_35.coef_[0][2]
b4_less_35 = classifier_less_35.coef_[0][3]
b5_less_35 = classifier_less_35.coef_[0][4]
b6_less_35 = classifier_less_35.coef_[0][5]
b7_less_35 = classifier_less_35.coef_[0][6]
b8_less_35 = classifier_less_35.coef_[0][7]


def LR_less(history, age, gender, trestbps, cp, chol, fbs, restecg):
    under = -(b0_less_35 + b1_less_35*history + b2_less_35*age + b3_less_35*gender + b4_less_35*trestbps + b5_less_35*cp
            + b6_less_35*chol + b7_less_35*fbs + b8_less_35*restecg)
    LR_probability_less = 1 / (1 + np.exp(under))
    LR_cm_less = confusion_matrix(y_test_less_35, y_pred_less_35)
    LR_accuracy_less = accuracy_score(y_test_less_35, y_pred_less_35)

    return LR_probability_less, LR_cm_less, LR_accuracy_less

