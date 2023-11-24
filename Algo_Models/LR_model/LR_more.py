from ..source_dataset import dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


#Logistic Regression More than 35 stage 1

LR_Dataset_more_35 = dataset[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", "thalach", "thal", 'heart_disease']].copy()
LR_x_values_more_35 = LR_Dataset_more_35[["history", "age", "gender", "trestbps", "cp", "chol", "fbs", "restecg", "thalach", "thal"]].values
LR_y_values_more_35= LR_Dataset_more_35.iloc[:, 10].values

#Logistic Regression More than 35 stage 2

x_train_more_35, x_test_more_35, y_train_more_35, y_test_more_35 = train_test_split(LR_x_values_more_35, LR_y_values_more_35,test_size = 0.25,
                                                    random_state=0)
sc = StandardScaler()
x_train_more_35 = sc.fit_transform(x_train_more_35)
x_test_more_35 = sc.transform(x_test_more_35)

classifier_more_35 = LogisticRegression(random_state = 0)
classifier_more_35.fit(x_train_more_35, y_train_more_35)

y_pred_more_35= classifier_more_35.predict(x_test_more_35)

#Logistic Regression More than 35 stage 3

b0_more_35 = classifier_more_35.intercept_[0]
b1_more_35 = classifier_more_35.coef_[0][0]
b2_more_35 = classifier_more_35.coef_[0][1]
b3_more_35 = classifier_more_35.coef_[0][2]
b4_more_35 = classifier_more_35.coef_[0][3]
b5_more_35 = classifier_more_35.coef_[0][4]
b6_more_35 = classifier_more_35.coef_[0][5]
b7_more_35 = classifier_more_35.coef_[0][6]
b8_more_35 = classifier_more_35.coef_[0][7]
b9_more_35 = classifier_more_35.coef_[0][8]
b10_more_35 = classifier_more_35.coef_[0][9]




def LR_more(history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal):
    under = -(b0_more_35 + b1_more_35*history + b2_more_35*age + b3_more_35*gender + b4_more_35*trestbps + b5_more_35*cp
            + b6_more_35*chol + b7_more_35*fbs + b8_more_35*restecg + b9_more_35*thalach + b10_more_35*thal)
    LR_probability_more = 1 / (1 + np.exp(under))
    LR_cm_more = confusion_matrix(y_test_more_35, y_pred_more_35)
    LR_accuracy_more = accuracy_score(y_test_more_35, y_pred_more_35)

    return LR_probability_more, LR_cm_more, LR_accuracy_more

