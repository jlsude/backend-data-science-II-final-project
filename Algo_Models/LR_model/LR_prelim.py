from ..source_dataset import dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score



#Logistic Regression Preliminary stage 1

LR_DatasetPrelim = dataset[["history", "age", "gender", "trestbps", "cp", 'heart_disease']].copy()
LR_x_values_prelim = LR_DatasetPrelim[["history", "age", "gender", "trestbps", "cp"]].values
LR_y_values_prelim = LR_DatasetPrelim.iloc[:, 5].values

#Logistic Regression Preliminary stage 2

x_train_prelim, x_test_prelim, y_train_prelim, y_test_prelim = train_test_split(LR_x_values_prelim, LR_y_values_prelim,test_size = 0.25,
                                                    random_state=0)
sc = StandardScaler()
x_train_prelim = sc.fit_transform(x_train_prelim)
x_test_prelim = sc.transform(x_test_prelim)

classifier_prelim = LogisticRegression(random_state = 0)
classifier_prelim.fit(x_train_prelim, y_train_prelim)

y_pred_prelim = classifier_prelim.predict(x_test_prelim)
cm_prelim = confusion_matrix(y_test_prelim, y_pred_prelim)
LR_prelim_accuracy = accuracy_score(y_test_prelim, y_pred_prelim)


#Logistic Regression Preliminary stage 3

b0_prelim = classifier_prelim.intercept_[0]
b1_prelim = classifier_prelim.coef_[0][0]
b2_prelim = classifier_prelim.coef_[0][1]
b3_prelim = classifier_prelim.coef_[0][2]
b4_prelim = classifier_prelim.coef_[0][3]
b5_prelim = classifier_prelim.coef_[0][4]


def LRModel_Preliminary(history, age, gender, trestbps, cp):
    under = -(b0_prelim + b1_prelim*history + b2_prelim*age + b3_prelim*gender + b4_prelim*trestbps + b5_prelim*cp)
    LR_probability_prelim = 1 / (1 + np.exp(under))
    LR_cm_prelim = confusion_matrix(y_test_prelim, y_pred_prelim)
    LR_accuracy_prelim = accuracy_score(y_test_prelim, y_pred_prelim)

    return LR_probability_prelim, LR_cm_prelim, LR_accuracy_prelim