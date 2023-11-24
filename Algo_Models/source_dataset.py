import pandas as pd

rawdataset = pd.read_csv('../heart_attack.csv')

history = rawdataset[['heart_disease']].copy()
history.rename(columns={'heart_disease': 'history'}, inplace=True)

dataset = rawdataset[['age', 'gender', 'trestbps', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal', 'heart_disease']].copy()

dataset = pd.concat([history, dataset], axis = 1)