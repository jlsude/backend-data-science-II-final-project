import os
import pandas as pd

# if the csv file is where this .py file is located
# script_directory = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(script_directory, 'heart_attack.csv')
# rawdataset = pd.read_csv(file_path)

# otherwise
file_path = os.path.join(os.getcwd(), 'heart_attack.csv')
rawdataset = pd.read_csv(file_path)

history = rawdataset[['heart_disease']].copy()
history.rename(columns={'heart_disease': 'history'}, inplace=True)

dataset = rawdataset[['age', 'gender', 'trestbps', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal', 'heart_disease']].copy()

dataset = pd.concat([history, dataset], axis = 1)