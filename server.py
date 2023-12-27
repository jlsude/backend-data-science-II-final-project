from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from Algo_Models.LR_model.LR_prelim import LRModel_Preliminary
from Algo_Models.LR_model.LR_more import LR_more
from Algo_Models.LR_model.LR_less import LR_less

from Algo_Models.KNN_model.KNN_prelim import KNN_preliminary
from Algo_Models.KNN_model.KNN_more import KNN_more
from Algo_Models.KNN_model.KNN_less import KNN_less

from Algo_Models.SVM_model.SVM_prelim import SVM_preliminary
from Algo_Models.SVM_model.SVM_more import SVM_more
from Algo_Models.SVM_model.SVM_less import SVM_less

from Algo_Models.DataViz.SVM_more_dataviz import SVM_more_dataviz
from Algo_Models.DataViz.SVM_less_dataviz import SVM_less_dataviz


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Data Science API is up and running!"

@app.route('/activate', methods=['GET'])
def activate():
    return "API is activated!"

@app.route('/preliminary', methods=['POST'])
def preliminary_prediction():
    data = request.get_json(force=True)
    history = data['history']
    age = data['age']
    gender = data['gender']
    trestbps = data['trestbps']
    cp = data['cp']

    LR_probability_prelim, LR_cm_prelim, LR_accuracy_prelim = LRModel_Preliminary(history, age, gender, trestbps, cp)
    KNN_prediction_prelim, KNN_accuracy_prelim, KNN_probabilities_prelim, KNN_cm_prelim = KNN_preliminary(history, age, gender, trestbps, cp)
    SVM_prediction_prelim, SVM_accuracy_prelim, SVM_probabilities_prelim, SVM_cm_prelim = SVM_preliminary(history, age, gender, trestbps, cp)

    return jsonify({
        "LR_Model_Preliminary": {
                'LR_probability': LR_probability_prelim.tolist(),
                'LR_confusion_matrix': {
                    'TN': LR_cm_prelim[0][0].tolist(),
                    'FP': LR_cm_prelim[0][1].tolist(),
                    'FN': LR_cm_prelim[1][0].tolist(),
                    'TP': LR_cm_prelim[1][1].tolist()
                },
                'LR_accuracy': LR_accuracy_prelim.tolist()
        },
        "KNN_Model_Preliminary": {
                'KNN_probability': KNN_probabilities_prelim[0][1].tolist(),
                'KNN_prediction': KNN_prediction_prelim[0].tolist(),
                'KNN_confusion_matrix': {
                    'TN': KNN_cm_prelim[0][0].tolist(),
                    'FP': KNN_cm_prelim[0][1].tolist(),
                    'FN': KNN_cm_prelim[1][0].tolist(),
                    'TP': KNN_cm_prelim[1][1].tolist()
                },
                'KNN_accuracy': KNN_accuracy_prelim.tolist()
        },
        "SVM_Model_Preliminary": {
                'SVM_probability': SVM_probabilities_prelim[0][1].tolist(),
                'SVM_prediction': SVM_prediction_prelim[0].tolist(),
                'SVM_confusion_matrix': {
                    'TN': SVM_cm_prelim[0][0].tolist(),
                    'FP': SVM_cm_prelim[0][1].tolist(),
                    'FN': SVM_cm_prelim[1][0].tolist(),
                    'TP': SVM_cm_prelim[1][1].tolist()
                },
                'SVM_accuracy': SVM_accuracy_prelim.tolist()
        }
        
    })

@app.route('/more_35', methods=['POST'])
def more_35_prediction():
    data = request.get_json(force=True)
    history = data['history']
    age = data['age']
    gender = data['gender']
    trestbps = data['trestbps']
    cp = data['cp']
    chol = data['chol']
    fbs = data['fbs']
    restecg = data['restecg']
    thalach = data['thalach']
    thal = data['thal']

    LR_probability_more, LR_cm_more, LR_accuracy_more = LR_more(history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal)
    KNN_prediction_more, KNN_accuracy_more, KNN_probabilities_more, KNN_cm_more = KNN_more(history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal)
    SVM_prediction_more, SVM_accuracy_more, SVM_probabilities_more, SVM_cm_more = SVM_more(history, age, gender, trestbps, cp, chol, fbs, restecg, thalach, thal)

    return jsonify({
        "LR_Model_more": {
                'LR_probability': LR_probability_more.tolist(),
                'LR_confusion_matrix': {
                    'TN': LR_cm_more[0][0].tolist(),
                    'FP': LR_cm_more[0][1].tolist(),
                    'FN': LR_cm_more[1][0].tolist(),
                    'TP': LR_cm_more[1][1].tolist()
                },
                'LR_accuracy': LR_accuracy_more.tolist()
        },
        "KNN_Model_more": {
                'KNN_probability': KNN_probabilities_more[0][1].tolist(),
                'KNN_prediction': KNN_prediction_more[0].tolist(),
                'KNN_confusion_matrix': {
                    'TN': KNN_cm_more[0][0].tolist(),
                    'FP': KNN_cm_more[0][1].tolist(),
                    'FN': KNN_cm_more[1][0].tolist(),
                    'TP': KNN_cm_more[1][1].tolist()
                },
                'KNN_accuracy': KNN_accuracy_more.tolist()
        },
        "SVM_Model_more": {
                'SVM_probability': SVM_probabilities_more[0][1].tolist(),
                'SVM_prediction': SVM_prediction_more[0].tolist(),
                'SVM_confusion_matrix': {
                    'TN': SVM_cm_more[0][0].tolist(),
                    'FP': SVM_cm_more[0][1].tolist(),
                    'FN': SVM_cm_more[1][0].tolist(),
                    'TP': SVM_cm_more[1][1].tolist()
                },
                'SVM_accuracy': SVM_accuracy_more.tolist()
        }
        
    })

@app.route('/less_35', methods=['POST'])
def less_35_prediction():
    data = request.get_json(force=True)
    history = data['history']
    age = data['age']
    gender = data['gender']
    trestbps = data['trestbps']
    cp = data['cp']
    chol = data['chol']
    fbs = data['fbs']
    restecg = data['restecg']

    LR_probability_less, LR_cm_less, LR_accuracy_less = LR_less(history, age, gender, trestbps, cp, chol, fbs, restecg)
    KNN_prediction_less, KNN_accuracy_less, KNN_probabilities_less, KNN_cm_less = KNN_less(history, age, gender, trestbps, cp, chol, fbs, restecg)
    SVM_prediction_less, SVM_accuracy_less, SVM_probabilities_less, SVM_cm_less = SVM_less(history, age, gender, trestbps, cp, chol, fbs, restecg)

    return jsonify({
        "LR_Model_less": {
                'LR_probability': LR_probability_less.tolist(),
                'LR_confusion_matrix': {
                    'TN': LR_cm_less[0][0].tolist(),
                    'FP': LR_cm_less[0][1].tolist(),
                    'FN': LR_cm_less[1][0].tolist(),
                    'TP': LR_cm_less[1][1].tolist()
                },
                'LR_accuracy': LR_accuracy_less.tolist()
        },
        "KNN_Model_less": {
                'KNN_probability': KNN_probabilities_less[0][1].tolist(),
                'KNN_prediction': KNN_prediction_less[0].tolist(),
                'KNN_confusion_matrix': {
                    'TN': KNN_cm_less[0][0].tolist(),
                    'FP': KNN_cm_less[0][1].tolist(),
                    'FN': KNN_cm_less[1][0].tolist(),
                    'TP': KNN_cm_less[1][1].tolist()
                },
                'KNN_accuracy': KNN_accuracy_less.tolist()
        },
        "SVM_Model_less": {
                'SVM_probability': SVM_probabilities_less[0][1].tolist(),
                'SVM_prediction': SVM_prediction_less[0].tolist(),
                'SVM_confusion_matrix': {
                    'TN': SVM_cm_less[0][0].tolist(),
                    'FP': SVM_cm_less[0][1].tolist(),
                    'FN': SVM_cm_less[1][0].tolist(),
                    'TP': SVM_cm_less[1][1].tolist()
                },
                'SVM_accuracy': SVM_accuracy_less.tolist()
        }
        
    })

@app.route('/dataviz_svm_more', methods=['GET'])
def dataviz_more():
    img_path_more = SVM_more_dataviz()

    return send_file(img_path_more, mimetype='image/png', as_attachment=True)

@app.route('/dataviz_svm_less', methods=['GET'])
def dataviz_less():
    img_path_less = SVM_less_dataviz()

    return send_file(img_path_less, mimetype='image/png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
