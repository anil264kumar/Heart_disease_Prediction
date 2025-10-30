import pickle

with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
