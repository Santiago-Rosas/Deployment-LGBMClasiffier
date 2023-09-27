from joblib import load
from scipy.sparse import data
import os 
from io import BytesIO

def get_model():
    model_path = os.environ.get('MODEL_PATH','models/model.pk')
    with open(model_path,'rb') as model_file:
        model = load(BytesIO(model_file.read()))
    return model

def get_scaler():
    sacler_path = os.environ.get('SCALER_PATH','models/scaler.pk')
    with open(sacler_path,'rb') as scaler_file:
        scaler = load(BytesIO(scaler_file.read()))
    return scaler

def get_columns_equivalence():
    columns_equivalence_path = os.environ.get('COLUMNS_PATH','models/column_equivalence.pk')
    with open(columns_equivalence_path,'rb') as columns_equivalence_file:
        columns_equivalence = load(BytesIO(columns_equivalence_file.read()))
    return columns_equivalence
