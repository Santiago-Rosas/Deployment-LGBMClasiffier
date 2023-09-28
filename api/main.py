from fastapi import FastAPI 
from .app.models import Input, Output 
from .app.views import get_model_response

app = FastAPI(docs_url='/')
app.title = "Drinker or not drinker prediction using LGBM model" 
app.version = "v0.0.1"


model_param = "LGBM model"
accuracy= "73%"

@app.get('/info_model',tags=['Model information and accuracy'])
async def model_info():
    """Return model information and accuracy"""
    return {
        "Model": model_param,
        "Accuracy": accuracy
    }


@app.post('/predict',  tags=['Model prediction'], response_model= Output)
async def model_predict(input: Input) -> Output:
    """Predict with input data as the example below"""
    response = get_model_response(input)
    return response
