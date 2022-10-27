import os
import os.path as osp
import numpy as np
from fastapi import FastAPI
from backend_dependencies.bank_note_classification.bank_note_api_utils import get_prediction, BankNote

# FastAPI app
app = FastAPI()

@app.post("/predict_bank_note")
def predict_bank_note(data: BankNote):
    """
    Function to predict bank note authenticity
    """
    data = np.array(
        [data.variance, data.skewness, data.curtosis, data.entropy], dtype=np.float32
    )
    prediction = get_prediction(data)

    # if prediction == 1:
    #     return {"Prediction": "Warning! The bank note is Fake"}
    # else:
    #     return {"Prediction": ":) The bank note is Real"}

    if prediction == 1:
        return {"Prediction": 1}
    else:
        return {"Prediction": 0}




