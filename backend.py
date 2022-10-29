import numpy as np
from fastapi import FastAPI, File, UploadFile
import cv2

from starlette.responses import Response
from backend_dependencies.bank_note_classification.bank_note_api_utils import (
    get_prediction,
    BankNote,
)
from backend_dependencies.people_recognition_caption.people_recognition_caption_utils import (
    image_caption_draw_rectangle,
)
from backend_dependencies.translator_en_nl.translator_en_nl import en_nl_query, Payload


# FastAPI app
app = FastAPI(
    name="appfolio_backend",
    title="backend",
    description="Backend for my small projects",
)


@app.get("/")
def index():
    return {"status": "ok"}


# Bank note prediction
@app.post("/predict_bank_note")
def predict_bank_note(data: BankNote):
    """
    Function to predict bank note authenticity
    """
    data = np.array(
        [data.variance, data.skewness, data.curtosis, data.entropy], dtype=np.float32
    )
    prediction = get_prediction(data)

    if prediction == 1:
        return {"Prediction": "Warning! The bank note is Fake (1) ⚠️"}
    else:
        return {"Prediction": "The bank note is Real (0) ✅"}


# Translator En-Nl
@app.post("/translate_en_nl")
def translate_en_nl(payload: Payload):
    """
    Function to translate from English to Dutch
    """
    return en_nl_query(payload.dict()["text"])


# People recognition
@app.post("/people_recognition_caption")
async def people_recognition_caption(img: UploadFile = File(...)):
    ## Receiving and decoding the image
    contents = await img.read()

    # Do all the necessary processing
    image = image_caption_draw_rectangle(contents)

    ### Encoding and responding with the image
    captioned_image = cv2.imencode(".png", image)[
        1
    ]  # extension depends on which format is sent from Streamlit
    return Response(content=captioned_image.tobytes(), media_type="image/png")
