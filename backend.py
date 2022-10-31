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
from backend_dependencies.translator.translator import hf_translation_request, Payload
from backend_dependencies.translator.hf_api_urls import (
    API_URL_EN_NL,
    API_URL_NL_EN,
    API_URL_DE_EN,
    API_URL_EN_DE,
    API_URL_DE_NL,
)

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
@app.post("/translator")
def translator(payload: Payload):
    """
    Function to translate from English to Dutch
    """
    if payload.input_language == "en" and payload.output_language == "nl":
        API_URL = API_URL_EN_NL
    elif payload.input_language == "nl" and payload.output_language == "en":
        API_URL = API_URL_NL_EN
    elif payload.input_language == "de" and payload.output_language == "en":
        API_URL = API_URL_DE_EN
    elif payload.input_language == "en" and payload.output_language == "de":
        API_URL = API_URL_EN_DE
    elif payload.input_language == "de" and payload.output_language == "nl":
        API_URL = API_URL_DE_NL

    return hf_translation_request(payload.dict()["text"], API_URL)


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
