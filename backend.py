import os
import os.path as osp
import numpy as np
from fastapi import FastAPI, File, UploadFile
import cv2
import face_recognition
import onnxruntime as ort
from starlette.responses import Response
from backend_dependencies.bank_note_classification.bank_note_api_utils import (
    get_prediction,
    BankNote,
)
from backend_dependencies.people_recognition_caption.people_recognition_caption_utils import (
    face_image_for_onnx_model,
    MyRec,
    crop_image,
)


# FastAPI app
app = FastAPI()


@app.get("/")
def index():
    return {"status": "ok"}


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


@app.post("/people_recognition_caption")
async def people_recognition_caption(img: UploadFile = File(...)):
    font, fontScale, color, thickness = (cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    ## Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resizing without changing aspect ratio
    h, w, c = cv2_img.shape

    aspect_ratio = w / h

    image = cv2.resize(
        cv2_img, (720, int(720 / aspect_ratio)), interpolation=cv2.INTER_AREA
    )

    # Loading trained Model
    ort_session = ort.InferenceSession(
        osp.join(
            "backend_dependencies",
            "people_recognition",
            "people_recognition_model.onnx",
        )
    )
    # Load class Encoding
    encoding_ = np.load(
        osp.join(
            "backend_dependencies", "people_recognition_caption", "class_encoding.npy"
        )
    )

    # Finding face location
    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = crop_image(image, (top, left, bottom, right))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        # prepare for NN inference
        face_image = face_image_for_onnx_model(face_image)

        # get model prediction
        ort_inputs = {ort_session.get_inputs()[0].name: face_image}
        prediction = ort_session.run(None, ort_inputs)[0].argmax()

        prediction = encoding_[prediction]

        # Draw rectangle and write predicted label
        cv2.rectangle(image, (left, top), (right, bottom), (220, 255, 220), 1)
        MyRec(image, left, top, right - left, bottom - top, 10, (0, 250, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image,
            prediction,
            (left, top),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    ### Encoding and responding with the image
    captioned_image = cv2.imencode(".png", image)[
        1
    ]  # extension depends on which format is sent from Streamlit
    return Response(content=captioned_image.tobytes(), media_type="image/png")
