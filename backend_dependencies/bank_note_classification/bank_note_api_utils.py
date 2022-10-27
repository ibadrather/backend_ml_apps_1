import onnxruntime as ort
from pydantic import BaseModel
import os.path as osp


def get_prediction(data):
    # Loadin the ONNX model
    ort_session = ort.InferenceSession(
        osp.join(
            "backend_dependencies", "bank_note_classification", "bank_note_model.onnx"
        )
    )
    """
    Function to get predictions from ONNX model
    """
    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: data}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    return ort_outs.argmax()


class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
