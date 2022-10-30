import cv2
import numpy as np
import os.path as osp
import face_recognition
import onnxruntime as ort


def face_image_for_onnx_model(image):
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = image.transpose((2, 0, 1))
    # Normalising the image here
    image = image / 255.0
    image = image.astype("float32")
    image = image.reshape(1, 3, 224, 224)
    return image


def MyRec(rgb, x, y, w, h, v=20, color=(200, 0, 0), thikness=2):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x, y), (x + v, y), color, thikness)
    cv2.line(rgb, (x, y), (x, y + v), color, thikness)

    cv2.line(rgb, (x + w, y), (x + w - v, y), color, thikness)
    cv2.line(rgb, (x + w, y), (x + w, y + v), color, thikness)

    cv2.line(rgb, (x, y + h), (x, y + h - v), color, thikness)
    cv2.line(rgb, (x, y + h), (x + v, y + h), color, thikness)

    cv2.line(rgb, (x + w, y + h), (x + w, y + h - v), color, thikness)
    cv2.line(rgb, (x + w, y + h), (x + w - v, y + h), color, thikness)

    return


def display_image_for_inference(face_image):
    face_image = face_image * 255
    face_image = face_image.squeeze(0)
    face_image = face_image.transpose((1, 2, 0)).astype("uint8")
    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    print(face_image.shape)
    cv2.imshow("Face", face_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return


def crop_image(image, box):
    return image[box[0] : box[2], box[1] : box[3]]


def image_caption_draw_rectangle(contents):
    font, fontScale, color, thickness = (cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
            "people_recognition_caption",
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

    return image
