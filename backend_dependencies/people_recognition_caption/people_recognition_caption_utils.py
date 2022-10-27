import cv2


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
