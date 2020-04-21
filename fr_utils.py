"""
Module with models and functions for face detection and recognition.
Available functions with required arguments:

read_img(filepath)
resize_img(image, required_size=(160, 160))
detect_faces(image, detector="hog")
get_roi(image, coordinates)
prewhiten_img(x)
l2_normalize(x, axis=-1, epsilon=1e-10)
get_embedding(face_image, model=facenet_model)
calculate_distance(face_encodings, face_to_compare)
"""

import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import dlib
from keras.models import load_model


# Load MTCNN face detector.
mtcnn_detector = MTCNN()
# Load HOG face detector.
hog_detector = dlib.get_frontal_face_detector()
# load face encoding model
try:
    facenet_model = load_model('facenet_keras_model.h5')
except OSError:
    print("Problems to load facenet_keras_model.")


def read_img(filepath, mode="RGB"):
    """
    Function to read image from file and convert it to RGB by default, or leaves BGR.
    Input: file path.
    Output: image as numpy array.
    """
    img = cv2.imread(filepath)
    if mode == "RGB":
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_img(image, required_size=(160, 160)):
    """
    Function to resize image.
    Input: image as numpy array.
    Param required_size: required_size as tuple (x,x); default is 160x160.
    Output: image as numpy array.
    """
    img = cv2.resize(image, required_size, interpolation=cv2.INTER_AREA)
    return img


def detect_faces(image, detector="mtcnn"):
    """
    Returns an array of bounding boxes of each face detected in provided image.
    Input: image in RGB mode as numpy array.
    Param detector: face detection model - hog (default) or mtcnn.
    Output: List of tuples with detected face coordinates - top, bottom, left, right.
            Coordinates are in right order to extract roi from image.
    """
    img_boundaries = image.shape
    if detector == "hog":
        detections = hog_detector(image)
        # Use max/min function and img_boundaries to make sure coordinates are within image boundaries.
        faces = [(max(face.top(), 0), min(face.bottom(), img_boundaries[0]), max(
            face.left(), 0), min(face.right(), img_boundaries[1])) for face in detections]
        return faces
    elif detector == "mtcnn":
        faces = []
        detections = mtcnn_detector.detect_faces(image)
        for d in detections:
            if d["confidence"] > 0.9:
                face = max(d["box"][1], 0), min(d["box"][1] + d["box"][3], img_boundaries[0]
                                                ), max(d["box"][0], 0), min(d["box"][0] + d["box"][2], img_boundaries[1])
                faces.append(face)
        return faces
    else:
        return list()


def get_roi(image, coordinates):
    """
    Function to extract region of interest (roi) from image using coordinates.
    Input image: image as numpy array to extract from.
    Input coordinates: coordinates of roi in form of tuple (top, bottom, left, right)
    """
    roi = image[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
    return roi


def prewhiten_img(x):
    """
    Funtion to subtract the mean and normalize the range of the pixel values of input images.
    Input: either one image or list of images as numpy array.
    Output: numpy array.
    """
    # For list of images.
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    # For single image.
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    """
    L2 normalization function to normalize 128D face embedding.
    Input: numpy array.
    Output: numpy array.
    """
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def get_embedding(face_image, model=facenet_model):
    """
    Given an image, return the 128-dimension face encoding.
    Input: image as numpy array of shape (160, 160, 3).
    Param model: face embedding model; built-in (default) is FaceNet Keras model.
    Output: numpy array of shape (1, 128)
    """
    # Resize image to 160x160 (default dimensions applicable for FaceNet model).
    img = resize_img(face_image)
    # Normalize face image.
    img = prewhiten_img(img)
    # Exapand dimensions from 3 to 4 as model expects list of samples (4D array).
    sample = np.expand_dims(img, axis=0)
    # Make prediction to get embedding
    pred = model.predict(sample)
    # Normalize 128D output.
    pred = l2_normalize(pred)
    return pred


def calculate_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    Input face_encodings: List of face encodings to compare. Should be numpy array of shape (1, 128) or list of them.
    Input face_to_compare: A face encoding to compare against. Should be numpy array of shape (1, 128)
    """
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
