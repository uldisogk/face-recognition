# Models and functions for face detection and recognition.

This repository contains models (or links to them) and convenience functions to build face detection and recognition systems as well as few Jupyter Notebooks with usage examples.

Every face recognition system can be devided into two parts - face detection and face recognition. For face detection here in this solution are two options - multitask cascaded convolutional network (MTCNN) or histogram of oriented gradients (HOG).
Face recognition part is based on face embeddings created with FaceNet. These 128D vectors can then be easely compared by calculating euclidean distance or used to train classification model.  

fr_utils.py - library of all necessary functions, which can be imported as module. 
face_detection.ipynb - Jupyter Notebook with face detection functions and usage examples.  
face_recognition.ipynb - Jupyter Notebook with face recognition functions and usage examples.  
Examples folder contains one sample image to work with.

Please intstall prerequisites and download (into root folder) pretrained FaceNet model before using project files.

## Prerequisites

[Python 3.7](https://www.python.org/downloads/)  
[Tensorflow 2.0](https://www.tensorflow.org/install)  
[Keras 2.3 ](https://keras.io/#installation) 
[OpenCV 4.1](https://pypi.org/project/opencv-python/)  
[MTCNN 0.1](https://pypi.org/project/mtcnn/)  
[Dlib 19.18](https://pypi.org/project/dlib/)  
[Scikit-learn 0.22](https://scikit-learn.org/stable/install.html)  
[Jupyter 1.0](https://jupyter.org/install) (optional)  
[Seaborn 0.10](https://seaborn.pydata.org/installing.html) (optional)  
[Pandas 0.25](https://pypi.org/project/pandas/) (optional)  

## Pretrained models

[Here](https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn) is pretrained Keras model (trained by MS-Celeb-1M dataset).

## Inspiration and related resources
[Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)  
[Facial recognition API](https://github.com/ageitgey/face_recognition?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)  
[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)  
[Facenet Keras model](https://github.com/nyoki-mtl/keras-facenet)  
[FaceNet, David Sandberg implementation](https://github.com/davidsandberg/facenet)  
[OpenFace](http://cmusatyalab.github.io/openface/)  
[OpenCV](https://opencv.org/)  
[Dlib](http://dlib.net/)  
[PyImageSearch - Face Applications](https://www.pyimagesearch.com/category/faces/)  
[Face detection - OpenCV, Dlib and Deep Learning](https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/)  
[Machinelearningmastery article on face recognition](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)  
[MegaFace](http://megaface.cs.washington.edu/?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&utm_content=78452820&_hsenc=p2ANqtz-_YyjDQURXiBDLYh-6uusHXMakpWIBUDl8IglSlh7h3fDC0tXZvwta3g63z0AZYmRLQmR_95YdFL6UP6Z0yLf2X10zHlA&_hsmi=78452820)  
[Wider Face](http://shuoyang1213.me/WIDERFACE/)  
[Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)  
[Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
[Convolutional Neural Networks Course by Andrew Ng on Coursera](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)

## Research papers
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)  
[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
