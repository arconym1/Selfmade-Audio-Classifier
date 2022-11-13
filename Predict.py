from keras import models
import cv2 as cv
import numpy as np

def predict():
    class_names = ["Your", "Categories"] # ( Must be in right order )
    model = models.load_model("YourModelName.h5")
    img = cv.imread("PathToImageToPredict")
    img = cv.resize(img, (64, 64)) # You may be getting better results by resizing the image

    np_img = np.array([img]) / 255
    prediction = model.predict(np_img)
    index = np.argmax(prediction)

    print("Prediction : ", class_names[index])
    print("Full_Prediction: ", prediction)

    return class_names[index]


if __name__ == '__main__':
    predict()
