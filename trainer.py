import cv2
import numpy as np
from PIL import Image
import os


path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("face.xml")


# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids


<<<<<<< HEAD
print("\nTraining faces. It will take a few seconds, Wait...")
=======
print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
>>>>>>> 9a1a2166ec72b32e87bedf1a9112b7e7cc2dd0b9
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')
# Print the numer of faces trained and end program
<<<<<<< HEAD
print("\n{0} faces trained.\nExiting Program".format(len(np.unique(ids))))
=======
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
>>>>>>> 9a1a2166ec72b32e87bedf1a9112b7e7cc2dd0b9
