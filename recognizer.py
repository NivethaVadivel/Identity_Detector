import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Lalith', 'Lalith', 'Lalith', 'Lalith', 'Nivetha', 'Nivetha', 'Nivetha', 'Nivetha', 'Raguram',
         'Raguram','Raguram', 'Raguram', 'Mouliraj', 'Mouliraj', 'Mouliraj', 'Mouliraj']
rno = ['None', '20CSR105', '20CSR105', '20CSR105', '20CSR105', '20CSR143', '20CSR143', '20CSR143', '20CSR143',
       '20CSR160',
       '20CSR160', '20CSR160', '20CSR160', '20CSR125', '20CSR125', '20CSR125', '20CSR125']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print("\nRecognizing Faces")
while True:

    ret, img = cam.read()
    # img = cv2.flip(img, -1)  # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # print(confidence)
        confidence = round((100 - confidence))

        # Check if confidence is less them 100 ==> "0" is perfect match
        if int(confidence) > 30:
            no = rno[id]
            id = names[id]

            confidence = "  {0}%".format(round(confidence))
            cv2.putText(img, str(id), (x + 5, y - 40), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(no), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        else:
            id = "unknown"
            no = "unknown"
            confidence = "  {0}%".format(round(confidence))
            cv2.putText(img, str(id), (x + 5, y - 40), font, 1, (255, 255, 255), 2)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\nExiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()