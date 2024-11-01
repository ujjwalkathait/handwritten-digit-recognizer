import cv2
import tensorflow as tf
import numpy as np
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
IMG_SIZE = 28


model = tf.keras.models.load_model('model.keras')
cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#   cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()
# if not cap.isOpened():
#   raise IOError("Cannot open webcam")

text = "Some text in a box!"
# get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale = font_scale, thickness = 1)[0]
# set the text start position
text_offset_x = 10
text_offset_y = 648 - 25
# make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
# cv2.rectangle (img, box_coords[0], box_coords[1], rectangle_bar, cv2.FILLED)
# cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale = font_scale, color = (0,0,0), thickness = 1)
cntr = 0;   
while True:
  ret, frame = cap.read()
  cntr = cntr+1;
  if((cntr%2) == 0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
    newimg = tf.keras.utils.normalize(resized, axis=1)
    newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    predictions = model.predict(newimg)
    status = np.argmax(predictions)
    print(status)
    print(type(status))

    x1, y1, w1, h1 = 0, 0, 175, 75
    # Draw black background rectangle
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
    # Add Text
    cv2.putText(frame, status.astype(str), (x1 + int(w1/5), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)


    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(faceCascade.empty())
    # faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces
    # for(x, y, w, h) in faces:
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)


    # Use putText() method for inserting text on video

    cv2.imshow('Handwritten Digits Recognition', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()