import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('hand_gesture_detector')

img = cv2.imread("Test3.jpg")
img = cv2.resize(img, (640, 240))
cv2.imshow("img", img)
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = grayimg[:120, :320]
print(img1.shape)
cv2.imshow("img1", img1)
img2 = grayimg[:120, 320:640]
print(img2.shape)
cv2.imshow("img2", img2)
img3 = grayimg[120:240, :320]
print(img3.shape)
cv2.imshow("img3", img3)
img4 = grayimg[120:240, 320:640]
print(img4.shape)
cv2.imshow("img4", img4)

img1 = img1.reshape((1, 120, 320, 1))
img2 = img2.reshape((1, 120, 320, 1))
img3 = img3.reshape((1, 120, 320, 1))
img4 = img4.reshape((1, 120, 320, 1))

prediction1 = model.predict(np.array(img1/255))
prediction2 = model.predict(np.array(img2/255))
prediction3 = model.predict(np.array(img3/255))
prediction4 = model.predict(np.array(img4/255))

print(prediction1, prediction2, prediction3, prediction4)

cv2.waitKey()
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)   #id 3 is for width
# cap.set(4, 240)   #id 4 is for height
#
# while True:
#     success, img = cap.read()
#     grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img1 = grayimg[:320, :120]
#     img2 = grayimg[320:640, :120]
#     img3 = grayimg[:320, 120:240]
#     img4 = grayimg[320:640, 120:240]
#
#     img1 = img1.reshape((1, 120, 320, 1))
#     img2 = img2.reshape((1, 120, 320, 1))
#     img3 = img3.reshape((1, 120, 320, 1))
#     img4 = img4.reshape((1, 120, 320, 1))
#
#     prediction1 = model.predict(np.array(img1))
#     prediction2 = model.predict(np.array(img2))
#     prediction3 = model.predict(np.array(img3))
#     prediction4 = model.predict(np.array(img4))
#
#     print(prediction1, prediction2, prediction3, prediction4)
#
#     cv2.imshow("ourchestra", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
