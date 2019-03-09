import cv2
import numpy as np
import tensorflow as tf


#This opens a camera feed and makes it a 3D tensor
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.resize(frame, dsize=(513, 513), interpolation=cv2.INTER_CUBIC)
tensorImg = tf.convert_to_tensor(frame)


print(str(frame))
print(str(frame.shape))
sess = tf.Session()

# print(sess.run(tf.print(tensorImg)))
cv2.imwrite("OutTensor.jpg", sess.run(tensorImg))


cv2.imwrite("./img.jpg", frame)