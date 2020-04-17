import cv2
import numpy as np

cap=cv2.VideoCapture("vtest.avi")
#cap=cv2.VideoCapture("Panasonic_test_cnn.avi")

proceed = True
first_frame = True

alpha = 0.01
t0 = 14.0

result, frame = cap.read()
if not result:
    quit(-1)

background = frame.astype(float)
threshold = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), float)
threshold.fill(t0)
label = np.zeros((frame.shape[0], frame.shape[1], 1), np.float)
one_mat = np.ones_like(label)


while proceed:
    result, frame = cap.read()
    if not result:
        break
    coef1 = one_mat[:, :, 0] - alpha * (one_mat[:, :, 0] - label[:, :, 0])
    coef2 = alpha * (one_mat[:, :, 0] - label[:, :, 0])
    for c in range(0, frame.shape[2]):
        background[:, :, c] = np.multiply(background[:, :, c], coef1) + np.multiply(frame[:, :, c], coef2)
        threshold[:, :, c] = np.multiply(threshold[:, :, c], coef1) + \
            np.multiply(np.abs(background[:, :, c]-frame[:, :, c]), coef2)

    label[:,:,0] = np.amax(np.abs(background - frame) > 5 * threshold , axis=2)

    cv2.imshow("frame", frame)
    background_to_show = background/255.0
    cv2.imshow("background", background_to_show)
    threshold_to_show=threshold/15.0
    cv2.imshow("threshold", threshold_to_show)
    cv2.imshow("label", label)
    key=cv2.waitKey(1)
    if key in [ord('q'),  27, 32]:
        quit(0)

