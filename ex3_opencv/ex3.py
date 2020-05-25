import numpy as np
import cv2
import time
from skimage.measure import compare_ssim
from util.perception import get_diff_images


def ex1():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        #img = cv2.imread("CMEaA.jpg")

        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #rgb =
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red = np.array([115, 100, 100])
        upper_red = np.array([130, 255, 255])

        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        #mask2 = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))
        #mask2 = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))

        red = cv2.bitwise_and(img, img, mask=mask2)

        grayImage = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # filter for length
        contours = [x for x in contours if len(x) < 50]
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def ex2():

    cap = cv2.VideoCapture(0)
    ret, background = cap.read()

    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    while True:
        cv2.imshow('frame', background)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # get some n first pictures to set background
    n = 2
    for i in range(n):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        weight = (i+1) / (i+2)
        print(weight)
        background = cv2.addWeighted(background, weight, frame, 1-weight, 0)
        time.sleep(0.5)

    # convert background to gray
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    print(background.shape)

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(background, frame, full=True)
        diff = (diff * 255).astype("uint8")

        #print("SSIM: {}".format(score))

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cv2.imshow('frame', diff)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":

    ex2()
