import cv2 as cv
import numpy as np
import time


COLOR_WINDOW_NAME = "Find Pen Color"


def emptyFunction(x):
    """ helper empty function for trackbars """
    pass

def init_color_window():
    cv.namedWindow(COLOR_WINDOW_NAME)
    cv.createTrackbar('Lower H', COLOR_WINDOW_NAME, 110, 180, emptyFunction) 
    cv.createTrackbar('Lower S', COLOR_WINDOW_NAME, 120, 255, emptyFunction) 
    cv.createTrackbar('Lower V', COLOR_WINDOW_NAME, 120, 255, emptyFunction)
    cv.createTrackbar('Upper H', COLOR_WINDOW_NAME, 130, 180, emptyFunction) 
    cv.createTrackbar('Upper S', COLOR_WINDOW_NAME, 255, 255, emptyFunction) 
    cv.createTrackbar('Upper V', COLOR_WINDOW_NAME, 255, 255, emptyFunction)

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    init_color_window()

    while True:
        ret, frame = cap.read()
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lowerH = cv.getTrackbarPos('Lower H', COLOR_WINDOW_NAME) 
        lowerS = cv.getTrackbarPos('Lower S', COLOR_WINDOW_NAME) 
        lowerV = cv.getTrackbarPos('Lower V', COLOR_WINDOW_NAME) 
        upperH = cv.getTrackbarPos('Upper H', COLOR_WINDOW_NAME) 
        upperS = cv.getTrackbarPos('Upper S', COLOR_WINDOW_NAME) 
        upperV = cv.getTrackbarPos('Upper V', COLOR_WINDOW_NAME) 
        lower = np.array([lowerH,lowerS,lowerV])
        upper = np.array([upperH,upperS,upperV])

        mask = cv.inRange(hsv, lower, upper)
        res = cv.bitwise_and(frame,frame, mask= mask)
        mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        stacked = np.hstack((mask_3,frame,res))
        cv.imshow(COLOR_WINDOW_NAME, stacked)

        key = cv.waitKey(1)
        if key == 27: # ESC 
            break
        elif key == ord('s'):
            savedColor = np.array([[lowerH,lowerS,lowerV],[upperH, upperS, upperV]])
            print("Found pen color: ", savedColor)
            break

    cap.release()
    cv.destroyAllWindows()
    

if __name__=="__main__": 
    main()