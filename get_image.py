import numpy as np
import sys
import cv2 as cv
from pathlib import Path
import argparse
import logging
import time

logger = logging.getLogger(__name__)

def get_image(gesture_name, identification):
    """ 
    utility function for quickly gathering data in a data set. 
    Change the class name each time you wish to collect data 
    for a different gesture. 
    """
    Class = gesture_name
    identity = identification
    Path('DATASET/'+Class).mkdir(parents=True, exist_ok=True) #TODO: replace pictures from directory if directory already exists
    cap = cv.VideoCapture(6) #TODO : add argument for changing camera
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0 
    time.sleep(1)   
    while True:
       
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv.flip(frame,1)
        i+= 1
        if i % 5==0: # TODO: add argument for changer picture frequency
            cv.imwrite('DATASET/'+Class+'/'+identity+str(i)+'.png',frame)
      
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q') or i > 50: #TODO 500 frame tolerance for data training. Can be changed as desired
            break
  
    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gesture", default="", type=str, help="dataset name")
    parser.add_argument("-i", "--identity", default="", type=str, help="id of gesture trainer")
    args = parser.parse_args()

    if args.gesture == "":
        logger.error(f"please specify a gesture name")
        sys.exit(1)

    if args.identity == "":
        logger.error(f"please specify a valid id")
        sys.exit(1)

    print("taking pictures in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")

    get_image(args.gesture, args.identity)
  