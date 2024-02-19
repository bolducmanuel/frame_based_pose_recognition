import numpy as np
import sys
import cv2 as cv
from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__name__)

def get_image(gesture_name):
    """ 
    utility function for quickly gathering data in a data set. 
    Change the class name each time you wish to collect data 
    for a different gesture. 
    """
    Class = gesture_name
    Path('DATASET/'+Class).mkdir(parents=True, exist_ok=True) #TODO: replace pictures from directory if directory already exists
    cap = cv.VideoCapture(0) #TODO : add argument for changing camera
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0    
    while True:
       
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv.flip(frame,1)
        i+= 1
        if i % 5==0: # TODO: add argument for changer picture frequency
            cv.imwrite('DATASET/'+Class+'/'+str(i)+'.png',frame)
      
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q') or i > 500: #TODO 500 frame tolerance for data training. Can be changed as desired
            break
  
    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gesture", default="", type=str, help="dataset name")

    args = parser.parse_args()

    if args.gesture == "":
        logger.error(f"please specify a gesture name")
        sys.exit(1)

    

    get_image(args.gesture)
  