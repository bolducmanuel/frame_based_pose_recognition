import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 
import logging
import sys

logger = logging.getLogger(__name__)

def image_processed(file_path, pose):
    
    # reading the static image
    hand_img = cv2.imread(file_path)

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    


    # Results
    output = pose.process(img_flip)



    try:
        data = output.pose_landmarks.landmark
        #print(data)

        bad_data = False

        clean = []

        for i,landmark in enumerate(data): 

            clean.append(float(landmark.x))
            clean.append(float(landmark.y))


            if (landmark.x > 1.0 or landmark.y > 1.0) or (landmark.x < 0. or landmark.y < 0.) : #ignore bad data

                bad_data = True

                break
            
        if bad_data:
            return
        else:
            return clean
        
    except:
        pass

def make_csv(pose):
    """
    Utility function for writing keypoint information in csv format for each frame of the dataset.

    TODO: frame notation of the dataset can be optimized by calling mediapipe pose only once at 
    the beginning of the for loop, rather than in an individual function such as in image_processed.  
    """
    
    mypath = 'DATASET'
    file_name = open('dataset.csv', 'a')

    num_classes = len(os.listdir(mypath))

    if num_classes < 2:
        logger.error(f"Not enough gestures trained. Please train at least two gestures.")
        sys.exit(1)



    for each_folder in os.listdir(mypath):

        good_frames = 0
        bad_frames = 0

        label = each_folder

        if '._' in each_folder:
            pass

        else:
            for each_number in os.listdir(mypath + '/' + each_folder):
                if '._' in each_number:
                    pass
                
                else:
                    

                    file_loc = mypath + '/' + each_folder + '/' + each_number

                    data = image_processed(file_loc, pose)
                    try:
                        for id,i in enumerate(data):
                            
                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(label)
                        file_name.write('\n')

                        good_frames += 1
                    
                    except:
                        bad_frames += 1
                        pass
                        # file_name.write('0')
                        # file_name.write(',')

                        # file_name.write('None')
                        # file_name.write('\n')
        
        print('gesture', label, ': Data was generated for', good_frames, 'frames, and ', bad_frames, 'frames were discarded') 
       
    file_name.close()
    print('Data Created !!!')

if __name__ == "__main__":

    # accessing MediaPipe solutions
    mp_pose = mp.solutions.pose

        # Initialize Hands
    pose = mp_pose.Pose(smooth_landmarks=True)


    make_csv(pose)

    pose.close()

