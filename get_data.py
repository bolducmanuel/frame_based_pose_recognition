import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 

def image_processed(file_path):
    
    # reading the static image
    hand_img = cv2.imread(file_path)

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_pose = mp.solutions.pose

    # Initialize Hands
    pose = mp_pose.Pose(smooth_landmarks=True)

    # Results
    output = pose.process(img_flip)

    pose.close()

    try:
        data = output.pose_landmarks.landmark
        #print(data)

        clean = []

        for i,landmark in enumerate(data): 

            clean.append(float(landmark.x))
            clean.append(float(landmark.y))
            #clean.append(float(landmark.z))
            clean.append(float(landmark.visibility))


        return(clean)
        
    except:
        return(np.zeros([1,99], dtype=int)[0])

def make_csv():
    """
    Utility function for writing keypoint information in csv format for each frame of the dataset.

    TODO: frame notation of the dataset can be optimized by calling mediapipe pose only once at 
    the beginning of the for loop, rather than in an individual function such as in image_processed.  
    """
    
    mypath = 'DATASET'
    file_name = open('dataset.csv', 'a')

    for each_folder in os.listdir(mypath):
        if '._' in each_folder:
            pass

        else:
            for each_number in os.listdir(mypath + '/' + each_folder):
                if '._' in each_number:
                    pass
                
                else:
                    label = each_folder

                    file_loc = mypath + '/' + each_folder + '/' + each_number

                    data = image_processed(file_loc)
                    try:
                        for id,i in enumerate(data):
                            if id == 0:
                                print(i)
                            
                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(label)
                        file_name.write('\n')
                    
                    except:
                        file_name.write('0')
                        file_name.write(',')

                        file_name.write('None')
                        file_name.write('\n')
       
    file_name.close()
    print('Data Created !!!')

if __name__ == "__main__":
    make_csv()

