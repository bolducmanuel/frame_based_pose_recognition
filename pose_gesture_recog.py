import cv2
import numpy as np
import pickle
# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

import mediapipe as mp
mp_pose = mp.solutions.pose
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(smooth_landmarks=True) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    results = pose.process(image)

    try:
      data = results.pose_landmarks.landmark


      clean = []
      #accepted_keypoints = [0, 13, 14, 15, 16, 25, 26, 29, 30]
      for i,landmark in enumerate(data): 
      #    if i in accepted_keypoints:
            clean.append(float(landmark.x))
            clean.append(float(landmark.y))
            clean.append(float(landmark.visibility))


      # for i in without_garbage:
      #     i = i.strip()
      #     clean.append(i[2:])

    except:
      clean = np.zeros([1,99], dtype=int)

    #print(clean)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
     
    clean = np.array(clean)
    y_pred = svm.predict(clean.reshape(-1,99))
    print(y_pred)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 100)
    
    # fontScale
    fontScale = 3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 5
    
    # Using cv2.putText() method
    image = cv2.putText(image, str(y_pred[0]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()