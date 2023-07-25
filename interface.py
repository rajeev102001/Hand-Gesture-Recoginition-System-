import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from tensorflow.keras import models
from keytotext import pipeline
import pyttsx3
text_speech = pyttsx3.init()
nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")
params = {"do_sample":False, "num_beams":2, "no_repeat_ngram_size":3, "early_stopping":True}

model1 = models.load_model('Action_NPY_Hap_Bod.h5')
model2 = models.load_model('Action_NPY_WATR_AER.h5')
model3 = models.load_model('Action_NPY_I_YO.h5')
model4 = models.load_model('Action_NPY_Wat_Wav.h5')
model5 = models.load_model('Action_NPY_LI_GI.h5')

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return (np.concatenate([pose, lh, rh])),pose

def section(pose_point,points):
    posel =np.asarray(pose_point).reshape(20,33,4)

    if posel[16][15][1]<(3/4)*posel[16][23][1] and posel[16][16][1]<(3/4)*posel[16][24][1]:# double hand 
        actions = np.array(['happy','body'])
        Y = model1.predict(points) 
        print(Y.max())
      
        return (actions[np.argmax(Y)])
                
    elif posel[18][15][1]<(3/4)*posel[18][23][1] :
        if posel[18][15][1]<posel[18][11][1] and posel[18][16][1] >posel[18][12][1]: #single hand
            actions = np.array(['water','aeroplane'])
            Y = model2.predict(points) 
            print(Y.max())
            
            return (actions[np.argmax(Y)])

        if posel[18][12][0]<posel[18][15][0]<posel[18][11][0] and posel[18][11][1]<posel[18][15][1]<posel[18][23][1]:
            actions = np.array(['I','you'])
            Y = model3.predict(points) 
            print(Y.max())
           
            return (actions[np.argmax(Y)])
            
        if posel[18][12][0]>posel[18][15][0] and posel[18][11][1]<posel[18][15][1]<posel[18][23][1]:
            actions = np.array(['watch','waves'])
            Y = model4.predict(points) 
            print(Y.max())
           
            return (actions[np.argmax(Y)])
          
        if posel[18][11][0]<posel[18][15][0] and posel[18][11][1]<posel[18][15][1]<posel[18][23][1]:
            actions = np.array(['like','girl'])
            Y = model5.predict(points) 
            print(Y.max())
           
            return (actions[np.argmax(Y)])
            
#Like 
def Ready(flag,img):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4  
    h, w, c = img.shape
    HAND_RESULTS = hands.process(img)
    
    if HAND_RESULTS.multi_hand_landmarks:
        for hand_landmark in HAND_RESULTS.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            finger_fold_status = []
            for tip in finger_tips:
                x, y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)
             
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                if lm_list[tip].x > lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)
      
            if all(finger_fold_status):
                if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y:
                    #print("LIKE")
                    flag=1
                   
                    cv2.putText(img, "Start", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                if lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y:
                    #print("DISLIKE")
                    flag=-1
                    cv2.putText(img, "Restart", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                if lm_list[8].y > lm_list[8-2].y and lm_list[8].y > lm_list[12].y :
        
                    flag=2
                    cv2.putText(img, "Predicting....", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
    
    return flag,img

   #capturing
def start_cap(frame_num,image):
    
    image ,results = mediapipe_detection(image,holistic)
    draw_styled_landmarks(image,results)
    keypoints,pose = extract_keypoints(results)
    points.append(keypoints)
    pose_point.append(pose)
        
    
    if frame_num == 1: 
                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames {} '.format(frame_num), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
             
                cv2.waitKey(1000)
    else: 
        cv2.putText(image, 'Collecting frames  {} '.format(frame_num), (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
  
        cv2.waitKey(100)    
    return image

st.sidebar.title('Minor Project B.Tech 2021-2022')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Demo'])

if app_mode == 'Home':
    st.title('About Our Project')
    st.markdown("The goal of this research is to present and develop a technique for hand gesture recognition that incorporates MediaPipe for extracting hand landmarks and LSTM to train and recognize the gesture.")
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSU0hBbUa_k1xKWJETgMkshQHnhUd9nu4RfHw&usqp=CAU',caption='Hand Gesture Recognition',use_column_width=True,)
elif app_mode == 'Demo':
    header = st.container()
    
    FRAME_WINDOW = st.image([])
    with header:
        st.title('Hand Gesture Detection')
        st.text("""This app detects the Sign Language of Deaf and Dumb person!""")
    
    run = st.checkbox('Ready')
    cam = cv2.VideoCapture(0)
    points =[]
    pose_point =[]
    sen=[]
    flag=0
    flag2 =0
    count=0
    text =''
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        if run:
            while True:
    
                SUCCESS, frame = cam.read()
                frame = cv2.flip(frame, 1)
                image = frame
    
                if flag==0:
                    flag,frame = Ready(flag,frame)
        
                if len(text)>1:
                    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if flag2 ==0:
                        text_speech.say(text)
                        text_speech.runAndWait()
                        flag2=1
    
            
                if flag==1 and count<=20:
                    count+=1
                    
    
                    frame =start_cap(count,frame)
                    if count==20:
                        flag=0
                        count=0
                        points_arr =np.asarray(points).reshape(-1,20,258)
                        word =section(pose_point,points_arr)
                        if word!=None: 
                            sen.append(word)
                            print(sen)
                            points =[]
                            pose_point =[]
                            text=[]
                                            
                elif flag==-1 :  
                    sen=sen[:-1]
                    flag=0
                elif flag==2:
                    S =nlp(sen,**params)
                    p=len(S)
                    for i in range(p):
                        if S[i]=='.':
                            p=i
                            break
                    text =S[:i]
                    print(text)
                    flag=0
                    sen =[]
                
                FRAME_WINDOW.image(frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
