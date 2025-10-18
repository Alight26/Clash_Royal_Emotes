import cv2 as cv 
from deepface import DeepFace
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
from emotes import * 
import imageio 

# Accessing the hand object 
mp_hands = mp.solutions.hands 


# Drawing the hand landmarks 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles 

# All Emote gesture functions 

# Getting gif images 
happy_barb_path = "clash-royale-happy.gif"
happy_barb_frames = imageio.mimread(happy_barb_path)



# Face detection using HaarCascades 
detector = cv.CascadeClassifier('haarcascade_frontalFace_default.xml')

# Converting gif frames 
frames_bgr = []
for frame in happy_barb_frames:
        frames_bgr.append(cv.cvtColor(frame, cv.COLOR_RGB2BGR))

# Screen Capture
idx = 0
happy_displayed = False
cap = cv.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, img = cap.read()


        # For Face 
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 3)

        # For Hands 
        RGB_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)

        # Selecting the handedness 
        right_hand_landmarks = None 
        left_hand_landmarks = None 


        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                        
                if hand_label == "Right":
                    right_hand_landmarks = hand_landmarks
                    mp_drawing.draw_landmarks(
                        img, right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                elif hand_label == "Left":
                    left_hand_landmarks = hand_landmarks
                    mp_drawing.draw_landmarks(
                        img, left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )



        if right_hand_landmarks and left_hand_landmarks:
            right_landmark = right_hand_landmarks.landmark
            left_landmark = left_hand_landmarks.landmark     


            # Happy barbarian
        
            if right_landmark[12].y < right_landmark[9].y and left_landmark[12].y < left_landmark[9].y:
                happy_displayed = True
                    
            else:
                happy_displayed = False


            # Should show gif please work this time
            if happy_displayed:
                frame = frames_bgr[idx]
                cv.imshow('happy_barb', frame)
                idx = (idx + 1) % len(frames_bgr)

            if not happy_displayed:
                cv.destroyWindow("happy_barb")
        else:
            happy_displayed = False


        # For Emotion Detection
        for (x, y, w, h) in faces:

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Does the Facial Analysis 
            detection = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)

            # detects the most dominant emotion 
            emotion = detection[0]['dominant_emotion']

            cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 225), 2)
            cv.putText(img, emotion, (50, 50), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)

        img_flipped = cv.flip(img, 1)
        hand_flipped = cv.flip(RGB_frame, 1)

        hand_flipped_BGR = cv.cvtColor(hand_flipped, cv.COLOR_RGB2BGR)

        cv.imshow('hands', hand_flipped_BGR)
        cv.imshow('img', img_flipped)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break






