import cv2 as cv 
from deepface import DeepFace
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
from emotes import * 

# Accessing the hand object 
mp_hands = mp.solutions.hands 


# Drawing the hand landmarks 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles 

# All Emote detector functions 

# Face detection using HaarCascades 
detector = cv.CascadeClassifier('haarcascade_frontalFace_default.xml')

# Screen Capture
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
        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                if result.multi_handedness and i < len(result.multi_handedness):
                    hand_type = result.multi_handedness[i].classification[0].label
                else:
                    hand_type = "Unknown"

                if hand_type == 'Right':
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                else:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )


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