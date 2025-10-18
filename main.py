import cv2 as cv 
import mediapipe as mp 
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



# Converting gif frames 
frames_bgr = []
for frame in happy_barb_frames:
        frames_bgr.append(cv.cvtColor(frame, cv.COLOR_RGB2BGR))

# Screen Capture
i = 0
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


        # For Hands 
        RGB_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)

        # Selecting the handedness 
        right_hand_landmarks = None 
        left_hand_landmarks = None 


        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                hand_label = result.multi_handedness[i].classification[0].label
                
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

                    
                
                    if right_landmark[12].y < right_landmark[9].y and left_landmark[12].y < left_landmark[9].y:
                        happy_displayed = True

                            
                           
                    else:
                        happy_displayed = False
                else:
                    happy_displayed = False

                # Should show gif please work this time
                if happy_displayed:
                    frame = frames_bgr[i]
                    cv.imshow('happy_barb', frame)
                    i = (i + 1) % len(frames_bgr)

                else:
                    if cv.getWindowProperty("happy_barb", cv.WND_PROP_VISIBLE) >= 1:
                        cv.destroyWindow("happy_barb")
                

        img_flipped = cv.flip(img, 1)
        hand_flipped = cv.flip(RGB_frame, 1)

        hand_flipped_BGR = cv.cvtColor(hand_flipped, cv.COLOR_RGB2BGR)

        cv.imshow('hands', hand_flipped_BGR)
        cv.imshow('img', img_flipped)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break






