import imageio 
import cv2 as cv 
from deepface import DeepFace
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision

def sad_king(sad_king_frames, sad_displayed):

    king_idx = 0
    frames_bgr = []
    for frame in sad_king_frames:
        frames_bgr.append(cv.cvtColor(frame, cv.COLOR_RGB2BGR))

    frame = frames_bgr[king_idx]
    cv.imshow('sad_king', frame)
    king_idx = (king_idx + 1) % len(frames_bgr)
    
    if not sad_displayed:
        cv.destroyWindow('sad_king')
