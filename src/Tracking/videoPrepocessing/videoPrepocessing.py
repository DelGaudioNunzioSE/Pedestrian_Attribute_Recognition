import cv2
import numpy as np
import torch
from ultralytics import YOLO

# our imports
from playVideo import play_video_with_fps




    

def uniform_brightness_rgb(frame, threshold=[100,100,100], incrise=1.5 ,decrise=0):
    
    # Applica la soglia a ciascun canale
    mask0 = frame[:,:,0] < threshold[0]  # Maschera per il canale rosso
    mask1 = frame[:,:,1] < threshold[1]  # Maschera per il canale verde
    mask2 = frame[:,:,2] < threshold[2]  # Maschera per il canale blu

    # Applica la trasformazione solo ai pixel sotto la soglia per ogni canale
    frame[:,:,0] = np.where(mask0, cv2.convertScaleAbs(frame[:,:,0], alpha=incrise, beta=decrise), frame[:,:,0])
    # Canale verde
    frame[:,:,1] = np.where(mask1, cv2.convertScaleAbs(frame[:,:,1], alpha=incrise, beta=decrise), frame[:,:,1])
    # Canale blu
    frame[:,:,2] = np.where(mask2, cv2.convertScaleAbs(frame[:,:,2], alpha=incrise, beta=decrise), frame[:,:,2])

    frame[:,:,0] = np.clip(frame[:,:,0], 0, 255)
    frame[:,:,1] = np.clip(frame[:,:,1], 0, 255)
    frame[:,:,2] = np.clip(frame[:,:,2], 0, 255)
    
    return frame


def easy_uniform_brightness_rgb(frame, threshold=[100,100,100], incrise=1.2 ,decrise=0):
    
    # Applica la soglia a ciascun canale

    # Applica la trasformazione solo ai pixel sotto la soglia per ogni canale
    frame[:,:,0] =  cv2.convertScaleAbs(frame[:,:,0], alpha=incrise, beta=decrise)
    # Canale verde
    frame[:,:,1] = cv2.convertScaleAbs(frame[:,:,1], alpha=incrise, beta=decrise)
    # Canale blu
    frame[:,:,2] = cv2.convertScaleAbs(frame[:,:,2], alpha=incrise, beta=decrise)

    frame[:,:,0] = np.clip(frame[:,:,0], 0, 255)
    frame[:,:,1] = np.clip(frame[:,:,1], 0, 255)
    frame[:,:,2] = np.clip(frame[:,:,2], 0, 255)
    
    return frame

def histo(img_rgb):
        r, g, b = cv2.split(img_rgb)

        # Equalizzazione dell'istogramma per ciascun canale
        r_equalized = cv2.equalizeHist(r)
        g_equalized = cv2.equalizeHist(g)
        b_equalized = cv2.equalizeHist(b)

        # Ricompone l'immagine a colori con i canali equalizzati
        img_equalized = cv2.merge([r_equalized, g_equalized, b_equalized])

        # Converti l'immagine back a BGR per il salvataggio
        img_equalized_bgr = cv2.cvtColor(img_equalized, cv2.COLOR_RGB2BGR)

        return img_equalized_bgr





TRANSFORM = [
    easy_uniform_brightness_rgb,  # Applica luminosità uniforme per canale RGB
]

video_path = "./src/Tracking/videos/Atrio.mp4"
play_video_with_fps(video_path, 120, transformations= TRANSFORM, output_path="./src/Tracking/videos/Atrio_bright.mp4")



