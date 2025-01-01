import math
import cv2
import torch
from ultralytics import YOLO  # Ensure you have the Ultralytics YOLO library installed
import os
import json
from torchvision import transforms
from Classifier.classifier import CNNWithAttention
from PIL import Image
from Projection.projectionFunctions import *




# path_corrente = os.getcwd()
# print(f"Il path corrente è: {path_corrente}")

#dataset
# Struttura dei dati
data = {
    "people": {},  # Dizionario con ID come chiavi e dati della persona come valori
    "trajectory":[]
}

#Dall'immagine del bounding box all'input della rete
transform= transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Funzione per calcolare l'orientamento (cross product) di tre punti
def orientation(p, q, r):
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

# Funzione per verificare se due segmenti si intersecano
def on_segment(p, q, r):
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

def do_intersect(p1, q1, p2, q2):
    # Calcolare le 4 orientazioni
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # Generale caso
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    
    # Casuali casi di collineari
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    return False


def getPoints(frame):
    # Parametri per test1
    x_real = np.array([-4.6, -4.6, 4.6, 4.6])
    y_real = np.array([9, 14.00, 9, 14.00])
    z_real = np.zeros_like(x_real)
    xt= 0
    yt= 0
    zt= 7.20  # Coordinate della camera
    thyaw = 0* np.pi / 180  # Z
    thpitch = (((360-32) * np.pi) / 180)  # X
    throll = 0 * np.pi / 180  # Y
    f = 0.003  # Distanza focale (m)
    s_w = 0.00498  # Larghezza sensore (m)
    s_h = 0.003  # Altezza sensore (m)
    U = frame.shape[1]   # Larghezza immagine (pixel)
    V = frame.shape[0]  # Altezza immagine (pixel)
    return inversion_points(x_real=x_real, y_real=y_real, z_real=z_real, camera_x=xt, camera_y=yt, camera_z=zt, thyaw=thyaw, thpitch=thpitch, throll=throll, focal=f, resolution_x=U, resolution_y=V, sensor_x=s_w, sensor_y=s_h)

def drawLine(frame,p1,p2,orientation):
    cv2.line(frame, p1,p2, color=(255, 0, 0), thickness=1)  # Cerchi rossi
    cx = (p1[0] + p2[0]) // 2
    cy = (p1[1] + p2[1]) // 2
    center = (cx, cy)
     # Calcola la pendenza della retta tra P1 e P2
    if p1[0] - p2[0] != 0:
        m12 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        m_perp = -1 / m12 
    else:
        m_perp = 0 
    dx = 30 / math.sqrt(1 + m_perp**2)
    dy = m_perp * dx
    
    
    if(orientation):
        third_point = (int(cx + dx), int(cy + dy))  # Terzo punto sopra
    else:
        third_point = (int(cx - dx), int(cy - dy))  # Terzo punto sotto
    cv2.arrowedLine(frame, center, third_point, color=(255, 0, 0), thickness=2)
    
    return frame
def drawBox(box, frame,device):
    x1, y1, x2, y2 = map(int, box[:4])
    # Ritaglia il bounding box dall'immagine
    cropped_img = frame[y1:y2, x1:x2]
    cropped_img = Image.fromarray(cropped_img)
    cropped_img = transform(cropped_img) #la trasformo per darla in input alla rete
    cropped_img = cropped_img.unsqueeze(0).to(device)
    return cropped_img


def my_track(video_path, tracker, show=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")


    # Load YOLO model with weights onto the selected device
    model = YOLO('./src/Tracking/yolov8m.pt')
    classifier_model = CNNWithAttention()   
    checkpoint = torch.load('./src/Classifier/Models/checkpoint_9_31_1925.pth')
    classifier_model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move the model to the selected device
    classifier_model.to(device)

    # Confirm the device of the model
    print(f"The model is loaded on: {next(model.parameters()).device}")

    # Run tracking with the specified tracker configuration file
    results = model.track(source=video_path, show=False, tracker=tracker, stream=True, classes=0, imgsz = (1920,1080), vid_stride=15
                          ,iou = 0.9) #video, visualizza mentre elabora, parametri del tracker, stream = risultati in tempo reale
    
    image=next(results)
    frame=image.orig_img
    points=getPoints(frame)
    for result in results:
        frame = result.orig_img  # Immagine originale del frame
        trajectory=0
        orientation=True
        for i,point in enumerate(points):
           if(i%2==1):
            frame=drawLine(frame,point,point_temp,orientation)
            orientation=False
           point_temp=point
        # Itera su ogni bounding box e ID
        for box, id in zip(result.boxes.xyxy, result.boxes.id):
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            img = drawBox(box, frame,device)  # Prepara input per la rete di classificazione
            
            # Classificazioni (gender, hat, bag)
            gender, hat, bag = classifier_model(img)
            gender_pred = torch.sigmoid(gender) > 0.5
            hat_pred = torch.sigmoid(hat) > 0.5
            bag_pred = torch.sigmoid(bag) > 0.5
            # Disegna il bounding box sul frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if(do_intersect((x1,y1),(x2,y2),points[0],points[1])):
                trajectory=1
            if(do_intersect((x1,y1),(x2,y2),points[2],points[3])):
                trajectory=1
                    
            # Crea un dizionario con le informazioni
            
            new_person = {
                "id": id.item(),
                "gender": "Female" if gender_pred else "Male",
                "hat": "Yes" if hat_pred else "No",
                "bag": "Yes" if bag_pred else "No",
                "trajectory": [trajectory]
            }
            if new_person["id"] in data["people"]:
                # Aggiorna la persona esistente
                person = data["people"][new_person["id"]]
                person["gender"] = new_person["gender"]
                person["hat"] = new_person["hat"]
                person["bag"] = new_person["bag"]
                # Aggiungi la nuova traiettoria
                if(person["trajectory"][-1]!=trajectory):
                    person["trajectory"].append(trajectory)
                text = f"ID: {person["id"]} | G: {person['gender']} | Hat: {person['hat']} | Bag: {person['bag'] }| trajectory:{person['trajectory']}"
                 # Posiziona il testo sopra il bounding box
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Aggiungi una nuova persona
                data["people"][new_person["id"]] = new_person
                text = f"ID: {new_person['id']} | G: {new_person['gender']} | Hat: {new_person['hat']} | Bag: {new_person['bag'] }| trajectory:{new_person['trajectory']}"
                # Posiziona il testo sopra il bounding box
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            trajectory=0
        # Mostra il frame con i risultati
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()




video_path = './src/Tracking/videos/Atrio.mp4' # Path to the input video file (`video_fish.mp4`)
tracker='./src/Tracking/confs/botsort.yaml' # Path to the tracker configuration file (`botsort.yaml`)
show=True # A boolean flag to display the processed video with tracked objects

my_track(video_path, tracker, show)

# Scrittura del file JSON
file_path = './src/Tracking/videos/data.json'
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)  # indent=4 per rendere leggibile, ensure_ascii=False per caratteri non ASCII
    print(f"File salvato in {file_path}")



