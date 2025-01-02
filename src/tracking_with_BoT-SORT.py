import math
import cv2
import torch
from ultralytics import YOLO  # Ensure you have the Ultralytics YOLO library installed
import os
import json
from torchvision import transforms
from Classifier.classifier import CNNWithAttention
from PIL import Image, ImageOps
from Projection.projectionFunctions import *

from finalFile import * #per il file finale






# path_corrente = os.getcwd()
# print(f"Il path corrente è: {path_corrente}")

#dataset
# Struttura dei dati
data = {
    "people": {},  # Dizionario con ID come chiavi e dati della persona come valori
    "trajectory":[]
}

probs = {"people": {}
}





class HistogramEqualization:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input deve essere un'immagine PIL.Image")
        return ImageOps.equalize(img)
#Dall'immagine del bounding box all'input della rete
    
def init(data, id):
    prs={"id" : id,
    "gender" : [],
    "bag" : [],
    "hat" : [],
    "trajectory" : []
    }
    while len(data["people"]) < id:
        data["people"].append(None)
    data["people"][id-1] = prs
    return data
    
class CLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    def __call__(self, img):
        img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_clahe = self.clahe.apply(img_gray)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

transform = transforms.Compose([
    CLAHE(),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



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


    
def has_gender_peak(probabilities, threshold=0.6, window_size=20):
    # Considera solo le ultime N predizioni
    recent_probs = probabilities[-window_size:]
    # Calcola la media delle probabilità recenti
    avg_prob = sum(recent_probs) / len(recent_probs)
    # Se la media supera la soglia, restituisce True (c'è lo zaino)
    return avg_prob > threshold

def has_bag_peak(probabilities, threshold=0.4, window_size=20, peak_threshold=0.8):
    recent_probs = probabilities[-window_size:]
    avg_prob = sum(recent_probs) / len(recent_probs)
    max_prob = max(recent_probs)

    return avg_prob > threshold or max_prob > peak_threshold

    
def has_hat_peak(probabilities, threshold=0.4, window_size=10):
    # Considera solo le ultime N predizioni
    recent_probs = probabilities[-window_size:]
    # Calcola la media delle probabilità recenti
    avg_prob = sum(recent_probs) / len(recent_probs)
    # Se la media supera la soglia, restituisce True (c'è lo zaino)
    return avg_prob > threshold



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

def drawBox(box, frame,device):
    x1, y1, x2, y2 = map(int, box[:4])
    # Ritaglia il bounding box dall'immagine
    cropped_img = frame[y1:y2, x1:x2]
    cropped_img = Image.fromarray(cropped_img).convert('RGB')
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
    classifier_model = CNNWithAttention(hidden_dim=512)   
    checkpoint = torch.load('./src/Classifier/Models/HistogramEqualization_512_neurons_7_01_0818.pth')
    classifier_model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move the model to the selected device
    classifier_model.to(device)

    # Confirm the device of the model
    print(f"The model is loaded on: {next(model.parameters()).device}")

    # Run tracking with the specified tracker configuration file
    results = model.track(source=video_path, show=False, tracker=tracker, 
                          stream=True, classes=0, imgsz = (1080,1920), vid_stride=7, conf=0.3
                          ,iou = 0.8, max_det=30, persist=True, half=True)
    
    #video, visualizza mentre elabora, parametri del tracker, stream = risultati in tempo reale
    #1920x1080 riesce a prendersi il ragazzo dietro
    #ma forse possiamo usare 1280x720 e recuperarlo col detector
    #half migliora la velocità anche se abbassa un pò l'accuracy


    # Prendo image prima solo per disegnare le linee
    image=next(results)
    frame=image.orig_img
    points=getPoints(frame)
    extra_width = 30  
    line_height = 20

    for i,result in enumerate(results):
        if i%2 == 0:
            frame = result.orig_img.copy()
            frame_original = result.orig_img  # Immagine originale del frame
            trajectory=0
            orientation=True
            for i,point in enumerate(points):
                if(i%2==1):
                    frame=drawLine(frame,point,point_temp,orientation)
                    orientation=False
                point_temp=point
            # Itera su ogni bounding box e ID
            if result.boxes.xyxy is not None and result.boxes.id is not None:
                for box, id in zip(result.boxes.xyxy, result.boxes.id):
                    x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                    img = drawBox(box, frame_original,device)  # Prepara input per la rete di classificazione
                    x1_extended = x1 - extra_width  # Aggiungi margine a sinistra
                    x2_extended = x2 + extra_width  # Aggiungi margine a destra

                    # Classificazioni (gender, hat, bag)
                    gender, bag, hat = classifier_model(img)
                    gender_pred = torch.sigmoid(gender)  #0.4
                    hat_pred = torch.sigmoid(hat)  
                    bag_pred = torch.sigmoid(bag) #0.3
                    # Disegna il bounding box sul frame

                    box_x, box_y, box_width, box_height = 10, 10, 250, 100  
                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # qua disegna i bounding_box
                    cv2.rectangle(frame,(x1,y1),(x1+20,y1+20),(255,255,255),-1)
                    text_box_height = 70
                    x1_extended = max(x1_extended, 0)
                    x2_extended = min(x2_extended, frame.shape[1])
                    cv2.rectangle(frame, (x1_extended, y2), (x2_extended, y2 + text_box_height), (255, 255, 255), -1)  # Box bianco

                    if(do_intersect((x1,y1),(x2,y2),points[0],points[1])):
                        trajectory=1
                    if(do_intersect((x1,y1),(x2,y2),points[2],points[3])):
                        trajectory=1
                            
                    # Crea un dizionario con le informazioni
                    
                    new_person = {
                        "id": id.item(),
                        "gender": gender_pred.item(),
                        "hat": hat_pred.item(),
                        "bag": bag_pred.item(),
                        "trajectory": [trajectory]
                    }
                    if new_person["id"] in data["people"]:
                        # Aggiorna la persona esistente
                        person = data["people"][new_person["id"]]
                        person["gender"] = new_person["gender"]
                        person["hat"] = new_person["hat"]
                        person["bag"] = new_person["bag"]

                        prs = probs["people"][new_person["id"]]

                        prs["gender_pred"].append(new_person["gender"])
                        prs["bag_pred"].append(new_person["bag"])
                        prs["hat_pred"].append(new_person["hat"])

            
                        bag_pred = "Yes" if has_bag_peak(prs["bag_pred"]) else "No"
                        person["bag"] = bag_pred

                        gender_pred = "Female" if has_gender_peak(prs["gender_pred"]) else "Male"
                        person["gender"] = gender_pred

                        hat_pred = "Yes" if has_hat_peak(prs["hat_pred"]) else "No"
                        person["hat"] = hat_pred

                        final_f = append(final_f, int(id.item()), gender_pred, bag_pred, hat_pred)

                        # Aggiungi la nuova traiettoria
                        if(person["trajectory"][-1]!=trajectory):
                            person["trajectory"].append(trajectory)
                            final_f["people"][int(id.item()-1)]["trajectory"].append(trajectory)

                        id_text=f"{int(person['id'])}"
                        gender_text = f"Gender: {person['gender']}"
                        hat_bag_text = f"Hat: {'Yes' if person['hat'] == 'Yes' else 'No'} | Bag: {'Yes' if person['bag'] == 'Yes' else 'No'}"
                        trajectory_text = f"Trajectory: {person['trajectory'] }"
                        cv2.putText(frame,id_text,(x1,y1+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                        cv2.putText(frame, gender_text, (x1_extended, y2 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, hat_bag_text, (x1_extended, y2 + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, trajectory_text, (x1_extended, y2 + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        total_people= len(data["people"])
                        cv2.putText(frame, f"Total People: {total_people}", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        cv2.putText(frame, f"Trajectory 1: {trajectory_1}", (box_x + 10, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        cv2.putText(frame, f"Trajectory 2: {trajectory_2}", (box_x + 10, box_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        trajectory_1 = sum(1 for person in data["people"].values() if 1 in person["trajectory"])
                        trajectory_2 = sum(1 for person in data["people"].values() if 2 in person["trajectory"])
                    else:
                        final_f = init(final, int(id.item())) #file finale
                        data["people"][new_person["id"]] = new_person

                        pr = {"id": id.item(), "gender_pred": [], "bag_pred": [], "hat_pred": []}
                        
                        probs["people"][new_person["id"]] = pr

                        prs = probs["people"][new_person["id"]]

                        prs["gender_pred"].append(new_person["gender"]) #file delle probabilità totali 
                        prs["bag_pred"].append(new_person["bag"])
                        prs["hat_pred"].append(new_person["hat"])


                        id_text=f"{int(new_person['id'])}"
                        gender_text = f"Gender: {new_person['gender']}"
                        hat_bag_text = f"Hat: {'Yes' if new_person['hat'] == 'Yes' else 'No'} | Bag: {'Yes' if new_person['bag'] == 'Yes' else 'No'}"
                        trajectory_text = f"Trajectory: {new_person['trajectory']}"    
                        cv2.putText(frame,id_text,(x1,y1+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                        cv2.putText(frame, gender_text, (x1_extended, y2 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, hat_bag_text, (x1_extended, y2 + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, trajectory_text, (x1_extended, y2 + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        total_people= len(data["people"])
                        cv2.putText(frame, f"Total People: {total_people}", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        trajectory_1 = sum(1 for person in data["people"].values() if 1 in person["trajectory"])
                        trajectory_2 = sum(1 for person in data["people"].values() if 2 in person["trajectory"])
                    trajectory=0
            # Mostra il frame con i risultati
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    return final_f


final = {
    "people" : []
}

video_path = './src/Tracking/videos/Atrio.mp4' # Path to the input video file (`video_fish.mp4`)
tracker='./src/Tracking/confs/botsort.yaml' # Path to the tracker configuration file (`botsort.yaml`)
show=True # A boolean flag to display the processed video with tracked objects

final_f = my_track(video_path, tracker, show)

# Scrittura del file JSON
file_path = './src/Tracking/videos/data.json'
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)  # indent=4 per rendere leggibile, ensure_ascii=False per caratteri non ASCII
    print(f"File salvato in {file_path}")

file_path = './src/Tracking/videos/probs.json'
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(probs, file, indent=4, ensure_ascii=False)  # indent=4 per rendere leggibile, ensure_ascii=False per caratteri non ASCII
    print(f"File salvato in {file_path}")

print(final_f)
final = classify(final_f)
file_path = './src/Tracking/videos/results.json'
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(final, file, indent=4, ensure_ascii=False)  # indent=4 per rendere leggibile, ensure_ascii=False per caratteri non ASCII
    print(f"File salvato in {file_path}")




