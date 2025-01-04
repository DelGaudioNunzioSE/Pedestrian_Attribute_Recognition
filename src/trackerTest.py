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


# path_corrente = os.getcwd()
# print(f"Il path corrente è: {path_corrente}")

#dataset
# Struttura dei dati
data = {
    "people": {},  # Dizionario con ID come chiavi e dati della persona come valori
    "trajectory":[]
}
id_trajectory = {
}

class HistogramEqualization:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input deve essere un'immagine PIL.Image")
        return ImageOps.equalize(img)
#Dall'immagine del bounding box all'input della rete
transform= transforms.Compose([transforms.Resize((224, 224)), HistogramEqualization(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



def calculate_crossing(box1, box2, p1, p2, direction):
    if direction:
        direction = "down"
    else:
        direction = "up"
    # Estrarre i punti
    (x1, y1, x2, y2) = box1
    (x3, y3, x4, y4) = box2
    x_start, y_start = p1
    x_end, y_end = p2

    # Calcolare i piedi dei bounding box (punto centrale in basso)
    px1, py1 = (x1 + x2) / 2, y2
    px2, py2 = (x3 + x4) / 2, y4

    # Coefficienti della linea
    A = y_end - y_start
    B = x_start - x_end
    C = x_end * y_start - x_start * y_end

    # Calcolare d per i piedi dei bounding box
    d1 = A * px1 + B * py1 + C
    d2 = A * px2 + B * py2 + C

    # Controllare attraversamento e direzione
    print("d1:", d1, "d2:", d2, "d1 * d2:", d1 * d2)
    
    if d1 * d2 < 0:  # Segni opposti: attraversa
        if direction == "down" and d1 > 0 and d2 < 0:
            print("Attraversa verso il basso")
            return True
        elif direction == "up" and d1 < 0 and d2 > 0:
            print("Attraversa verso l'alto")
            return True
    else:
        print("Non attraversa")
    return False



# Funzione per calcolare l'orientamento (cross product) di tre punti
def orientation(p, q, r):
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

# Funzione per verificare se due segmenti si intersecano
def on_segment(p, q, r):
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

def do_intersect(p1, q1, p2, q2,direction):
    # Calcolare le 4 orientazioni
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    # Generale caso
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    # Controllo collineare
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def get_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
        config["x_real"] = (config["x_real"])
        config["y_real"] = (config["y_real"])
        config["z_real"] = np.zeros_like(config["x_real"])
        config["thyaw"] = config["thyaw"] * np.pi / 180
        config["thpitch"] = (360 - config["thpitch"]) * np.pi / 180
        config["throll"] = config["throll"] * np.pi / 180
    return config

def getPoints(frame, config_path='./src/config.json'):
    config = get_config(config_path)
    U = frame.shape[1]  # Larghezza immagine (pixel)
    V = frame.shape[0]  # Altezza immagine (pixel)
    return inversion_points(
        x_real=config["x_real"],
        y_real=config["y_real"],
        z_real = config["z_real"],
        camera_x=config["xt"],
        camera_y=config["yt"],
        camera_z=config["zt"],
        thyaw=config["thyaw"],
        thpitch=config["thpitch"],
        throll=config["throll"],
        focal=config["f"],
        resolution_x=U,
        resolution_y=V,
        sensor_x=config["s_w"],
        sensor_y=config["s_h"]
    )


def drawLine(frame,p1,p2,orientation, i):
    cv2.line(frame, p1,p2, color=(255, 0, 0), thickness=1)  # Cerchi rossi
    cx = (p1[0] + p2[0]) // 2
    cy = (p1[1] + p2[1]) // 2
    center = (cx, cy)
     # Calcola la pendenza della retta tra P1 e P2
    if p1[0] - p2[0] != 0 and p1[1] - p2[1]!=0:
        m12 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        m_perp = -1 / m12 
    elif p1[0] - p2[0] == 0:
        m_perp = 0 
    else:
        m_perp=1
    dx = 30 / math.sqrt(1 + m_perp**2)
    dy = m_perp * dx
    
    
    if(orientation):
        third_point = (int(cx + dx), int(cy + dy))  # Terzo punto sopra
    else:
        third_point = (int(cx - dx), int(cy - dy))  # Terzo punto sotto
    cv2.arrowedLine(frame, center, third_point, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, str(i),(p1[0],p1[1]+15),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
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
                          ,iou = 0.7, max_det=30, persist=True, half=True)
    
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
    trajectory=(len(points)//2)

    for i,result in enumerate(results):
        if i%2 == 0:
            count_line=0
            frame = result.orig_img.copy()
            frame_original = result.orig_img  # Immagine originale del frame
            traj=0
            orientation=True
            for j in range(0, len(points) - 1, 2):  # Itera con passi di 2
                count_line+=1
                frame=drawLine(frame,points[j],points[j+1],orientation, count_line)
                orientation=not orientation
                
            # Itera su ogni bounding box e ID
            if result.boxes.xyxy is not None and result.boxes.id is not None:
                for box, id in zip(result.boxes.xyxy, result.boxes.id):
                    x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                    img = drawBox(box, frame_original,device)  # Prepara input per la rete di classificazione
                    x1_extended = x1 - extra_width  # Aggiungi margine a sinistra
                    x2_extended = x2 + extra_width  # Aggiungi margine a destra

                    # Classificazioni (gender, hat, bag)
                    gender, bag, hat = classifier_model(img)
                    gender_pred = torch.sigmoid(gender) > 0.4 #0.4
                    hat_pred = torch.sigmoid(hat) > 0.5
                    bag_pred = torch.sigmoid(bag) > 0.3 #0.3
                    # Disegna il bounding box sul frame

                    box_x, box_y, box_width, box_height = 10, 10, 250, 33*(trajectory+1)  
                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # qua disegna i bounding_box
                    cv2.rectangle(frame,(x1,y1),(x1+20,y1+20),(255,255,255),-1)
                    text_box_height = 70
                    x1_extended = max(x1_extended, 0)
                    x2_extended = min(x2_extended, frame.shape[1])
                    cv2.rectangle(frame, (x1_extended, y2), (x2_extended, y2 + text_box_height), (255, 255, 255), -1)  # Box bianco
                    
                    orientation=True
                    if(id.item() in data["people"]):
                        person=data["people"][id.item()]
                        for k in range(trajectory):
                            if(do_intersect((x1,y1),(x2,y2),points[k*2],points[2*k+1],orientation)):
                                if(calculate_crossing(person["xyxy"],(x1,x2,y1,y2),points[k*2],points[k*2+1],orientation)):
                                        traj=k+1
                                        
                            orientation= not orientation
                            
                            
                    # Crea un dizionario con le informazioni
                    
                    new_person = {
                        "id": id.item(),
                        "gender": "Female" if gender_pred else "Male",
                        "hat": "Yes" if hat_pred else "No",
                        "bag": "Yes" if bag_pred else "No",
                        "trajectory": [traj],
                        "xyxy": (x1,y1,x2,y2)
                    }
                    if new_person["id"] in data["people"]:
                        # Aggiorna la persona esistente
                        person = data["people"][new_person["id"]]
                        person["gender"] = new_person["gender"]
                        person["hat"] = new_person["hat"]
                        person["bag"] = new_person["bag"]
                        person["xyxy"] = new_person["xyxy"]
                        # Aggiungi la nuova traiettoria
                        if(person["trajectory"][-1]!=traj):
                            person["trajectory"].append(traj)
                        id_text=f"{int(person['id'])}"
                        gender_text = f"Gender: {person['gender']}"
                        hat_bag_text = f"Hat: {'Yes' if person['hat'] == 'Yes' else 'No'} | Bag: {'Yes' if person['bag'] == 'Yes' else 'No'}"
                        trajectory_text = f"Trajectory: {person['trajectory'] }"
                        cv2.putText(frame,id_text,(x1,y1+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                        cv2.putText(frame, gender_text, (x1_extended, y2 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, hat_bag_text, (x1_extended, y2 + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, trajectory_text, (x1_extended, y2 + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                    else:
                        data["people"][new_person["id"]] = new_person
                        id_text=f"{int(new_person['id'])}"
                        gender_text = f"Gender: {new_person['gender']}"
                        hat_bag_text = f"Hat: {'Yes' if new_person['hat'] == 'Yes' else 'No'} | Bag: {'Yes' if new_person['bag'] == 'Yes' else 'No'}"
                        trajectory_text = f"Trajectory: {new_person['trajectory']}" 
                        cv2.putText(frame, id_text,(x1,y1+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                        cv2.putText(frame, gender_text, (x1_extended, y2 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, hat_bag_text, (x1_extended, y2 + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, trajectory_text, (x1_extended, y2 + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    traj= 0
            total_people= len(result)
            cv2.putText(frame, f"Total People: {total_people}", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            for i in range (trajectory):
                e=sum(1 for person in data["people"].values() if i+1 in person["trajectory"])
                cv2.putText(frame, f"Trajectory {i+1}: {e}", (box_x + 10, box_y + 30*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Mostra il frame con i risultati
            frame= cv2.resize(frame,(1280,720))
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


