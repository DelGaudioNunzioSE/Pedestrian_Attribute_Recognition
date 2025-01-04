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
import time




# path_corrente = os.getcwd()
# print(f"Il path corrente è: {path_corrente}")

#dataset
# Struttura dei dati
data = {
    "people": {},  # Dizionario con ID come chiavi e dati della persona come valori
    "trajectory":[]
}

line_dict={
    "line":{}
}

probs = {"people": {}
}

lines ={
    
}

MODEL = "HistogramEqualization_512_neurons_7_01_0818.pth"




class HistogramEqualization:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input deve essere un'immagine PIL.Image")
        return ImageOps.equalize(img)
#Dall'immagine del bounding box all'input della rete

    
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
    #transforms.Resize((90, 200)),
    transforms.Resize((224, 224)),
    CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




def calculate_crossing(box, box2, center, arrowEnd):
    vet_track = np.array([box2[1] - box[1], box2[3] - box[3]])
    vet_line = np.array([arrowEnd[0] - center[0], arrowEnd[1] - center[1]])
    dot_product = np.dot(vet_track, vet_line)
    if dot_product > 0:
        return True
    else: 
        return False
    
# Funzione per calcolare l'orientamento (cross product) di tre punti
def orientation(p, q, r):
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

# Funzione per verificare se due segmenti si intersecano
def on_segment(p, q, r):
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

def do_intersect(p1, q1, p2, q2):
    x1,x2,y1,y2 = p1
    x3,x4,y3,y4 = q1
    
    p1= ((x1+x2)/2,y2)
    q1= ((x3+x4)/2,y4)


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


    
def has_gender_peak(probabilities, threshold=0.6, window_size=20):
    # Considera solo le ultime N predizioni
    recent_probs = probabilities[-window_size:]
    # Calcola la media delle probabilità recenti
    avg_prob = sum(recent_probs) / len(recent_probs)
    # Se la media supera la soglia, restituisce True (c'è lo zaino)
    return avg_prob > threshold

def has_bag_peak(probabilities, threshold=0.5, window_size=30):
    # Considera solo le ultime N predizioni
    recent_probs = probabilities[-window_size:]
    # Calcola la media delle probabilità recenti
    recent_probs = np.array(recent_probs)  # Converte la lista in un array NumPy
    recent_probs[recent_probs > 0.3] *= 2
    avg_prob = sum(recent_probs) / len(recent_probs)
    # Se la media supera la soglia, restituisce True (c'è lo zaino)
    return avg_prob > threshold

    
def has_hat_peak(probabilities, threshold=0.3, window_size=30):
    # Considera solo le ultime N predizioni
    recent_probs = probabilities[-window_size:]
    # Calcola la media delle probabilità recenti
    avg_prob = sum(recent_probs) / len(recent_probs)
    # Se la media supera la soglia, restituisce True (c'è lo zaino)
    return avg_prob > threshold


config={
    "x_real":[],
    "y_real":[]
}

def get_config(file_path):
    with open(file_path, 'r') as f:
        f_config = json.load(f)
        lines=f_config["lines"]                     #Per disegnare linee 
        for line in f_config["lines"]:              #Per proiettare punti 
            config["x_real"].append(line["x1"])
            config["y_real"].append(line["y1"])
            config["x_real"].append(line["x2"])
            config["y_real"].append(line["y2"])

        config["z_real"] = np.zeros_like(config["x_real"])
        config["xc"] = f_config["xc"]
        config["yc"] = f_config["yc"]
        config["zc"] = f_config["zc"]
        config["thyaw"] = f_config["thyaw"] 
        config["thpitch"] = f_config["thpitch"]
        config["throll"] = f_config["throll"] 
        config["U"] = f_config["U"]  # Larghezza immagine (pixel)
        config["V"] = f_config["V"]  # Altezza immagine (pixel)
        config["f"] = f_config["f"]
        config["s_w"] = f_config["sw"]
        config["s_h"] = f_config["sh"]
        print(config)
    return config, lines
           
def getPoints(config_path='./src/config/config.json'):
    config,lines = get_config(config_path)

    return inversion_points(
        x_real=config["x_real"],
        y_real=config["y_real"],
        z_real = config["z_real"],
        camera_x=config["xc"],
        camera_y=config["yc"],
        camera_z=config["zc"],
        thyaw=config["thyaw"],
        thpitch=config["thpitch"],
        throll=config["throll"],
        focal=config["f"],
        resolution_x=config["U"],
        resolution_y=config["V"],
        sensor_x=config["s_w"],
        sensor_y=config["s_h"]
    ), lines


def drawLine(frame,p1,p2, i):
    cv2.line(frame, p1,p2, color=(255, 0, 0), thickness=3)  # Cerchi rossi
    cx = (p1[0] + p2[0]) // 2
    cy = (p1[1] + p2[1]) // 2
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    length = np.sqrt(dx**2 + dy**2) 
    unit_dx = dx / length
    unit_dy = dy / length

    perp_dx = -unit_dy
    perp_dy = unit_dx
    arrowEnd=(int(cx+perp_dx*25),int(cy+perp_dy*25))
    cv2.circle(frame, p1 , 3,(0, 0, 255), thickness=3)
    cv2.circle(frame, p2 , 3,(0, 0, 255), thickness=3)
    cv2.arrowedLine(frame, (cx, cy), arrowEnd, (255, 0, 0), thickness=3)
    if(p1[0]<p2[0]):
        cv2.putText(frame, str(i),(p1[0],p1[1]-25),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
    else:
        cv2.putText(frame, str(i),(p2[0],p2[1]-25),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
    new_line = {        "id": i,
                        "p1": p1,
                        "p2": p2,
                        "center": (cx,cy),
                        "arrowEnd": arrowEnd
                    }
    line_dict["line"][new_line["id"]]= new_line
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
    checkpoint = torch.load('./src/Classifier/Models/'+MODEL)
    classifier_model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move the model to the selected device
    classifier_model.to(device)

    # Confirm the device of the model
    print(f"The model is loaded on: {next(model.parameters()).device}")

    # Run tracking with the specified tracker configuration file
    results = model.track(source=video_path, show=False, tracker=tracker, 
                          stream=True, classes=0, imgsz = (1280,1920), vid_stride=8, conf=0.3
                          ,iou = 0.8, max_det=30, persist=True, half=True)
    
    #video, visualizza mentre elabora, parametri del tracker, stream = risultati in tempo reale
    #1920x1080 riesce a prendersi il ragazzo dietro
    #ma forse possiamo usare 1280x720 e recuperarlo col detector
    #half migliora la velocità anche se abbassa un pò l'accuracy


    # Prendo image prima solo per disegnare le linee
    
    
    points,lines=getPoints()
    
    gender={}
    hat={}
    bag={}

    extra_width = 30  
    line_height = 20
    trajectory=(len(points)//2)
    for i,result in enumerate(results):
        if(i%2==0):
            frame = result.orig_img.copy()
            frame_original = result.orig_img  # Immagine originale del frame
            traj=0
            
            for j,line in enumerate(lines):  # Itera con passi di 2
                frame = drawLine(frame,points[2*j],points[2*j+1], line["id"])
            # Itera su ogni bounding box e ID
            if result.boxes.xyxy is not None and result.boxes.id is not None:
                for box, id in zip(result.boxes.xyxy, result.boxes.id):
                    x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                    img = drawBox(box, frame_original,device)  # Prepara input per la rete di classificazione
                    x1_extended = x1 - extra_width  # Aggiungi margine a sinistra
                    x2_extended = x2 + extra_width  # Aggiungi margine a destra
                    # Classificazioni (gender, hat, bag)
                    #(vid stride 8/ 2 primo if/ 5 secondo if) 71 sec
                    #(vid stride 8/ 2 primo if/ 5 secondo if) 76 sec
                    if(i%3==0):
                        gender[id.item()], bag[id.item()], hat[id.item()] = classifier_model(img)
                    if(id.item() in gender):
                        gender_pred = torch.sigmoid(gender[id.item()])  #0.4
                        hat_pred = torch.sigmoid(hat[id.item()])
                        bag_pred = torch.sigmoid(bag[id.item()]) #0.3
                    else:
                        gender_pred = torch.sigmoid(torch.tensor(0))  #0.4
                        hat_pred = torch.sigmoid(torch.tensor(0))
                        bag_pred = torch.sigmoid(torch.tensor(0)) #0.3

                   
                    # Disegna il bounding box sul frame
                    
                    box_x, box_y, box_width, box_height = 10, 10, 250, 33*(trajectory+1)  
                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # qua disegna i bounding_box
                    cv2.rectangle(frame,(x1,y1),(x1+20,y1+20),(255,255,255),-1)
                    text_box_height = 70
                    x1_extended = max(x1_extended, 0)
                    x2_extended = min(x2_extended, frame.shape[1])
                    cv2.rectangle(frame, (x1_extended, y2), (x2_extended, y2 + text_box_height), (255, 255, 255), -1)  # Box bianco
                    if(id.item() in data["people"]):
                        person=data["people"][id.item()]
                        for line in lines:
                            if(do_intersect(person["xyxy"],(x1,x2,y1,y2),line_dict["line"][line["id"]]["p1"],line_dict["line"][line["id"]]["p2"])):
                                if(calculate_crossing(person["xyxy"],(x1,x2,y1,y2),line_dict["line"][line["id"]]["center"],line_dict["line"][line["id"]]["arrowEnd"])):
                                    traj=line["id"]
                       

                    
                            
                           
                    # Crea un dizionario con le informazioni
                    
                    new_person = {
                        "id": id.item(),
                        "gender": gender_pred.item(),
                        "hat": hat_pred.item(),
                        "bag": bag_pred.item(),
                        "trajectory": [],
                        "xyxy":(x1,x2,y1,y2)
                    }
                    if new_person["id"] in data["people"]:
                        # Aggiorna la persona esistente
                        person = data["people"][new_person["id"]]
                        person["gender"] = new_person["gender"]
                        person["hat"] = new_person["hat"]
                        person["bag"] = new_person["bag"]
                        person["xyxy"] = new_person["xyxy"]

                        prs = probs["people"][new_person["id"]]
                        if(prs["gender_pred"][-1]!= new_person["gender"]):
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
                        
                        if(len(person["trajectory"])>0):
                            if(person["trajectory"][-1]!=traj and traj!=0):
                                person["trajectory"].append(traj)
                                final_f["people"][int(id.item()-1)]["trajectory"].append(traj)
                        elif traj!=0:
                            person["trajectory"].append(traj)
                            final_f["people"][int(id.item()-1)]["trajectory"].append(traj)

                        id_text=f"{int(person['id'])}"
                        gender_text = f"Gender: {'M' if person['gender'] == 'Male' else 'F'}"
                        hat_bag_text = f"Hat: {'Yes' if person['hat'] == 'Yes' else 'No'} | Bag: {'Yes' if person['bag'] == 'Yes' else 'No'}"
                        if(person['trajectory'] is not None):
                            trajectory_text = f"Trajectory: {person['trajectory'] }"
                        else:
                            trajectory_text = f"Trajectory: []"
                        cv2.putText(frame,id_text,(x1,y1+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                        cv2.putText(frame, gender_text, (x1_extended, y2 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, hat_bag_text, (x1_extended, y2 + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, trajectory_text, (x1_extended, y2 + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

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
                        if(new_person['trajectory'] is not None):
                            trajectory_text = f"Trajectory: {new_person['trajectory'] }"
                        else:
                            trajectory_text = f"Trajectory: []"
                        cv2.putText(frame, id_text,(x1,y1+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                        cv2.putText(frame, gender_text, (x1_extended, y2 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, hat_bag_text, (x1_extended, y2 + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(frame, trajectory_text, (x1_extended, y2 + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    traj= 0
                total_people= len(result)
                if box_x is not None:
                    cv2.putText(frame, f"Total People: {total_people}", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                for i,line in enumerate(lines):
                    id=line["id"]
                    e=sum(1 for person in data["people"].values() if id in person["trajectory"])
                    cv2.putText(frame, f"Trajectory {id}: {e}", (box_x + 10, box_y + 30*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
            # Mostra il frame con i risultati
            frame= cv2.resize(frame,(1280,720))
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
test_path='./src/Tracking/videos/Atrio.mp4'
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
file_path = './src/Tracking/videos/results.txt'
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(final, file, indent=4, ensure_ascii=False)  # indent=4 per rendere leggibile, ensure_ascii=False per caratteri non ASCII
    print(f"File salvato in {file_path}")




