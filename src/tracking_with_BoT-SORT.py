import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO  # Ensure you have the Ultralytics YOLO library installed
import json
from torchvision import transforms
from Classifier.classifier import CNNWithAttention
from PIL import Image, ImageOps

from finalFile import * #per il file finale

# our imports
from Tracking.SupportScripts.init import init
from Tracking.SupportScripts.crossingDirection import calculate_crossing
from Tracking.SupportScripts.crossing import do_intersect#, orientation, on_segment
from Tracking.SupportScripts.configurationReading import getPoints
from Tracking.SupportScripts.drowLine import drawLine, line_dict
from Classifier.classifierTest import CNNWithAttention2





# Database forma
data = {
    "people": {},  # Dictionary with ID keys e dati della persona come valori
    "trajectory":[]
}



probs = {"people": {}
}

lines ={
    
}

MODEL = "_ciriprovo_1_try.pth"



####TRANSFORMS#######################################
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


#Christian hack###########################################################
def has_gender_peak(probabilities, threshold=0.35, window_size=20):
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
    #recent_probs[recent_probs > 0.3] *= 2
    avg_prob = sum(recent_probs) / len(recent_probs)
    # Se la media supera la soglia, restituisce True (c'è lo zaino)
    return avg_prob > threshold

    
def has_hat_peak(probabilities, threshold=0.5, window_size=30):
    # Considera solo le ultime N predizioni
    recent_probs = probabilities[-window_size:]
    # Calcola la media delle probabilità recenti
    avg_prob = sum(recent_probs) / len(recent_probs)
    # Se la media supera la soglia, restituisce True (c'è lo zaino)
    return avg_prob > threshold


#####


def classifier_preparer(box, frame, device):
    '''prepare the input for the classifier
        box-> the bounding box of the person
        frame-> the frame of the video
        device-> the device to use
    '''
    x1, y1, x2, y2 = map(int, box[:4])
    # Ritaglia il bounding box dall'immagine
    cropped_img = frame[y1:y2, x1:x2]
    cropped_img = Image.fromarray(cropped_img)
    cropped_img = transform(cropped_img) #la trasformo per darla in input alla rete
    cropped_img = cropped_img.unsqueeze(0).to(device)
    return cropped_img
####################################################################################



# the main function (video processing)
def my_track(video_path, tracker):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")


    # Load YOLO model with weights onto the selected device
    model = YOLO('./src/Tracking/yolov8m.pt')
    classifier_model = CNNWithAttention2()   
    checkpoint = torch.load('./src/Classifier/Models/'+MODEL)
    classifier_model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move the model to the selected device
    classifier_model.to(device)

    # Confirm the device of the model
    print(f"The model is loaded on: {next(model.parameters()).device}")

    # Run tracking with the specified tracker configuration file
    # verbose -> False to suppress the output
    # source -> Path to the input video file
    # show -> True to display the processed video with tracked objects (we use cv2.imshow)
    # tracker -> Path to the tracker configuration file (`botsort.yaml`)
    # stream -> True to display the results in real-time (True= don't wait the end of the video to show the results)
    # classes -> 0 track only person
    # imgsz -> Image size (pixels)
    # vid_stride -> Frame stride for video (how many frames to skip)
    # conf -> Object confidence threshold (you have tohigher to be a valid deteced object)
    # iou -> IOU threshold for NMS (Non-Maximum Suppression) (you have to lower to be a valid deteced object) (it takes the most confident one)
    # max_det -> Maximum number of detections per image
    # persist -> True to keep tracking between frames (afther desappear and reappear the object must mantain the same ID)
    # half -> True to use half precision (faster but less accurate)
    results = model.track(device = device, verbose= False, source=video_path, show=False, tracker=tracker, 
                          stream=True, classes=0, imgsz = (1080,1920), vid_stride=7, conf=0.3
                          ,iou = 0.8, max_det=30, persist=True, half=True)
    
    #video, visualizza mentre elabora, parametri del tracker, stream = risultati in tempo reale
    #1920x1080 riesce a prendersi il ragazzo dietro
    #ma forse possiamo usare 1280x720 e recuperarlo col detector
    #half migliora la velocità anche se abbassa un pò l'accuracy


    # Prendo image prima solo per disegnare le linee
    
    points,lines=getPoints()
    
    extra_width = 30  
    line_height = 20
    trajectory=(len(points)//2) #number of lines

    # Iteration on the detector frames
    for i,result in enumerate(results):
        if i%2 == 0:
            frame = result.orig_img.copy() #original frame
            frame_original = result.orig_img  # frame for the classifier (without the lines)
            traj=0 #
            
            for j,line in enumerate(lines):  # Itera con passi di 2
                frame = drawLine(frame,points[2*j],points[2*j+1], line["id"])

            # Itera su ogni bounding box e ID
            if result.boxes.xyxy is not None and result.boxes.id is not None:
                # Interation on evry box
                for box, id in zip(result.boxes.xyxy, result.boxes.id):
                    x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                    img = classifier_preparer(box, frame_original, device)  # Prepara input per la rete di classificazione
                    x1_extended = x1 - extra_width  # Aggiungi margine a sinistra
                    x2_extended = x2 + extra_width  # Aggiungi margine a destra
                    
                    # Classificazioni (gender, hat, bag)
                    gender, bag, hat = classifier_model(img) # <---------------------------CLASSIFICATION
                    gender_pred = torch.sigmoid(gender)  #0.4
                    hat_pred = torch.sigmoid(hat)  
                    bag_pred = torch.sigmoid(bag) #0.3
                    # Disegna il bounding box sul frame

                    # dorw labels of the box
                    box_x, box_y, box_width, box_height = 10, 10, 250, 33*(trajectory+1)  
                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1) # on the dop
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # drow the bouinding box
                    cv2.rectangle(frame,(x1,y1),(x1+20,y1+20),(255,255,255),-1) # id

                    x1_extended = max(x1_extended, 0)
                    x2_extended = min(x2_extended, frame.shape[1])
                    cv2.rectangle(frame, (x1_extended, y2), (x2_extended, y2 + 70), (255, 255, 255), -1)  # undre the box

                    #for each person check if it is crossing a line
                    if(id.item() in data["people"]):
                        person=data["people"][id.item()]
                        for line in lines:
                            if(do_intersect(person["xyxy"],(x1,x2,y1,y2),line_dict["line"][line["id"]]["p1"],line_dict["line"][line["id"]]["p2"])):
                                if(calculate_crossing(person["xyxy"],(x1,x2,y1,y2),line_dict["line"][line["id"]]["center"],line_dict["line"][line["id"]]["arrowEnd"])):
                                    traj=line["id"]
                                        

                    
                            
                            
                    # Initialize a new person
                    new_person = {
                        "id": id.item(),
                        "gender": gender_pred.item(),
                        "hat": hat_pred.item(),
                        "bag": bag_pred.item(),
                        "trajectory": [],
                        "xyxy":(x1,x2,y1,y2)
                    }
                    # if the person is already in the data
                    if new_person["id"] in data["people"]:
                        person = data["people"][new_person["id"]]
                        person["gender"] = new_person["gender"]
                        person["hat"] = new_person["hat"]
                        person["bag"] = new_person["bag"]
                        person["xyxy"] = new_person["xyxy"]

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

                        # for the final file
                        final_f = append(final_f, int(id.item()), gender_pred, bag_pred, hat_pred)

                        # add a new person's trajectory (if the person is crossing a new line)
                        if(len(person["trajectory"])>0):
                            if(person["trajectory"][-1]!=traj and traj!=0):
                                person["trajectory"].append(traj)
                                final_f["people"][int(id.item()-1)]["trajectory"].append(traj)
                        elif traj!=0:
                            person["trajectory"].append(traj)
                            final_f["people"][int(id.item()-1)]["trajectory"].append(traj)

                        # lable insie the box in the video
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

                    # if the person is not in the data
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

                # counting box on the top left of the video
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
final_f = my_track(video_path, tracker)


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