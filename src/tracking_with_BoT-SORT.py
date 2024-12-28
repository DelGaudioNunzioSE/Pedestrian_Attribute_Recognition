import cv2
import torch
from ultralytics import YOLO  # Ensure you have the Ultralytics YOLO library installed
import os
import json
from torchvision import transforms
from Classifier.firstClassifier import CNNWithAttention
from PIL import Image





# path_corrente = os.getcwd()
# print(f"Il path corrente è: {path_corrente}")

#dataset
data = {
    "people": []
}

#Dall'immagine del bounding box all'input della rete
transform = transforms.Compose([transforms.ToTensor(),
				transforms.Resize((224, 224))])



def drawBox(box, frame,device):
    x1, y1, x2, y2 = map(int, box[:4])
    # Ritaglia il bounding box dall'immagine
    cropped_img = frame[y1:y2, x1:x2]
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
    checkpoint = torch.load('./src/Classifier/Models/checkpoint.pth')
    classifier_model.load_state_dict(checkpoint['model_state_dict'])   
    model.to(device)  # Move the model to the selected device
    classifier_model.to(device)

    # Confirm the device of the model
    print(f"The model is loaded on: {next(model.parameters()).device}")

    # Run tracking with the specified tracker configuration file
    results = model.track(source=video_path, show=False, tracker=tracker, stream=True, classes=0, imgsz = (540,920), vid_stride=3
                          ,iou = 0.9) #video, visualizza mentre elabora, parametri del tracker, stream = risultati in tempo reale
       
    for result in results:
        frame = result.orig_img  # Immagine originale del frame
    
        # Itera su ogni bounding box e ID
        for box, id in zip(result.boxes.xyxy, result.boxes.id):
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            img = drawBox(box, frame,device)  # Prepara input per la rete di classificazione
            
            # Classificazioni (gender, hat, bag)
            gender, hat, bag = classifier_model(img)
            gender_pred = torch.sigmoid(gender) > 0.5
            hat_pred = torch.sigmoid(hat) > 0.5
            bag_pred = torch.sigmoid(bag) > 0.5
            
            # Crea un dizionario con le informazioni
            new_person = {
                "id": id.item(),
                "gender": "Female" if gender_pred else "Male",
                "hat": "Yes" if hat_pred else "No",
                "bag": "Yes" if bag_pred else "No",
                "trajectory": "???"
            }
            data["people"].append(new_person)
            
            # Disegna il bounding box sul frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Testo con le informazioni
            text = f"ID: {new_person['id']} | G: {new_person['gender']} | Hat: {new_person['hat']} | Bag: {new_person['bag']}"
            
            # Posiziona il testo sopra il bounding box
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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



