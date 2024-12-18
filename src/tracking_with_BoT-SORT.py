import torch
from ultralytics import YOLO  # Ensure you have the Ultralytics YOLO library installed
import os
import json


# path_corrente = os.getcwd()
# print(f"Il path corrente è: {path_corrente}")

#dataset
data = {
    "people": []
} 

def my_track(video_path, tracker, show=False):
    # Dynamically determine the best device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load YOLO model with weights onto the selected device
    model = YOLO('yolov8m.pt')
    model.to(device)  # Move the model to the selected device

    # Confirm the device of the model
    print(f"The model is loaded on: {next(model.parameters()).device}")

    # Run tracking with the specified tracker configuration file
    results = model.track(source=video_path, show=show, tracker=tracker, stream=True) #video, visualizza mentre elabora, parametri del tracker, stream = risultati in tempo reale
       
    for result in results:
       for id in result.boxes.id:
        new_person = {"id":id.item(),
                        "gender":"???",
                        "hat":"???",
                        "bag":"???",
                        "trajectory":"???"}
        data["people"].append(new_person) #le metto tutte durante il video poi alla fine vedo quali tenere




video_path = './videos/Atrio.mp4' # Path to the input video file (`video_fish.mp4`)
tracker='./confs/botsort.yaml' # Path to the tracker configuration file (`botsort.yaml`)
show=True # A boolean flag to display the processed video with tracked objects

my_track(video_path, tracker, show)

# Scrittura del file JSON
file_path = './videos/data.json'
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)  # indent=4 per rendere leggibile, ensure_ascii=False per caratteri non ASCII
    print(f"File salvato in {file_path}")



