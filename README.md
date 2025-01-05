# Real-Time Video Analysis Pipeline 

## About this Repository 

Welcome to the repository for our university competition project! This challenge involves creating a complete pipeline for **real-time video analysis**. The main objectives of the project are:

- **Detecting people** 
- **Identifying attributes of individuals**, such as:
  - Gender 🧑
  - Whether they are wearing a hat or not 🎩
  - Whether they are carrying a backpack 🎒

## Rules and Requirements 

We are allowed to utilize **pre-trained networks** for most of the pipeline. However, for the **classifier** responsible for identifying the attributes, we are required to train it ourselves using the datasets provided by [MIVIA PAR2023](https://mivia.unisa.it/par2023/).

## Goals

1. Design and implement a robust pipeline capable of analyzing video streams in real-time.
2. Accurately detect and classify the specified attributes.
3. Optimize the system for performance and reliability.

## About repository
### **1. `src/`**
Contains the primary files for running the project:
- **`mian.py`**: The main file that manages the entire workflow.
- **`projectionTest.py`**: Used to test the projection of points from the real world to the image.
- **`tracker.py`**: A support file for pedestrian tracking.
- **`finalFile.py`**: A support file for building classification and final analyses.

### **2. `config/`**
This folder contains configuration files required by the project:
- **Camera information**: Parameters such as position, orientation, and optical properties.
- **Crossing lines definition**: Coordinates of the points defining imaginary crossing lines for analysis.

### **3. `classifier/`**
Includes all files related to the classification of pedestrian attributes:
- **`supportScripts/`**: Contains utility functions such as dataset readers for classification purposes.

### **4. `tracking/`**
Contains all files related to pedestrian detection and tracking:
- **`supportScripts/`**: Includes utility functions like projection algorithms used by the tracker.

---

## **How to Run **
### Project
1. Ensure that the **camera configuration** and **crossing line definitions** are correctly set up in the `config/` folder.
2. Run the main script `mian.py` to start the system.
### Classifier
- use src/Classifier/train.py in order to train
- use src/Classifier/modelTester.py in order to test a model



