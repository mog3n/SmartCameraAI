# SmartCameraAI

### Notice: Currently in a "working" phase: Do not use this script if you are not a programmer.

### AI Features For Your Smart Camera
The purpose of this project is to create an analysis tool for any smart security system.

**Current Features**:
* Person detection
* Face recognition and grouping

**Proposed Features**

These features are currently being worked on

* Car recognition (Remembers your cars, alerts you when suspicious cars are on camera) ~ WIP
* License plate recognition (Extracts license plates from videos) ~ WIP

### Installation
Step 1. To setup this rep, you will need the following installed:
```
pip3 install detectron2 opencv-python face_recognition numpy
```
Step 2: Clone my repo

Step 3: Make a new folder called `video` in the root directory of this repo

Step 4: Copy/move footage from your smart cameras to the `video` folder you just created.

Step 5: Run `python3 object_extraction.py`. This will extract any important features used for analysis.

Step 6: Run `python3 face_detection.py`. This will detect and label faces. Each detected face will be a folder within `faces` (i.e. group0, group1, ..., etc.).

Step 7: More features to come!