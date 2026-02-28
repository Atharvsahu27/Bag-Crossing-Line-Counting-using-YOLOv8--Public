# Bag Crossing Line Counting using YOLOv8 

##  Project Overview
This project detects and counts bags crossing a virtual line at a doorway using a custom-trained YOLOv8 model.

The system:
- Detects bags using a custom-trained YOLOv8 model
- Tracks objects using built-in YOLO tracking
- Counts bags when they cross a vertical line
- Displays live count on video

---

##  Features
- Custom YOLOv8 model trained on manually 281 labeled images
- Object tracking to prevent double counting
- Vertical line crossing detection
- Real-time counting display


---

##  Tech Stack
- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch

---

## 🏷 Dataset Preparation

- Extracted frames from recorded doorway video.
- Manually annotated 281 images using bounding boxes.
- Created YOLO-format labels for single class: `bag`.
- Generated dataset version and trained custom YOLOv8 model.

##  File Descriptions

- **get_frames.py**
  Extracts frames from input video for dataset creation and manual labeling.

- **test.py**
  Runs object detection using trained model without counting logic.

- **count_bags.py**
  Tracks detected bags and counts them when crossing a vertical line.

- **best.pt**
  Custom trained YOLOv8 model file.

  **bag_training**
  It trains the model on our custom dataset


  **output.mp4**
  Its a recorded video of the output where the input is Problem Statement Scenario1.mp4



  **THE MODEL WAS TRAINED ON COLLAB**
