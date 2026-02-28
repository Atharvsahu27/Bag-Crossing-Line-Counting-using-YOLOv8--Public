import cv2
import os

videos = [
    "D:/project/bag_counting/Problem Statement Scenario1.mp4",
    "D:/project/bag_counting/Problem Statement Scenario2.mp4",
    "D:/project/bag_counting/Problem Statement Scenario3.mp4"
]

output_folder = "all_frames"
os.makedirs(output_folder, exist_ok=True)

frame_id = 0

for video_path in videos:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open:", video_path)
        continue
    else:
        print("Opened:", video_path)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % 8 == 0:
            cv2.imwrite(f"{output_folder}/frame_{frame_id}.jpg", frame)
            frame_id += 1
        
        count += 1

    cap.release()

print("Frames extracted.")