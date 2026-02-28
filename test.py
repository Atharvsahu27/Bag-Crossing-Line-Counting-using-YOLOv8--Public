from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture("D:/project/bag_counting/Problem Statement Scenario1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.2, imgsz=1280)

    annotated = results[0].plot()
    cv2.imshow("High Resolution Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()