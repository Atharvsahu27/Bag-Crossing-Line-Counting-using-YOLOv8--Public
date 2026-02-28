import cv2
from ultralytics import YOLO

# ===============================
# CONFIGURATION
# ===============================

VIDEO_PATH = "D:/project/bag_counting/Problem Statement Scenario1.mp4"   # change this
MODEL_PATH = "best.pt"
OUTPUT_PATH = "output.avi"

LINE_X = 600   # adjust to door position

# ===============================
# LOAD MODEL
# ===============================

model = YOLO(MODEL_PATH)

# ===============================
# OPEN VIDEO
# ===============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25
fps = int(fps)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ===============================
# COUNTING VARIABLES
# ===============================

count = 0
counted_ids = set()

# ===============================
# PROCESS VIDEO
# ===============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # ---- TOUCH-BASED COUNTING ----
            if x1 <= LINE_X <= x2:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    count += 1

    # Draw vertical line
    cv2.line(frame, (LINE_X, 0), (LINE_X, height), (255, 0, 0), 3)

    # Show count
    cv2.putText(frame, f"Bag Count: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Bag Counter", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete.")
print(f"Output saved as: {OUTPUT_PATH}")