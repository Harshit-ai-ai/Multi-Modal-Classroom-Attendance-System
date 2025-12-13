CAM_HEIGHT = float(input("Enter camera height from floor (m): "))
TILT_ANGLE = float(input("Enter camera tilt angle in degrees: "))
FOV = float(input("Enter camera vertical FOV in degrees: "))
import cv2
import math
from ultralytics import YOLO

CAM_HEIGHT = 2.8        
CAM_TILT = 25.0         
VERTICAL_FOV = 45.0     

REF_HEIGHT = 1.70       
REF_DISTANCE = 3.0      
REF_BBOX_PIXELS = 420   

def pixel_to_angle(y_pixel, frame_height):
    dy = y_pixel - (frame_height / 2)
    return (dy / frame_height) * VERTICAL_FOV

def estimate_distance(bbox_pixel_height):
    return REF_DISTANCE * (REF_BBOX_PIXELS / bbox_pixel_height)

def estimate_height(head_y, bbox_pixel_height, frame_height):
    angle_from_center = pixel_to_angle(head_y, frame_height)
    total_angle = math.radians(CAM_TILT + angle_from_center)

    D = estimate_distance(bbox_pixel_height)
    height = CAM_HEIGHT - D * math.tan(total_angle)
    return max(height, 0)


yolo = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    results = yolo(frame, stream=True)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bbox_h = y2 - y1

            est_h = estimate_height(y1, bbox_h, h)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"H={est_h:.2f}m",
                        (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,0), 2)

    cv2.imshow("Hybrid Height Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
