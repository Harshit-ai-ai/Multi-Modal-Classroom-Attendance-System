import cv2
import math
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
print("=== Camera & Calibration Setup ===")

CAM_HEIGHT = float(input("Enter camera height from floor (meters): "))
CAM_TILT = float(input("Enter camera tilt angle downward (degrees): "))
VERTICAL_FOV = float(input("Enter camera vertical FOV (degrees): "))

REF_DISTANCE = float(input("Enter reference person distance from camera (meters): "))
REF_BBOX_PIXELS = float(input("Enter reference bounding box height in pixels: "))

print("\nCalibration complete. Starting system...\n")

yolo = YOLO("yolov8n.pt")

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

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

def calculate_angle(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def approximate_head_top(landmarks, frame_height):
    le = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    re = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
    eye_mid_y = (le.y + re.y) / 2
    return (eye_mid_y - 0.08) * frame_height

def head_length(landmarks, h):
    return abs(
        landmarks[mp_pose.PoseLandmark.NOSE].y * h -
        approximate_head_top(landmarks, h)
    )

def nose_to_shoulder(landmarks, h):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_mid_y = (ls.y + rs.y) / 2 * h
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * h
    return abs(shoulder_mid_y - nose_y)

def cranio_shoulder_ratio(landmarks, h):
    hl = head_length(landmarks, h)
    if hl == 0:
        return None
    return nose_to_shoulder(landmarks, h) / hl


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

    for (x,y,fw,fh) in faces:
        cv2.rectangle(frame, (x,y), (x+fw,y+fh), (0,255,0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        ratio = cranio_shoulder_ratio(landmarks, h)
        if ratio:
            cv2.putText(frame, f"CS Ratio: {ratio:.2f}",
                        (30,60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,255), 2)

        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Face + Pose + Ratio", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    results = yolo(frame, stream=True)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bbox_h = y2 - y1

            height_m = estimate_height(y1, bbox_h, h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"H={height_m:.2f}m",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    pose_result = pose.process(rgb)
    rgb.flags.writeable = True

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark

        ear = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * w),
            int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * h)
        )
        shoulder = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
            int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        )

        angle = calculate_angle(shoulder, ear)

        cv2.putText(frame, f"Posture Angle: {int(angle)}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        mp_drawing.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Smart Classroom Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


