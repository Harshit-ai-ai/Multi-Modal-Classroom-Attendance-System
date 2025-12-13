import cv2
import mediapipe as mp
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
cap=cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame=cap.read()

        image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        result=pose.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks=result.pose_landmarks.landmark
        except:
            pass
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Mediapipe feed', image)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    def calculateangles(a,b):
        x1, y1=a
        x2, y2=b
        angle=np.degrees(np.arctan2((y2-y1), (x2-x1)))
        return angle
    ear=[
        landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
        landmarks[mp_pose.PoseLandmark.LEFT_EAR].y
    ]
    leftshoulder=[
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    ]
    angle=calculateangles(leftshoulder,ear)
    print("Angle:", angle)
cv2.putText(image, str(int(angle)), (50,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
