# Multi-Modal-Classroom-Attendance-System
The proposed algorithm is a real-time, vision-based attendance system designed to automatically record student presence in a classroom using a single overhead camera, without relying on manual roll calls or intrusive biometric hardware.

The system operates by combining deep learning–based human detection, geometric camera modeling, and pose estimation into a unified computer vision pipeline.

Core Working Principle

1)Real-Time Person Detection
Each video frame captured by the classroom camera is processed using a lightweight deep learning object detection model (YOLOv8) to accurately identify and localize all students present in the classroom. The detector outputs bounding boxes corresponding to each detected individual.

2)Head–Nose to Head–Shoulder Ratio
The head–nose to head–shoulder ratio is a normalized geometric metric used to analyze facial and upper-body proportions in a scale-independent way. It is calculated as the ratio between the distance from the top of the head to the nose and the distance from the top of the head to the shoulders. Because both distances are measured within the same frame, the ratio remains independent of camera distance, resolution, and subject size. This makes it especially useful in computer vision applications where consistent measurements are required across different images or video frames.

3)Geometric Height and Distance Estimation
To prevent proxy attendance and detect mid-class exits, the algorithm estimates the physical height and relative distance of each detected individual using camera geometry.
A one-time calibration step converts pixel measurements into real-world scale by using:

1)Known camera mounting height and tilt angle
2)Camera vertical field of view
3)A reference subject at a known distance
4)Using trigonometric projection, the algorithm computes the estimated vertical position of a person’s head relative to the floor, enabling spatial consistency checks across time.

4)Pose Estimation for Robust Body Analysis
MediaPipe Pose is applied in parallel to extract skeletal landmarks such as the head, shoulders, and torso. This enhances robustness when full body visibility is partially obstructed by desks or other students and enables posture-based verification and angle analysis.

5)Temporal Presence Validation
By continuously tracking detected individuals across frames, the system verifies whether a student remains present for a minimum duration. Students leaving the classroom for extended periods or failing to return after breaks are automatically flagged as absent for that session.

Output

For each lecture session, the system automatically generates:

1)Verified attendance records

2)Entry and exit timestamps

3)Visual evidence for disputed cases

Intended Impact

This algorithm significantly reduces manual effort, eliminates proxy attendance, and improves the accuracy and fairness of classroom attendance systems, making it well-suited for deployment in modern educational institutions.
