import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144, 145, 246]
RIGHT_EYE = [263, 387, 385, 362, 380, 373, 374, 466]

# Open webcam or read video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape

            # Extract and draw left eye landmarks
            for idx in LEFT_EYE:
                x, y = int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Extract and draw right eye landmarks
            for idx in RIGHT_EYE:
                x, y = int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Display frame
    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()