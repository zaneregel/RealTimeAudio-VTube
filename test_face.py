import cv2
import math
import mediapipe as mp
import pygame
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.95, refine_landmarks=True, max_num_faces=1)

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
SMOOTHING_FACTOR = 0.6
smoothed_landmarks = {}
left_last = []
right_last = []
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Eye & Mouth Tracking")

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Define standard facial feature sizes for ratios
MOUTH_WIDTH = 0.05
MOUTH_HEIGHT = 0.03

# Define initial eye dot and mouth ellipse
eye_dot = (WIDTH // 2, HEIGHT // 2)
mouth_x, mouth_y = WIDTH // 2, HEIGHT // 1.5
mouth_width = 80
mouth_height = 10  # Default flattened ellipse

# Capture webcam video
cap = cv2.VideoCapture(0)

# Fix face location defs
# Key points for reference
NOSE_TIP = 1
NOSE_BRIDGE = 6
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# rounding and binning function to herlp with eye tracking
def normalized_and_round(value, min, max):
    normalized_value = (value - min)/(max - min)
    rounded_value = round(normalized_value * 5) / 5 #rounds to the nearest 0.2
    return rounded_value

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally & convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Pygame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_x, nose_y, nose_z = face_landmarks.landmark[NOSE_TIP].x, face_landmarks.landmark[NOSE_TIP].y, face_landmarks.landmark[NOSE_TIP].z
            nose_bridge_x, nose_bridge_y, nose_bridge_z = face_landmarks.landmark[NOSE_BRIDGE].x, face_landmarks.landmark[NOSE_BRIDGE].y, face_landmarks.landmark[NOSE_BRIDGE].z 
            #print("%s %s %s" % (nose_x, nose_y, nose_z))
            # Fixed position of nose then calculate difference between the original nose position and the fixed position to translate all other points to that difference
            fixed_nose_x, fixed_nose_y, fixed_nose_z = 0.5, 0.5, -0.11
            # Optimal nose to bridge difference for correct scaling of the face. If the difference increases decrease the scaling amount to make sure the distance between nose and bridge remain the same
            fixed_nose_difference_x, fixed_nose_difference_y, fixed_nose_difference_z = 0, 0.11, -0.033
            bridge_difference_x, bridge_difference_y, bridge_difference_z = nose_x - nose_bridge_x, nose_y - nose_bridge_y, nose_z - nose_bridge_z
            #print("%s %s %s" % (bridge_difference_x, bridge_difference_y, bridge_difference_z))

            difference_x, difference_y, difference_z = fixed_nose_x - nose_x, fixed_nose_y - nose_y, fixed_nose_z - nose_z

            for landmark in face_landmarks.landmark:
                # Calculate required scaling based on its distance from the nose
                scale_factor_x = ((difference_x - (nose_x - landmark.x)) / difference_x)
                scale_factor_y = ((difference_y - (nose_y - landmark.y)) / difference_y)
                scale_factor_z = ((difference_z - (nose_z - landmark.z)) / difference_z)
                landmark.x += (difference_x) * scale_factor_x
                landmark.y += (difference_y) * scale_factor_y
                landmark.z += (difference_z) * scale_factor_z

            # After nose is fixed apply proper scaling to ensure consistant distances
            # Check nose point to nose bridge distance and then apply proper scaling
            nose = (face_landmarks.landmark[NOSE_TIP].x, face_landmarks.landmark[NOSE_TIP].y, face_landmarks.landmark[NOSE_TIP].z)
            nose_bridge = (face_landmarks.landmark[NOSE_BRIDGE].x, face_landmarks.landmark[NOSE_BRIDGE].y, face_landmarks.landmark[NOSE_BRIDGE].z)
            nose_distance = abs(math.dist(nose, nose_bridge))
            target_distance = 0.15
            nose_adjust_ratio = target_distance/nose_distance

            # Apply proper scaling based on nose nose bridge distance
            for landmark in face_landmarks.landmark:
                # Calculate required scaling based on its distance from the nose
                landmark.x += (landmark.x - face_landmarks.landmark[NOSE_TIP].x) * nose_adjust_ratio
                landmark.y += (landmark.y - face_landmarks.landmark[NOSE_TIP].y) * nose_adjust_ratio
                landmark.z += (landmark.z - face_landmarks.landmark[NOSE_TIP].z) * nose_adjust_ratio
                
            
            # Apply Smoothing because scaling causes more jittering as the small moves are now more exaggerated
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx not in smoothed_landmarks:
                    smoothed_landmarks[idx] = [landmark.x, landmark.y, landmark.z]
                
                # Apply smoothing
                smoothed_landmarks[idx][0] = SMOOTHING_FACTOR * smoothed_landmarks[idx][0] + (1 - SMOOTHING_FACTOR) * landmark.x
                smoothed_landmarks[idx][1] = SMOOTHING_FACTOR * smoothed_landmarks[idx][1] + (1 - SMOOTHING_FACTOR) * landmark.y
                smoothed_landmarks[idx][2] = SMOOTHING_FACTOR * smoothed_landmarks[idx][2] + (1 - SMOOTHING_FACTOR) * landmark.z

                landmark.x = smoothed_landmarks[idx][0]
                landmark.y = smoothed_landmarks[idx][1]
                landmark.z = smoothed_landmarks[idx][2]

            mp_drawing.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_TESSELATION, 
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Get landmark positions
            h, w, _ = frame.shape

            # Eye landmarks (right and left pupils)
            left_eye_x = int(face_landmarks.landmark[468].x * w)
            left_eye_y = int(face_landmarks.landmark[468].y * h)
            right_eye_x = int(face_landmarks.landmark[473].x * w)
            right_eye_y = int(face_landmarks.landmark[473].y * h)

            # Eye height (left and right upper and lower points)
            left_eye_upper = (face_landmarks.landmark[159].x, face_landmarks.landmark[159].y, face_landmarks.landmark[159].z)
            left_eye_lower = (face_landmarks.landmark[145].x, face_landmarks.landmark[145].y, face_landmarks.landmark[145].z)
            right_eye_upper = (face_landmarks.landmark[386].x, face_landmarks.landmark[386].y, face_landmarks.landmark[386].z)
            right_eye_lower = (face_landmarks.landmark[374].x, face_landmarks.landmark[374].y, face_landmarks.landmark[374].z)

            left_eye_height = (abs(math.dist(left_eye_upper, left_eye_lower)))
            right_eye_height = (abs(math.dist(right_eye_upper, right_eye_lower)))

            # Round and normalize the height ratio so they actually open and close the eyes
            left_ratio = normalized_and_round(left_eye_height*1000, 40, 65)
            right_ratio = normalized_and_round(right_eye_height*1000, 43, 55)

            # Further round to reduce flickering
            if(left_ratio < 0.4):
                left_ratio = 0
            elif(left_ratio > 0.8):
                left_ratio = 1
            if(right_ratio < 0.4):
                right_ratio = 0
            elif(right_ratio > 0.8):
                right_ratio = 1

            left_last.append(left_ratio)
            if(len(left_last) > 2):
                del left_last[0]
            
            right_last.append(right_ratio)
            if(len(right_last) > 2):
                del right_last[0]

            left_smooth = sum(left_last)/len(left_last)
            right_smooth = sum(right_last)/len(right_last)

            if(left_smooth > 0.79):
                left_smooth = 1
            if(right_smooth > 0.79):
                right_smooth = 1

            # Average eye position for smoother tracking
            left_eye_dot = (
                int((left_eye_x * 2) / 2 * WIDTH / w),
                int((left_eye_y * 2) / 2 * HEIGHT / h)
            )

            right_eye_dot = (
                int((right_eye_x * 2) / 2 * WIDTH / w),
                int((right_eye_y * 2) / 2 * HEIGHT / h)
            )

            # Mouth landmarks (upper lip & lower lip)
            upper_lip = face_landmarks.landmark[13]  # Upper lip center
            lower_lip = face_landmarks.landmark[14]  # Lower lip center

            # Compute mouth openness
            mouth_open = abs(upper_lip.y - lower_lip.y) * h

            # Set mouth ellipse height (scales up as mouth opens)
            mouth_height = max(10, int(mouth_open * 2))

            # Clear screen
            screen.fill(WHITE)

            # Define circle positions for eye tracking
            circle_width = 100
            circle_height = 100
            left_circle = [WIDTH // 3, HEIGHT // 3, circle_width, circle_height * left_smooth]
            right_circle = [2 * WIDTH // 3, HEIGHT // 3, circle_width, circle_height * right_smooth]

            # Draw two circles (eye tracking reference)
            pygame.draw.ellipse(screen, BLUE, left_circle, 2)
            pygame.draw.ellipse(screen, BLUE, right_circle, 2)

            # Draw the eye-tracking dot
            pygame.draw.circle(screen, RED, left_eye_dot, 10)
            pygame.draw.circle(screen, RED, right_eye_dot, 10)

            # Draw the mouth ellipseq
            pygame.draw.ellipse(screen, BLACK, (mouth_x - mouth_width // 2, mouth_y - mouth_height // 2, mouth_width, mouth_height))

    #Show Webcam output
    cv2.imshow("Face overlay", frame)

    # Quit
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    # Update the Pygame display
    pygame.display.flip()

cap.release()
cv2.destroyAllWindows()
pygame.quit()