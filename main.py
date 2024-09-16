"""
Main program to run the detection and TCP
"""

from argparse import ArgumentParser
import cv2
import mediapipe as mp
import numpy as np
import pygame
from PIL import Image
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# face detection and facial landmark
from facial_landmark import FaceMeshDetector

# pose estimation and stablization
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

# Miscellaneous detections (eyes/ mouth...)
from facial_features import FacialFeatures, Eyes

import sys

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

def print_debug_msg(args):
    msg = '%.4f ' * len(args) % args
    print(msg)

class ExponentialMovingAverageRoll:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_value = None  # Initialize with no value yet

    def update(self, new_value):
        if self.smoothed_value is None:  # First value is the same as the input
            self.smoothed_value = new_value
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value
    
class ExponentialMovingAveragePitch:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_value = None  # Initialize with no value yet

    def update(self, new_value):
        if self.smoothed_value is None:  # First value is the same as the input
            self.smoothed_value = new_value
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value
    
class ExponentialMovingAverageYaw:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_value = None  # Initialize with no value yet

    def update(self, new_value):
        if self.smoothed_value is None:  # First value is the same as the input
            self.smoothed_value = new_value
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value
    
class ExponentialMovingAverageX:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_value = None  # Initialize with no value yet

    def update(self, new_value):
        if self.smoothed_value is None:  # First value is the same as the input
            self.smoothed_value = new_value
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value

class ExponentialMovingAverageY:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_value = None  # Initialize with no value yet

    def update(self, new_value):
        if self.smoothed_value is None:  # First value is the same as the input
            self.smoothed_value = new_value
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value
    
class ExponentialMovingAverageZ:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_value = None  # Initialize with no value yet

    def update(self, new_value):
        if self.smoothed_value is None:  # First value is the same as the input
            self.smoothed_value = new_value
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value

def draw_cube():
    glBegin(GL_QUADS)

    # Front face (Red)
    glColor3f(1.0, 0.0, 0.0)  # Red
    glVertex3f(-1.0, -1.0,  1.0)
    glVertex3f( 1.0, -1.0,  1.0)
    glVertex3f( 1.0,  1.0,  1.0)
    glVertex3f(-1.0,  1.0,  1.0)

    # Back face (Green)
    glColor3f(0.0, 1.0, 0.0)  # Green
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0,  1.0, -1.0)
    glVertex3f( 1.0,  1.0, -1.0)
    glVertex3f( 1.0, -1.0, -1.0)

    # Top face (Blue)
    glColor3f(0.0, 0.0, 1.0)  # Blue
    glVertex3f(-1.0,  1.0, -1.0)
    glVertex3f(-1.0,  1.0,  1.0)
    glVertex3f( 1.0,  1.0,  1.0)
    glVertex3f( 1.0,  1.0, -1.0)

    # Bottom face (Yellow)
    glColor3f(1.0, 1.0, 0.0)  # Yellow
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f( 1.0, -1.0, -1.0)
    glVertex3f( 1.0, -1.0,  1.0)
    glVertex3f(-1.0, -1.0,  1.0)

    # Right face (Magenta)
    glColor3f(1.0, 0.0, 1.0)  # Magenta
    glVertex3f( 1.0, -1.0, -1.0)
    glVertex3f( 1.0,  1.0, -1.0)
    glVertex3f( 1.0,  1.0,  1.0)
    glVertex3f( 1.0, -1.0,  1.0)

    # Left face (Cyan)
    glColor3f(0.0, 1.0, 1.0)  # Cyan
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0,  1.0)
    glVertex3f(-1.0,  1.0,  1.0)
    glVertex3f(-1.0,  1.0, -1.0)

    glEnd()

def apply_object_rotation_translation(roll, pitch, yaw, x, y, z, texture_id):
    # Reset the transformations to identity before applying new ones
    glLoadIdentity()

    # Apply translation (optional, move the cube back)
    glTranslatef(0 + (x/20), 0 - (y/20), -3 - (z/40))
    #glTranslatef(0, 0, -5)

    # Apply rotations based on object rotation data
    glRotatef(-pitch, 1, 0, 0)  # Rotate around x-axis (roll)
    glRotatef(-yaw, 0, 1, 0)  # Rotate around y-axis (pitch)
    glRotatef((roll*0.7), 0, 0, 1)  # Rotate around z-axis (yaw)

    # Draw the cube with the new rotation
    #draw_cube()
    draw_tv()
    draw_tvface_with_texture()

@run_once
def load_texture():
    # Load new image
    try:
        image_path = "G:\Videos\VTubing\Python-Realtime-Audio\plot_img.png"
        img = Image.open(image_path)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = img.convert("RGBA").tobytes()
    except:
        return

    # Generate texture ID
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # Load texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    
    print("img loaded")

    return texture_id

def update_texture():
    # Load new image
    try:
        image_path = "G:\Videos\VTubing\Python-Realtime-Audio\plot_img.png"
        img = Image.open(image_path)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = img.convert("RGBA").tobytes()
    except:
        return
    
    # Generate texture ID
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # Load texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    
    #print("updated")

    return texture_id




def draw_tvface_with_texture():
    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)

    #glBindTexture(GL_TEXTURE_2D, texture_id)
    # Draw Just Monitor
    glColor3f(1.0, 1.0, 0.0)
    glTexCoord2f(0, 0)
    glVertex3f(-0.938077, -0.6964, 0.428424)
    
    glTexCoord2f(1, 0)
    glVertex3f(0.940166, -0.6964, 0.428424)
    
    glTexCoord2f(1, 1)
    glVertex3f(0.940166, 0.587625, 0.428424)
    
    glTexCoord2f(0, 1)
    glVertex3f(-0.938077, 0.587625, 0.428424)


    glEnd()
    glDisable(GL_TEXTURE_2D)

def draw_tv():
    glBegin(GL_QUADS)

    #Draw TV Body except monitor
    # Faces done in counter clock wise manner from top face
    # Blenders Coords to OpenGL
    # Blender -> X Y Z -> translated to OpenGL is Y Z X
    # Monitor Bevel
    glColor3f(0.1, 0.1, 0.1)
    glVertex3f(-0.938077, 0.587625, 0.428424)
    glVertex3f(0.940166, 0.587625, 0.428424)
    glVertex3f(0.993712, 0.665196, 0.493743)
    glVertex3f(-0.991623, 0.665196, 0.493743)

    glVertex3f(-0.991623, -0.773975, 0.493743)
    glVertex3f(-0.938077, -0.6964, 0.428424)
    glVertex3f(-0.938077, 0.587625, 0.428424)
    glVertex3f(-0.991623, 0.665196, 0.493743)

    glVertex3f(-0.991623, -0.773975, 0.493743)
    glVertex3f(0.993712, -0.773975, 0.493743)
    glVertex3f(0.940166, -0.696405, 0.428424)
    glVertex3f(-0.938077, -0.6964, 0.428424)

    glVertex3f(0.940166, -0.696405, 0.428424)
    glVertex3f(0.993712, -0.773975, 0.493743)
    glVertex3f(0.993712, 0.665196, 0.493743)
    glVertex3f(0.940166, 0.587625, 0.428424)

    #Monitor Front Edge
    glColor3f(0.2, 0.2, 0.2)
    glVertex3f(-0.991623, 0.665196, 0.493743)
    glVertex3f(0.993712, 0.665196, 0.493743)
    glVertex3f(1.03661, 0.705974, 0.493743)
    glVertex3f(-1.03452, 0.705974, 0.493743)

    glVertex3f(-1.03452, -0.824061, 0.493743)
    glVertex3f(-0.991623, -0.773975, 0.493743)
    glVertex3f(-0.991623, 0.665196, 0.493743)
    glVertex3f(-1.03452, 0.705974, 0.493743)

    glVertex3f(-1.03452, -0.824061, 0.493743)
    glVertex3f(1.03661, -0.824061, 0.493743)
    glVertex3f(0.993712, -0.773975, 0.493743)
    glVertex3f(-0.991623, -0.773975, 0.493743)

    glVertex3f(0.993712, -0.773975, 0.493743)
    glVertex3f(1.03661, -0.824061, 0.493743)
    glVertex3f(1.03661, 0.705974, 0.493743)
    glVertex3f(0.993712, 0.665196, 0.493743)

    #Monitor Thick Edge
    glVertex3f(-1.03452, 0.705974, 0.493743)
    glVertex3f(1.03661, 0.705974, 0.493743)
    glVertex3f(1.03661, 0.705201, 0.370013)
    glVertex3f(-1.03452, 0.705201, 0.370013)

    glVertex3f(-1.03279, -0.817768, 0.370013)
    glVertex3f(-1.03452, -0.824061, 0.493743)
    glVertex3f(-1.03452, 0.705974, 0.493743)
    glVertex3f(-1.03279, 0.705201, 0.370013)

    glVertex3f(1.03488, -0.817768, 0.370013)
    glVertex3f(1.03661, -0.824061, 0.493743)
    glVertex3f(-1.03452, -0.824061, 0.493743)
    glVertex3f(-1.03279, -0.817768, 0.370013)

    glVertex3f(1.03661, -0.824061, 0.493743)
    glVertex3f(1.03488, -0.817768, 0.370013)
    glVertex3f(1.03488, 0.705201, 0.370013)
    glVertex3f(1.03661, 0.705974, 0.493743)

    #Monitor Back Big Faces
    glColor3f(0.5, 0.5, 0.5)
    glVertex3f(-1.03279, 0.705201, 0.370013)
    glVertex3f(1.03488, 0.705201, 0.370013)
    glVertex3f(0.811153, 0.436703, -0.757124)
    glVertex3f(-0.809064, 0.436703, -0.757124)

    glVertex3f(-0.809064, -0.760436, -0.757124)
    glVertex3f(-1.03279, -0.817768, 0.370013)
    glVertex3f(-1.03279, 0.705201, 0.370013)
    glVertex3f(-0.809064, 0.436703, -0.757124)

    glVertex3f(0.811153, -0.760436, -0.757124)
    glVertex3f(1.03488, -0.817768, 0.370013)
    glVertex3f(-1.03279, -0.817768, 0.370013)
    glVertex3f(-0.809064, -0.760436, -0.757124)

    glVertex3f(1.03488, -0.817768, 0.370013)
    glVertex3f(0.811153, -0.760436, -0.757124)
    glVertex3f(0.811153, 0.436703, -0.757124)
    glVertex3f(1.03488, 0.705201, 0.370013)

    # Back Face
    glVertex3f(0.811153, -0.760436, -0.757124)
    glVertex3f(-0.809064, -0.760436, -0.757124)
    glVertex3f(-0.809064, 0.436703, -0.757124)
    glVertex3f(0.811153, 0.436703, -0.757124)

    glEnd()

@run_once
def pygame_init():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    print("pygame_init ran")

@run_once
def init():
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, 1, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    gluLookAt(0, 1, 0, 0, 0, 0, 0, 1, 0)
    glClearColor(0.0, 1.0, 0.0, 1.0)
    print("init ran")

def main():
    roll, pitch, yaw = 0, 0, 0

    # Higher alpha means less smoothing
    # We have different classes for each attribute because it saves the prior value to adjust the new one
    # so we need to save them seperately or it wont work
    alpha = 0.2
    smoothingRoll = ExponentialMovingAverageRoll(alpha = alpha)
    smoothingPitch = ExponentialMovingAveragePitch(alpha = alpha)
    smoothingYaw = ExponentialMovingAverageYaw(alpha = alpha)
    smoothingX = ExponentialMovingAverageX(alpha = alpha)
    smoothingY = ExponentialMovingAverageY(alpha = alpha)
    smoothingZ = ExponentialMovingAverageZ(alpha = alpha)
    # use internal webcam/ USB camera
    cap = cv2.VideoCapture(args.cam)

    # Facemesh
    detector = FaceMeshDetector()

    # get a sample frame for pose estimation img
    success, img = cap.read()

    # Pose estimation related
    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
    image_points = np.zeros((pose_estimator.model_points_full.shape[0], 2))

    # extra 10 points due to new attention model (in iris detection)
    iris_image_points = np.zeros((10, 2))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # for eyes
    eyes_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # for mouth_dist
    mouth_dist_stabilizer = Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    )

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # first two steps
        img_facemesh, faces = detector.findFaceMesh(img)

        # flip the input image so that it matches the facemesh stuff
        img = cv2.flip(img, 1)

        # if there is any face detected
        if faces:
            # only get the first face
            for i in range(len(image_points)):
                image_points[i, 0] = faces[0][i][0]
                image_points[i, 1] = faces[0][i][1]
                
            # for refined landmarks around iris
            for j in range(len(iris_image_points)):
                iris_image_points[j, 0] = faces[0][j + 468][0]
                iris_image_points[j, 1] = faces[0][j + 468][1]

            # The third step: pose estimation
            # pose: [[rvec], [tvec]]
            pose = pose_estimator.solve_pose_by_all_points(image_points)

            x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.LEFT)
            x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.RIGHT)


            ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
            ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

            pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

            mar = FacialFeatures.mouth_aspect_ratio(image_points)
            mouth_distance = FacialFeatures.mouth_distance(image_points)

            # print("left eye: %.2f, %.2f" % (x_ratio_left, y_ratio_left))
            # print("right eye: %.2f, %.2f" % (x_ratio_right, y_ratio_right))

            # print("rvec (y) = (%f): " % (pose[0][1]))
            # print("rvec (x, y, z) = (%f, %f, %f): " % (pose[0][0], pose[0][1], pose[0][2]))
            # print("tvec (x, y, z) = (%f, %f, %f): " % (pose[1][0], pose[1][1], pose[1][2]))

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()

            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])

            steady_pose = np.reshape(steady_pose, (-1, 3))

            # stabilize the eyes value
            steady_pose_eye = []
            for value, ps_stb in zip(pose_eye, eyes_stabilizers):
                ps_stb.update([value])
                steady_pose_eye.append(ps_stb.state[0])

            mouth_dist_stabilizer.update([mouth_distance])
            steady_mouth_dist = mouth_dist_stabilizer.state[0]

            # uncomment the rvec line to check the raw values
            # print("rvec steady (x, y, z) = (%f, %f, %f): " % (steady_pose[0][0], steady_pose[0][1], steady_pose[0][2]))
            # print("tvec steady (x, y, z) = (%f, %f, %f): " % (steady_pose[1][0], steady_pose[1][1], steady_pose[1][2]))

            # calculate the roll/ pitch/ yaw
            # roll: +ve when the axis pointing upward
            # pitch: +ve when we look upward
            # yaw: +ve when we look left
            roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
            pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
            yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)

            # print("Roll: %.2f, Pitch: %.2f, Yaw: %.2f" % (roll, pitch, yaw))
            # print("left eye: %.2f, %.2f; right eye %.2f, %.2f"
            #     % (steady_pose_eye[0], steady_pose_eye[1], steady_pose_eye[2], steady_pose_eye[3]))
            # print("EAR_LEFT: %.2f; EAR_RIGHT: %.2f" % (ear_left, ear_right))
            # print("MAR: %.2f; Mouth Distance: %.2f" % (mar, steady_mouth_dist))
            
            # print the sent values in the terminal
            if args.debug:
                print_debug_msg((roll, pitch, yaw,
                        ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
                        mar, mouth_distance))


            # pose_estimator.draw_annotation_box(img, pose[0], pose[1], color=(255, 128, 128))

            # pose_estimator.draw_axis(img, pose[0], pose[1])

            pose_estimator.draw_axes(img_facemesh, steady_pose[0], steady_pose[1])

        else:
            # reset our pose estimator
            pose_estimator = PoseEstimator((img_facemesh.shape[0], img_facemesh.shape[1]))
            

        cv2.imshow('Facial landmark', img_facemesh)

        pygame_init()
        init()

        # Clear buffers for new frame
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Initialize Image Texture
        texture_id = load_texture()
        texture_id = update_texture()

        try:
            if steady_pose.any():

                smoothed_roll = smoothingRoll.update(roll)
                smoothed_pitch = smoothingPitch.update(pitch)
                smoothed_yaw = smoothingYaw.update(yaw)
                smoothed_x = smoothingX.update(steady_pose[1][0])
                smoothed_y = smoothingY.update(steady_pose[1][1])
                smoothed_z = smoothingZ.update(steady_pose[1][2])
                apply_object_rotation_translation(smoothed_roll, smoothed_pitch, smoothed_yaw, smoothed_x, smoothed_y, smoothed_z, texture_id)
        except:
            none = 0

        # Swap the display buffer
        pygame.display.flip()

        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)

    parser.add_argument("--port", type=int, 
                        help="specify the port of the connection to unity. Have to be the same as in Unity", 
                        default=5066)

    parser.add_argument("--cam", type=int,
                        help="specify the camera number if you have multiple cameras",
                        default=0)

    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)

    args = parser.parse_args()

    # demo code
    main()
