"""
Main program to run the detection and TCP
"""

from argparse import ArgumentParser
import cv2
import mediapipe as mp
import numpy as np
import pygame
import requests
import json
from PIL import Image, ImageDraw
import pyassimp
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

count = 0

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

def apply_object_rotation_translation(roll, pitch, yaw, x, y, z):
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
    draw_fbx()

@run_once
def load_texture():
    # Load new image
    image = get_graph_image()
    if(image == None):
        return

    # Generate texture ID
    glBindTexture(GL_TEXTURE_2D, 1)
    
    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # Load texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    
    print("img loaded")

    return

def get_graph_image():
        url = "http://127.0.0.1:5000/data"
        try:
            response = requests.get(url)
        except:
            print("Error connecting to server")
            return None

        if response.status_code == 200:
            graph_data = json.loads(response.text)
        else:
            print("HTTP Error:", response.status_code)
            response.close()
            return None
        
        response.close()

        width = 800
        height = 600

       # Scale y-values to pixel coordinates (flip y-axis to match image space)
        y_scaled = [(1 - y) * (height - 1) for y in graph_data]

        # Scale x-values from [0,1023] to [0,799] for 800px width
        x_scaled = np.linspace(0, width - 1, 1024).astype(int)

        # Create blank image (black background)
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Convert to (x, y) pixel coordinates
        points_px = [(x_scaled[i], int(y_scaled[i])) for i in range(1024)]

        # Draw the graph
        img = Image.fromarray(img_array, "RGB")
        draw = ImageDraw.Draw(img)
        draw.line(points_px, fill="yellow", width=3)  # Draw line in yellow

        return img.tobytes()

def update_texture():
    # Load new image
    image = get_graph_image()
    if(image == None):
        return
    
    glBindTexture(GL_TEXTURE_2D, 1)
    # Update texture image
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, image)

    #print("updated")

    return


def draw_fbx():
    with pyassimp.load('TVHead.fbx') as scene:

        # Access First mesh in fbx
        meshes = []
        swap_material = 'Flat Black'
        for mesh in scene.meshes:
            vertices = np.array(mesh.vertices, dtype=np.float32).flatten()
            faces = np.array(mesh.faces, dtype=np.uint32).flatten()

            # Extract material for the mesh
            material = scene.materials[mesh.materialindex]

            # Extract material name
            name = material.properties['name']

            # Extracting color from material (diffuse color)
            if ('diffuse', 0) in material.properties:
                if material.properties['name'] == swap_material:
                    diffuse_color = (1.0, 1.0, 0)
                else:
                    diffuse_color = material.properties['diffuse']
            else:
                diffuse_color = (1.0, 0, 0)  # Default to red if no color found

            meshes.append((vertices, faces, diffuse_color, name))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
        for vertices, faces , diffuse_color, name in meshes:
            # Apply material color (diffuse)
            if name == swap_material:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, 1)
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glTexCoordPointer(2, GL_FLOAT, 0, [[0, 0], [1,0],[1,1],[0,1]])
                glColor3f(0.6,0.6,0.6)
            else:
                glColor3f(*diffuse_color)

            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, vertices)

            glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, faces)
            
            glDisableClientState(GL_VERTEX_ARRAY)

            if name == swap_material:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
                glDisable(GL_TEXTURE_2D)

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
    return

def main():
    roll, pitch, yaw = 0, 0, 0

    # Higher alpha means less smoothing
    # We have different classes for each attribute because it saves the prior value to adjust the new one
    # so we need to save them seperately or it wont work
    gen_alpha = 0.6
    smoothingRoll = ExponentialMovingAverageRoll(alpha = gen_alpha)
    smoothingPitch = ExponentialMovingAveragePitch(alpha = gen_alpha)
    smoothingYaw = ExponentialMovingAverageYaw(alpha = gen_alpha)
    smoothingX = ExponentialMovingAverageX(alpha = gen_alpha)
    smoothingY = ExponentialMovingAverageY(alpha = gen_alpha)
    smoothingZ = ExponentialMovingAverageZ(alpha = gen_alpha)
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
        load_texture()
        update_texture()

        try:
            if steady_pose.any():

                smoothed_roll = smoothingRoll.update(roll)
                smoothed_pitch = smoothingPitch.update(pitch)
                smoothed_yaw = smoothingYaw.update(yaw)
                smoothed_x = smoothingX.update(steady_pose[1][0])
                smoothed_y = smoothingY.update(steady_pose[1][1])
                smoothed_z = smoothingZ.update(steady_pose[1][2])
                apply_object_rotation_translation(smoothed_roll, smoothed_pitch, smoothed_yaw, smoothed_x, smoothed_y, smoothed_z)
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
