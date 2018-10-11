"""
    File name: structureFromMotion.py
    Version: 0.0.0
    Author: Drew Wagner
    Date created: 2018-10-10
    Date last modified: 2018-10-11
    Python Version: 3.7
"""
import cv2
import numpy as np
import time


class StructureFromMotion:
    def __init__(self, camera_pos=np.array([0, 0, 0]), camera_vel=np.array([0, 0, 0]), camera_space=False,
                 target_scale=(320, 240), focal_length=38.199):
        self.target_scale = target_scale
        self.focal_length = focal_length
        self.camera_pos = camera_pos
        self.camera_vel = camera_vel

        # TODO Try with SIFT features
        # params for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=30,
                                   qualityLevel=0.5,
                                   minDistance=7,
                                   blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.center = np.array([target_scale[0] / 2, target_scale[1] / 2])

        self.global_points = np.array([])
        self.frame_points = np.array([])
        self.find_in_camera_space = camera_space

        self.previous_frame = None
        self.p0 = None

    def set_camera_velocity(self, new_velocity):
        """
        :param new_velocity: np.array of shape (1,3)
        :return:

        Sets the camera velocity
        """
        self.camera_vel = new_velocity

    def preprocess_frame(self, raw_frame):
        """
        :param raw_frame: Raw frame from VideoCapture
        :return: Processed frame

        Preprocesses the captured frames: Resize, grayscale, histogram equalization
        """
        scaled_frame = cv2.resize(raw_frame, self.target_scale)
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        final_frame = cv2.equalizeHist(gray_frame)
        return final_frame

    def calculate_optical_flow(self, current_frame):
        """
        :param current_frame:
        :return: Good new and old points, as tuple, or None and old points if no good points found

        Calculates Optical Flow using Lucas-Kanada algorithm
        """
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.previous_frame, current_frame, self.p0, None, **self.lk_params)
        good_old = self.p0[st == 1]

        if p1 is not None:
            good_new = p1[st == 1]
            return good_new, good_old
        else:
            return None, good_old

    def calculate_distance_from_disparity(self, disparity_origin, disparity):
        """
        :param disparity_origin: 2D vector : position of feature in previous frame
        :param disparity: 2D vector: displacement of feature since previous frame
        :return: float: distance along z axis from camera

        Calculates the z-axis position from the disparity.
        TODO Or does it? Verify that it is not simply the straight line distance from camera...
        """
        # Calculate the vector pointing from the old feature position to the center of the frame (Vanishing point)
        # TODO Try normalizing with Fast InvSqrt()
        # https://en.wikipedia.org/wiki/Fast_inverse_square_root
        radial_vector = self.center - disparity_origin
        radial_vector = radial_vector / np.linalg.norm(radial_vector)

        # Distribute the disparity along the x, y and z(radial) axis
        x_disparity = -disparity[0]  # Negative because rightward camera motion causes leftward pixel motion
        y_disparity = -disparity[1]  # Same thing for upward and downward motion

        # dot of radial_vector and disparity
        z_disparity = radial_vector[0] * x_disparity + radial_vector[1] * y_disparity

        # Weight each axis of the disparity according to camera velocity (dot product)
        weighted_disparity = x_disparity * self.camera_vel[0] + \
                             y_disparity * self.camera_vel[1] + \
                             z_disparity * self.camera_vel[2]

        # Distance is inversely proportional to length of disparity
        distance = self.focal_length / weighted_disparity
        return distance

    def calculate_camera_space_position_of_feature(self, new, old):
        """
        :param new: pixel position of feature this frame
        :param old: pixel position of feature in previous frame
        :return: (x,y,z) position in camera space

        Calculates the 3D position in camera space of a feature from its disparity
        from one frame to the next.
        """

        # Disparity is the change in pixel position of a tracked feature from one frame to the next
        disparity = new - old
        distance = self.calculate_distance_from_disparity(old, disparity)

        # TODO Fix warping of position near edges
        # x = new[0] / focal_length * (distance * width) / 1000
        # y = new[1] / focal_length * (distance * height) / 1000
        # For now:
        x = new[0]
        y = new[1]

        return np.array([x, y, distance])

    def initialize(self, raw_previous_frame):
        """ 
        :param raw_previous_frame: 
        :return:
        
        Initializes the instance with the previous frame, and finds points to track
        """
        self.previous_frame = self.preprocess_frame(raw_previous_frame)
        self.p0 = cv2.goodFeaturesToTrack(self.previous_frame, mask=None, **self.feature_params)

    def get_frame_points(self, current_frame):
        """
        :param current_frame: The current raw video frame
        :return: np array of 3D vectors, points are relative to camera

        Calculates the 3D positions relative to the camera for points present between the current and last frame.
        """
        frame_points = []
        current_frame = self.preprocess_frame(current_frame)

        good_new, good_old = self.calculate_optical_flow(current_frame)
        if good_new is not None:  # If good points were found in current_frame
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                cam_pos = self.calculate_camera_space_position_of_feature(new, old)

                if not self.find_in_camera_space:
                    # TODO Camera rotation
                    world_pos = cam_pos + self.camera_pos  # Transforms from camera space to world space
                    frame_points.append(world_pos)
                else:
                    frame_points.append(cam_pos)
            self.p0 = good_new.reshape(-1, 1, 2)

        self.previous_frame = current_frame.copy()

        return frame_points

    def update(self):
        global dt
        self.camera_pos = self.camera_pos + self.camera_vel * dt


video_source = "ringtest.avi"
cap = cv2.VideoCapture(video_source)

structureFromMotion = StructureFromMotion(camera_pos=np.array([0, 0, 0]),
                                          camera_vel=np.array((0, 0, 5.2)),
                                          camera_space=True)
ret, first_frame = cap.read()
minFPS = 1000
maxFPS = 0
if ret:
    structureFromMotion.initialize(first_frame)

    dt = 1/20
    print("Processing video...")
    while True:
        # Get time at beginning of loop for frame rate calculation
        t = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        # This is where the magic happens....
        structureFromMotion.update()
        structureFromMotion.frame_points = structureFromMotion.get_frame_points(frame)
        # Keyboard input handling
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Update delta time and display frame rate
        dt = time.time() - t
        FPS = 1/dt
        if FPS > maxFPS:
            maxFPS = FPS
        if FPS < minFPS:
            minFPS = FPS
else:
    print("Problem with video stream... Unable to capture first frame")

print ("Min FPS: %f MaxFPS: %f" % (minFPS, maxFPS))
# Clean up
cap.release()
cv2.destroyAllWindows()
