"""
This file has mostly been replaced by gloveGestureDetection.py
"""
import cv2
import numpy as np
import time


def Color(h, s, v):
	return np.array([h, s, v], dtype=np.uint8)


class ColorFinder:
	def __init__(self, name, lower, upper, trackbars=False):
		self.name = name
		self.lower = lower
		self.upper = upper
		if trackbars:
			cv2.namedWindow(name)
			cv2.createTrackbar("lower_h", name, self.lower[0], 255, self.update_lower_h)
			cv2.createTrackbar("upper_h", name, self.upper[0], 255, self.update_upper_h)
			cv2.createTrackbar("lower_s", name, self.lower[1], 255, self.update_lower_s)
			cv2.createTrackbar("upper_s", name, self.upper[1], 255, self.update_upper_s)
			cv2.createTrackbar("lower_v", name, self.lower[2], 255, self.update_lower_v)
			cv2.createTrackbar("upper_v", name, self.upper[2], 255, self.update_upper_v)

	def update_lower_h(self, val):
		self.lower[0] = val

	def update_upper_h(self, val):
		self.upper[0] = val

	def update_lower_s(self, val):
		self.lower[1] = val

	def update_upper_s(self, val):
		self.upper[1] = val

	def update_lower_v(self, val):
		self.lower[2] = val

	def update_upper_v(self, val):
		self.upper[2] = val

	def find_color(self, image, show=False):
		mask = cv2.inRange(image, self.lower, self.upper)
		if show:
			cv2.imshow("mask_" + self.name, mask)
		return mask

class HandTracker:
	def morph_open_close(self, mask):
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
		return mask


	def find_sorted_contours(self, mask):
		_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if contours:
			return sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
		else:
			return np.array([])

	def get_centroid_of_contour(self, contour):
		moments = cv2.moments(contour)
		centroid_x = 0
		centroid_y = 0
		if moments["m00"] > 0:
			centroid_y = moments["m01"]/moments["m00"]
			centroid_x = moments["m10"]/moments["m00"]
		return (int(centroid_x), int(centroid_y))

	def draw_points(self, image, hand_center, fingertips):
		cv2.circle(image, tuple(hand_center), 10, (0, 0, 255), -1)
		for ft in fingertips:
			cv2.circle(image, ft, 5, (0,0,255), -1)
		return image

	def track_hand(self, image, mask):
		mask = self.morph_open_close(mask)
		contours = self.find_sorted_contours(mask)
		if len(contours) != 0:

			hand_center = contours[0]
			hand_pos = self.get_centroid_of_contour(hand_center)
			hand_area = cv2.contourArea(hand_center)
			hand_radius = int(np.sqrt(hand_area/3)*2)

			hand_angle = cv2.minAreaRect(hand_center)[2]

			if len(contours) <= 6:
				finger_tips = contours[1:]
			else:
				finger_tips = contours[1:6]

			finger_tip_pos = [self.get_centroid_of_contour(cnt) for cnt in finger_tips]
			finger_tip_distance_from_center = []
			finger_tip_angles = []
			for ftp in finger_tip_pos:
				v2h = np.array(ftp)-np.array(hand_pos)
				d = np.linalg.norm(v2h)
				finger_tip_distance_from_center.append(d)
				if 5*hand_radius > d > 1.5*hand_radius:
					angle = np.rad2deg(np.arctan2(v2h[1], v2h[0]))
					finger_tip_angles.append(angle-hand_angle+90)

				fingers_detected = [False, False, False, False, False] # pinky, ring, middle, index, thumb
			for angle in finger_tip_angles:
				if 60 < angle < 180:
					fingers_detected[0] = True
				if 30 < angle < 60:
					fingers_detected[1] = True
				if -30 < angle < 30:
					fingers_detected[2] = True
				if -60 < angle < -30:
					fingers_detected[3] = True
				if -180 < angle < -60 or 90 < angle < 270:
					fingers_detected[4] = True

			count = 0
			for d in finger_tip_distance_from_center:
				if d > hand_radius*1.5:
					count += 1



			self.draw_points(image, hand_pos, finger_tip_pos)
			### TESTING STUFF
			cv2.circle(image, hand_pos, int(hand_radius*1.5), (0,255,255), 1)
			cv2.circle(image, hand_pos, hand_radius, (0,0,255)), 1

			cv2.putText(image, str(fingers_detected), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

		cv2.imshow("hand", image)

purple_cdf = ColorFinder("purple", Color(164, 5, 21), Color(222, 255, 225))
hand_tracker = HandTracker()
cap = cv2.VideoCapture(1)
try:
	dt = 1 / 25
	while True:
		t = time.time()

		#### Code goes here
		ret, frame_bgr = cap.read()
		if not ret:
			print("End of stream")
			break

		frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV_FULL)
		mask = purple_cdf.find_color(frame_hsv)
		cv2.imshow("mask", mask)
		pos = hand_tracker.track_hand(frame_bgr, mask)
		###################

		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

		dt = time.time() - t - 0.03
		FPS = 1 / dt
		print(FPS)

finally:
	cap.release()
	cv2.destroyAllWindows()
