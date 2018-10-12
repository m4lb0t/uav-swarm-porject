import cv2
import numpy as np
import time


def Color(h, s, v):
	return np.array([h, s, v], dtype=np.uint8)


class ColoredDotFinder:
	def __init__(self, name, lower, upper, trackbars=True):
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


red_CDF = ColoredDotFinder("red", Color(0, 127, 127), Color(20, 255, 255))
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
		mask = red_CDF.find_color(frame_hsv, True)
		cv2.imshow("BGR", frame_bgr)
		###################

		k = cv2.waitKey(1) & 0xff
		if k == 27:
			break

		dt = time.time() - t
		FPS = 1 / dt
		print(FPS)

finally:
	cap.release()
	cv2.destroyAllWindows()
