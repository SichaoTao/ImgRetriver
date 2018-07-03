import cv2
import numpy

class Descriptor:


	def __init__(self, bins):
		self.bins = bins


	def histogram(self, image, mask):
		"""
		The Histogram function is responsible for
		Extracting the 3-dimensional color histogram
		from the image with the image and the mask
		being the parameters for the function.
		"""
		histogram = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
					[0, 180, 0, 256, 0, 256])
		histogram = cv2.normalize(histogram).flatten()

		return histogram


	def describe(self, image):
		#cv2Color in openCV converts from one color space
		#to another color space.
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		#Extracting the dimensions of the input image
		height, width = image.shape[:2]
		#Extracting the center of the image
		center_X, center_Y = int(0.5 * width), int(0.5 * height)


		#Dividing the image into five segments to produce
		#a region based histogram. The image is divided into
		#top-left, top-right, bottom-left, bottom-right and center
		#by dividing the regions into four rectangle parts and then
		#applying an elliptical mask in the center

		segments = [(0, center_X, 0, center_Y),
					(center_X, width, 0, center_Y),
					(center_X, width, center_Y, height),
					(0, center_X, center_Y, height)
				]

		# Contrucing an elliptical mask at the center
		# of the the four rectangles i.e at the center
		# of the image. The radius of the ellipse is
		# 75% of the width and height of the image.

		axes_X, axes_Y = int((width * 0.75) // 2), int((height * 0.75) // 2)
		elliptical_mask = numpy.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(elliptical_mask, (center_X, center_Y), (axes_X, axes_Y), 0, 0, 350, 255, -1)


		for (start_X, end_X, start_Y, end_Y) in segments:

			corner_mask = numpy.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(corner_mask, (start_X, start_Y), (end_X, end_Y), 255, -1)
			corner_mask = cv2.subtract(corner_mask, elliptical_mask)


			histogram = self.histogram(image, corner_mask)
			features.extend(histogram)

		# Extracting features from the elliptical region in the image
		histogram = self.histogram(image, elliptical_mask)
		features.extend(histogram)


		return features