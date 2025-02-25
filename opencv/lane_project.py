import numpy as np
import pandas as pd
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy

def region_selection(image):
	mask = np.zeros_like(image)
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
	rows, cols = image.shape[:2]
	bottom_left = [cols * 0.1, rows * 0.95]
	top_left	 = [cols * 0.4, rows * 0.6]
	bottom_right = [cols * 0.9, rows * 0.95]
	top_right = [cols * 0.6, rows * 0.6]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def hough_transform(image):
	rho = 1
	theta = np.pi / 180
	threshold = 20  # Pode ser reduzido para detectar mais linhas
	minLineLength = 10  # Reduzido para pegar linhas menores
	maxLineGap = 250  # Ajustado para permitir gaps maiores

	lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

	if lines is None:
		print("Aviso: Nenhuma linha detectada na imagem!")

	return lines


def average_slope_intercept(lines):
	if lines is None or len(lines) == 0:
		print("Aviso: Nenhuma linha detectada pela Transformada de Hough!")
		return None, None  # Retorna None para evitar erro de iteração

	left_lines = [] #(slope, intercept)
	left_weights = [] #(length,)
	right_lines = [] #(slope, intercept)
	right_weights = [] #(length,)

	for line in lines:
		for x1, y1, x2, y2 in line:
			if x1 == x2:
				continue
			# calculating slope of a line
			slope = (y2 - y1) / (x2 - x1)
			# calculating intercept of a line
			intercept = y1 - (slope * x1)
			# calculating length of a line
			length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
			# slope of left lane is negative and for right lane slope is positive
			if slope < 0:
				left_lines.append((slope, intercept))
				left_weights.append((length))
			else:
				right_lines.append((slope, intercept))
				right_weights.append((length))
	#
	left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
	right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
	return left_lane, right_lane

def pixel_points(y1, y2, line):
	if line is None:
		return None
	slope, intercept = line
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	y1 = int(y1)
	y2 = int(y2)
	return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
	left_lane, right_lane = average_slope_intercept(lines)
	y1 = image.shape[0]
	y2 = y1 * 0.6
	left_line = pixel_points(y1, y2, left_lane)
	right_line = pixel_points(y1, y2, right_lane)
	return left_line, right_line


def draw_lane_lines(image, lines, color=[139, 0, 0], thickness=12):
	line_image = np.zeros_like(image)
	for line in lines:
		if line is not None:
			cv2.line(line_image, *line, color, thickness)
	return cv2.addWeighted(image, 0.7, line_image, 1.5, 0.0)

def frame_processor(image):
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel_size = 5
	blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
	low_t = 50
	high_t = 150
	edges = cv2.Canny(blur, low_t, high_t)
	region = region_selection(edges)
	hough = hough_transform(region)
	result = draw_lane_lines(image, lane_lines(image, hough))
	return result

# driver function
def process_video(test_video, output_video):
	input_video = VideoFileClip(test_video, audio=False)
	processed = input_video.fl_image(frame_processor)
	processed.write_videofile(output_video, audio=False)

# calling driver function
process_video('input.mp4','output.mp4')
