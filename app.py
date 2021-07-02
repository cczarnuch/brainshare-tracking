# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import process_image, get_classes, draw, convert_boxes

from YOLOv3.model.yolo_model import YOLO

from deep_sort.deep_sort import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections


# %%
def detect_image(frame, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        frame: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        processed_frame: processed image.
    """
    processed_frame = process_image(frame)

    start = time.time()
    boxes, classes, scores = yolo.predict(processed_frame, frame.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(frame, boxes, scores, classes, all_classes)

    return frame, boxes, scores, classes


# %%
def tracking(video, yolo, all_classes):
    video_path = os.path.join("videos", "test", video) # Get absolute path of input video file
    camera = cv2.VideoCapture(video_path) # Open video path using OpenCV
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE) # Create OpenCV window for displaying output video

    # Prepare for saving the detected video
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))) # Get size of input video
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'h264') # Set codec for output video

    vout = cv2.VideoWriter() # Initialize video writer object
    vout.open(os.path.join("videos", "res", video), fourcc, fps, size, True) # Open video file to write to

    # Loop through an infinite while loop until the video stream ends
    while True:
        res, frame = camera.read() # Read next frame from video stream

        if not res:
            break # If there is no next frame, exit the loop

        frame, boxes, scores, classes = detect_image(frame, yolo, all_classes) # Perform detections using YOLOv3 to generate the bounding boxes, confidence scores, and class names of each detected object

        # Run conversions and create Detection objects
        features = encoder(frame, boxes)
        detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]

        # Init colour map
        cmap = plt.get_cmap('tab20b')
        colours = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # Non-max suppression
        boxes = np.array([d.tlwh for d in detections]) # Generate a numpy array of detections using TLWH coordinates
        scores = np.array([d.confidence for d in detections]) # Generate a numpy array of confidence scores
        indices = preprocessing.non_max_suppression(boxes, classes, nms_max_overlap, scores) # Find indices of max-confidence level bounding box
        detections = [detections[i] for i in indices] # Filter detections based on max confidence level/non-max suppression

        tracker.predict() # Predict the next location of the bounding boxes using the deep_sort tracker
        tracker.update(detections) # Update the tracker with the actual locations of the objects

        # Check for confirmed tracks and draw on output stream
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: # If track is not confirmed, do not draw on frame
                continue
            # If track is confirmed and "live", draw track on frame
            bbox = track.to_tlbr()
            colour = colours[int(track.track_id) % len(colours)]
            colour = [i * 255 for i in colour]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colour, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(track.track_id)))*17, int(bbox[1])), colour, -1)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
        # Send updated frame to output and write to video file
        cv2.imshow("tracking", frame)
        vout.write(frame)

        if cv2.waitKey(110) & 0xff == 27:
                break

    vout.release()
    camera.release()
    cv2.destroyAllWindows()


# %%
yolo = YOLO(0.6, 0.5, 'YOLOv3/data/yolo.h5')
file = 'YOLOv3/data/coco_classes.txt'
all_classes = get_classes(file)

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0

model_filename = 'deep_sort/resources/networks/mars-small128.pb'
encoder = generate_detections.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


# detect videos one at a time in videos/test folder    
video = 'library1.mp4'
tracking(video, yolo, all_classes)
