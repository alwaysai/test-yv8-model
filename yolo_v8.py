import edgeiq
from edgeiq import ObjectDetectionPostProcessParams, ObjectDetectionPreProcessParams

from typing import List
import numpy as np
import cv2
import time


def rescale_boxes(boxes, input_size, img_size):

    # Rescale boxes to original image dimensions
    input_shape = np.array([input_size[0], input_size[1], input_size[0], input_size[1]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_size[0], img_size[1], img_size[0], img_size[1]])
    return boxes


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def extract_boxes(predictions, input_size, image_size):
    # Extract boxes from predictions
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes, input_size, image_size)

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    return boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def yolo_v8_pre_process(params: ObjectDetectionPreProcessParams) -> np.ndarray:
    start_time = time.time()
    input_img = cv2.cvtColor(params.image, cv2.COLOR_BGR2RGB)
    # Resize input image
    input_img = cv2.resize(input_img, params.size)
    # Scale input pixel values to 0 to 1
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    print("[INFO] preprocessing time %s seconds" % (time.time() - start_time))
    return input_tensor


def yolo_v8_pre_process_trt(params: ObjectDetectionPreProcessParams) -> np.ndarray:
    start_time = time.time()
    input_img = cv2.cvtColor(params.image, cv2.COLOR_BGR2RGB)
    # Resize input image
    input_img = cv2.resize(input_img, params.size)
    # Scale input pixel values to 0 to 1
    input_img = input_img / 255.0
    input_tensor = input_img.transpose(2, 0, 1).ravel()
    print("[INFO] preprocessing time %s seconds" % (time.time() - start_time))
    return input_tensor


def yolo_v8_post_process(params: ObjectDetectionPostProcessParams):
    start_time = time.time()
    boxes: List[edgeiq.BoundingBox] = []
    confidences: List[float] = []
    indexes: List[int] = []

    results: np.ndarray = params.results
    predictions = np.squeeze(results[0])
    predictions = predictions.transpose()

    box_scores = predictions[:, 4:]
    possibility = np.where(box_scores >= params.confidence_level)
    detections = predictions[possibility[0]]

    scores = predictions[possibility[0], [possibility[1] + 4]]
    class_ids = possibility[1]

    if len(scores) == 0:
        return [], [], []

    # Get bounding boxes for each object
    v7_boxes = extract_boxes(detections, params.model_input_size, (params.image.shape[1], params.image.shape[0]))

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(v7_boxes, scores[0], params.overlap_threshold)
    v7_final_boxes = v7_boxes[indices].astype(int)
    v7_final_scores = scores[0][indices]
    v7_final_class_ids = class_ids[indices]

    boxes = [edgeiq.BoundingBox(x[0], x[1], x[2], x[3]) for x in v7_final_boxes]
    confidences = v7_final_scores.tolist()
    indexes = v7_final_class_ids.tolist()
    print("[INFO] post processing time %s seconds" % (time.time() - start_time))
    return boxes, confidences, indexes
