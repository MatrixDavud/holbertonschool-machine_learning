#!/usr/bin/env python3
"""YOLO algorithm implementation."""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """YOLO v3 object detection class."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the YOLO v3 model."""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process the outputs from the YOLO model for one image."""
        image_h, image_w = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]
            object_confidence = output[..., 4:5]
            class_probs = output[..., 5:]

            cx = np.tile(np.arange(grid_w).reshape(1, grid_w, 1),
                         (grid_h, 1, anchor_boxes))
            cy = np.tile(np.arange(grid_h).reshape(grid_h, 1, 1),
                         (1, grid_w, anchor_boxes))

            bx = (1 / (1 + np.exp(-tx)) + cx) / grid_w
            by = (1 / (1 + np.exp(-ty)) + cy) / grid_h

            bw = (anchors[:, 0] * np.exp(tw)) / self.model.input.shape[1]
            bh = (anchors[:, 1] * np.exp(th)) / self.model.input.shape[2]

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            box_confidences.append(1 / (1 + np.exp(-object_confidence)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter the boxes based on object and class confidence scores."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_scores_i = box_confidences[i] * box_class_probs[i]

            box_classes_i = np.argmax(box_scores_i, axis=-1)
            box_class_scores_i = np.max(box_scores_i, axis=-1)

            filtering_mask = box_class_scores_i >= self.class_t

            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(box_classes_i[filtering_mask])
            box_scores.append(box_class_scores_i[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Perform non-max suppression to remove overlapping boxes."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for c in set(box_classes):
            class_indices = np.where(box_classes == c)
            class_boxes = filtered_boxes[class_indices]
            class_box_scores = box_scores[class_indices]

            sorted_indices = np.argsort(class_box_scores)[::-1]
            class_boxes = class_boxes[sorted_indices]
            class_box_scores = class_box_scores[sorted_indices]

            keep = []

            while len(class_boxes) > 0:
                current_box = class_boxes[0]
                keep.append(0)

                x1 = np.maximum(current_box[0], class_boxes[1:, 0])
                y1 = np.maximum(current_box[1], class_boxes[1:, 1])
                x2 = np.minimum(current_box[2], class_boxes[1:, 2])
                y2 = np.minimum(current_box[3], class_boxes[1:, 3])

                inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                box1_area = (current_box[2] - current_box[0]) *\
                    (current_box[3] - current_box[1])
                box2_area = (class_boxes[1:, 2] - class_boxes[1:, 0]) *\
                    (class_boxes[1:, 3] - class_boxes[1:, 1])

                iou = inter_area / (box1_area + box2_area - inter_area)

                keep_indices = np.where(iou <= self.nms_t)[0] + 1
                class_boxes = class_boxes[keep_indices]
                class_box_scores = class_box_scores[keep_indices]

            keep_boxes = filtered_boxes[class_indices][sorted_indices[keep]]
            keep_scores = box_scores[class_indices][sorted_indices[keep]]
            keep_classes = np.full(len(keep_boxes), c)

            box_predictions.append(keep_boxes)
            predicted_box_scores.append(keep_scores)
            predicted_box_classes.append(keep_classes)

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores
