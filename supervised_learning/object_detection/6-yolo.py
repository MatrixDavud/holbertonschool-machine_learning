#!/usr/bin/env python3
"""YOLO algorithm implementation."""
import numpy as np
from tensorflow import keras as K
import os
import cv2


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
        """Apply Non-Maximum Suppression to filter overlapping boxes."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idx = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]
            cls_classes = box_classes[idx]

            sort_idx = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sort_idx]
            cls_scores = cls_scores[sort_idx]
            cls_classes = cls_classes[sort_idx]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls_classes[0])
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                ious = self._iou(cls_boxes[0], cls_boxes[1:])

                keep_idx = np.where(ious < self.nms_t)[0]
                cls_boxes = cls_boxes[keep_idx + 1]
                cls_scores = cls_scores[keep_idx + 1]
                cls_classes = cls_classes[keep_idx + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def _iou(self, box, boxes):
        """Calculate Intersection over Union between one box and others."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = box_area + boxes_area - intersection

        iou = intersection / union

        return iou

    @staticmethod
    def load_images(folder_path):
        """Load images from a folder."""
        images = []
        image_paths = []

        files = os.listdir(folder_path)

        for filename in sorted(files):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                image = cv2.imread(file_path)

                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images for YOLO model."""
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display image with bounding boxes, class names, and scores."""
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label = "{} {:.2f}".format(class_name, score)

            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')

            save_path = os.path.join('detections',
                                     os.path.basename(file_name))
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()
