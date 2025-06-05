import os
from typing import Optional, List

import numpy as np
import cv2
class ResultsProcessor:
    def read_yolo_obb_labels(self, results):
        """Extracts bounding box corners and class IDs from YOLO OBB results."""
        boxes = []

        for i in range(len(results[0].obb.xyxyxyxy)):  # Iterate through all detections
            class_id = int(results[0].obb.cls.cpu()[i])  # Get class ID
            corners = results[0].obb.xyxyxyxy[i].cpu().numpy()  # Extract 4 corners
            boxes.append((class_id, corners))

        return boxes

    def save_book_spines(self, image_path: str, yolo_results, output_dir: str) -> List[Optional[str]]:
        """
        Extract and save book spines to the specified directory

        Args:
            image_path: Path to the original image
            yolo_results: Results from YOLO model prediction
            output_dir: Directory to save the cropped images

        Returns:
            List of paths to saved spine images (None for failed crops)
        """
        # Get bounding boxes
        boxes = self.read_yolo_obb_labels(yolo_results)
        if not boxes:
            print("No bounding boxes found.")
            return []

        # Read original image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Extract and save each book spine
        saved_files = []
        for idx, (class_id, corners) in enumerate(boxes):
            corners_px = np.array(corners, dtype=np.float32)
            cropped = self.__crop_and_rotate_obb(image, corners_px)

            if cropped is not None and cropped.size > 0:
                output_path = os.path.join(output_dir, f"book_{idx}.jpg")
                cv2.imwrite(output_path, cropped)
                saved_files.append(output_path)
            else:
                saved_files.append(None)

        return saved_files

    def extract_book_spines(self, image_path: str, yolo_results) -> List[Optional[np.ndarray]]:
        """
        Extract all book spines from an image as numpy arrays

        Args:
            image_path: Path to the original image
            yolo_results: Results from YOLO model prediction

        Returns:
            List of numpy arrays, each containing a cropped book spine image
        """
        # Get bounding boxes
        boxes = self.read_yolo_obb_labels(yolo_results)
        if not boxes:
            print("No bounding boxes found.")
            return []

        # Read original image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Extract each book spine
        cropped_images = []
        for class_id, corners in boxes:
            corners_px = np.array(corners, dtype=np.float32)
            cropped = self.__crop_and_rotate_obb(image, corners_px)

            if cropped is not None and cropped.size > 0:
                cropped_images.append(cropped)
            else:
                cropped_images.append(None)

        return cropped_images

    def __order_corners(self, corners):
        """
        Orders bounding box corners in a consistent manner (clockwise).
        corners: np.array of shape (4, 2), in arbitrary order.
        """
        pts = corners.copy()
        cx, cy = np.mean(pts, axis=0)  # Compute centroid
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)  # Compute angles w.r.t centroid
        sorted_idx = np.argsort(angles)  # Sort by angle
        return pts[sorted_idx]

    def __crop_and_rotate_obb(self, image, corners):
        """Applies perspective transform to crop and align detected objects."""
        corners = self.__order_corners(corners)

        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners
        w1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        w2 = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
        h1 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        h2 = np.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)
        max_width = max(int(w1), int(w2))
        max_height = max(int(h1), int(h2))
        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped

