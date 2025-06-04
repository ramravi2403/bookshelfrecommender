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

