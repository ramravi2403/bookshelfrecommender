import sys

import ultralytics
from PIL import Image
from ultralytics import YOLO
import cv2
# from PIL import Image
import numpy as np

from ResultsProcessor import ResultsProcessor


def main():
    model = YOLO("./obb/v8x/yolov8x-obb-best_20240202.pt")

    image_path = "PXL_20240222_221257820.webp"


    results = model(image_path, save=False, show=False)

    results_processor = ResultsProcessor()
    boxes = results_processor.read_yolo_obb_labels(results)

    if not boxes:
        print("No bounding boxes found.")
        exit()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Process each detected object
    for idx, (class_id, corners) in enumerate(boxes):
        corners_px = np.array(corners, dtype=np.float32)

        cropped = results_processor.__crop_and_rotate_obb(image, corners_px)

        # Ensure valid crop before saving
        if cropped is not None and cropped.size > 0:
            cv2.imwrite(f'./OutputSpines3/book_{idx}.jpg', cropped)
        else:
            print(f"Skipping book_{idx}.jpg due to invalid crop.")


if __name__ == "__main__":
    main()
