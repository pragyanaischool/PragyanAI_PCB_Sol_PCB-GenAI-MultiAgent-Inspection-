import os
import cv2
import uuid
import numpy as np
from datetime import datetime


# =========================================================
# HELPER UTILITIES
# =========================================================

class PCBHelper:

    def __init__(self):

        self.allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff"
        ]

    # -----------------------------------------------------
    # VALIDATE FILE EXTENSION
    # -----------------------------------------------------

    def validate_file(self, filename):

        extension = os.path.splitext(
            filename
        )[1].lower()

        return extension in (
            self.allowed_extensions
        )

    # -----------------------------------------------------
    # GENERATE UNIQUE ID
    # -----------------------------------------------------

    def generate_unique_id(self):

        return str(uuid.uuid4())

    # -----------------------------------------------------
    # CURRENT TIMESTAMP
    # -----------------------------------------------------

    def current_timestamp(self):

        return datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # -----------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # -----------------------------------------------------

    def create_directory(self, path):

        if not os.path.exists(path):

            os.makedirs(path)

    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    def save_image(
        self,
        image,
        path
    ):

        cv2.imwrite(path, image)

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    def load_image(self, path):

        image = cv2.imread(path)

        return image

    # -----------------------------------------------------
    # RESIZE IMAGE
    # -----------------------------------------------------

    def resize_image(
        self,
        image,
        width=640,
        height=640
    ):

        resized = cv2.resize(
            image,
            (width, height)
        )

        return resized

    # -----------------------------------------------------
    # NORMALIZE IMAGE
    # -----------------------------------------------------

    def normalize_image(self, image):

        normalized = image / 255.0

        return normalized

    # -----------------------------------------------------
    # CONVERT TO RGB
    # -----------------------------------------------------

    def convert_bgr_to_rgb(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    # -----------------------------------------------------
    # CONVERT TO BGR
    # -----------------------------------------------------

    def convert_rgb_to_bgr(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    # -----------------------------------------------------
    # CALCULATE IMAGE STATISTICS
    # -----------------------------------------------------

    def image_statistics(self, image):

        stats = {
            "shape": image.shape,
            "height": image.shape[0],
            "width": image.shape[1],
            "channels": image.shape[2]
            if len(image.shape) == 3
            else 1,
            "min_pixel": int(np.min(image)),
            "max_pixel": int(np.max(image)),
            "mean_pixel": float(np.mean(image))
        }

        return stats

    # -----------------------------------------------------
    # GENERATE DEFECT ID
    # -----------------------------------------------------

    def generate_defect_id(
        self,
        defect_type
    ):

        uid = str(uuid.uuid4())[:8]

        return (
            f"{defect_type.upper()}-{uid}"
        )

    # -----------------------------------------------------
    # FORMAT DETECTION RESULT
    # -----------------------------------------------------

    def format_detection_result(
        self,
        defect,
        confidence,
        severity
    ):

        result = {
            "defect_id":
                self.generate_defect_id(
                    defect
                ),
            "defect": defect,
            "confidence": confidence,
            "severity": severity,
            "timestamp":
                self.current_timestamp()
        }

        return result

    # -----------------------------------------------------
    # MERGE RECOMMENDATIONS
    # -----------------------------------------------------

    def merge_recommendations(
        self,
        recommendation_lists
    ):

        merged = []

        for rec_list in recommendation_lists:

            merged.extend(rec_list)

        unique = list(set(merged))

        return unique

    # -----------------------------------------------------
    # HEALTH SCORE
    # -----------------------------------------------------

    def calculate_health_score(
        self,
        critical,
        high,
        medium,
        low
    ):

        score = 100

        score -= critical * 20
        score -= high * 10
        score -= medium * 5
        score -= low * 2

        return max(score, 0)

    # -----------------------------------------------------
    # FORMAT SUMMARY
    # -----------------------------------------------------

    def generate_summary(
        self,
        total,
        critical,
        high,
        medium,
        low
    ):

        return {
            "total_defects": total,
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "health_score":
                self.calculate_health_score(
                    critical,
                    high,
                    medium,
                    low
                )
        }

    # -----------------------------------------------------
    # DEBUG LOGGER
    # -----------------------------------------------------

    def log(self, message):

        timestamp = (
            self.current_timestamp()
        )

        print(
            f"[{timestamp}] {message}"
        )


# =========================================================
# GLOBAL INSTANCE
# =========================================================

helper = PCBHelper()


# =========================================================
# GLOBAL FUNCTIONS
# =========================================================

def validate_file(filename):

    return helper.validate_file(
        filename
    )


def generate_unique_id():

    return helper.generate_unique_id()


def current_timestamp():

    return helper.current_timestamp()


def image_statistics(image):

    return helper.image_statistics(
        image
    )


def log(message):

    helper.log(message)


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    dummy_image = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )

    stats = image_statistics(
        dummy_image
    )

    print(stats)
  
