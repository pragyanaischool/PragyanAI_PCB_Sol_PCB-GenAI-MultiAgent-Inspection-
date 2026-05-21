import cv2
import numpy as np
from PIL import Image


# =======================================================
# PCB IMAGE PREPROCESSING MODULE
# =======================================================

class PCBPreprocessor:

    def __init__(self):

        self.default_width = 640
        self.default_height = 640

    # ---------------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------------

    def load_image(self, image):

        """
        Convert uploaded image into numpy array
        """

        if isinstance(image, Image.Image):

            image = np.array(image)

        return image

    # ---------------------------------------------------
    # RESIZE IMAGE
    # ---------------------------------------------------

    def resize_image(
        self,
        image,
        width=None,
        height=None
    ):

        if width is None:
            width = self.default_width

        if height is None:
            height = self.default_height

        resized = cv2.resize(
            image,
            (width, height)
        )

        return resized

    # ---------------------------------------------------
    # RGB → BGR
    # ---------------------------------------------------

    def rgb_to_bgr(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    # ---------------------------------------------------
    # BGR → RGB
    # ---------------------------------------------------

    def bgr_to_rgb(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    # ---------------------------------------------------
    # GRAYSCALE CONVERSION
    # ---------------------------------------------------

    def convert_to_gray(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        return gray

    # ---------------------------------------------------
    # GAUSSIAN BLUR
    # ---------------------------------------------------

    def apply_gaussian_blur(
        self,
        image,
        kernel_size=(5, 5)
    ):

        blurred = cv2.GaussianBlur(
            image,
            kernel_size,
            0
        )

        return blurred

    # ---------------------------------------------------
    # MEDIAN BLUR
    # ---------------------------------------------------

    def apply_median_blur(
        self,
        image,
        kernel_size=5
    ):

        median = cv2.medianBlur(
            image,
            kernel_size
        )

        return median

    # ---------------------------------------------------
    # HISTOGRAM EQUALIZATION
    # ---------------------------------------------------

    def histogram_equalization(self, gray):

        equalized = cv2.equalizeHist(gray)

        return equalized

    # ---------------------------------------------------
    # CLAHE ENHANCEMENT
    # ---------------------------------------------------

    def apply_clahe(self, gray):

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        return enhanced

    # ---------------------------------------------------
    # EDGE DETECTION
    # ---------------------------------------------------

    def detect_edges(
        self,
        image,
        low_threshold=50,
        high_threshold=150
    ):

        edges = cv2.Canny(
            image,
            low_threshold,
            high_threshold
        )

        return edges

    # ---------------------------------------------------
    # THRESHOLDING
    # ---------------------------------------------------

    def apply_threshold(self, image):

        _, thresh = cv2.threshold(
            image,
            120,
            255,
            cv2.THRESH_BINARY_INV
        )

        return thresh

    # ---------------------------------------------------
    # ADAPTIVE THRESHOLD
    # ---------------------------------------------------

    def adaptive_threshold(self, image):

        adaptive = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        return adaptive

    # ---------------------------------------------------
    # MORPHOLOGICAL CLEANUP
    # ---------------------------------------------------

    def morphological_cleanup(self, image):

        kernel = np.ones((3, 3), np.uint8)

        morph = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            kernel
        )

        morph = cv2.morphologyEx(
            morph,
            cv2.MORPH_OPEN,
            kernel
        )

        return morph

    # ---------------------------------------------------
    # REMOVE NOISE
    # ---------------------------------------------------

    def remove_noise(self, image):

        denoised = cv2.fastNlMeansDenoising(
            image,
            None,
            10,
            7,
            21
        )

        return denoised

    # ---------------------------------------------------
    # SHARPEN IMAGE
    # ---------------------------------------------------

    def sharpen_image(self, image):

        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        sharpened = cv2.filter2D(
            image,
            -1,
            kernel
        )

        return sharpened

    # ---------------------------------------------------
    # NORMALIZE IMAGE
    # ---------------------------------------------------

    def normalize_image(self, image):

        normalized = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return normalized

    # ---------------------------------------------------
    # IMAGE AUGMENTATION
    # ---------------------------------------------------

    def horizontal_flip(self, image):

        return cv2.flip(image, 1)

    def vertical_flip(self, image):

        return cv2.flip(image, 0)

    def rotate_image(
        self,
        image,
        angle=90
    ):

        height, width = image.shape[:2]

        center = (
            width // 2,
            height // 2
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height)
        )

        return rotated

    # ---------------------------------------------------
    # FULL PREPROCESSING PIPELINE
    # ---------------------------------------------------

    def preprocess_pipeline(self, image):

        # Load

        image = self.load_image(image)

        # Resize

        resized = self.resize_image(image)

        # RGB → BGR

        bgr = self.rgb_to_bgr(resized)

        # Gray

        gray = self.convert_to_gray(bgr)

        # Denoise

        denoised = self.remove_noise(gray)

        # Blur

        blurred = self.apply_gaussian_blur(
            denoised
        )

        # Histogram Equalization

        equalized = self.histogram_equalization(
            blurred
        )

        # CLAHE

        clahe = self.apply_clahe(equalized)

        # Sharpen

        sharpened = self.sharpen_image(
            clahe
        )

        # Normalize

        normalized = self.normalize_image(
            sharpened
        )

        # Edge Detection

        edges = self.detect_edges(
            normalized
        )

        # Threshold

        threshold = self.apply_threshold(
            normalized
        )

        # Adaptive Threshold

        adaptive = self.adaptive_threshold(
            normalized
        )

        # Morphological Cleanup

        morphology = (
            self.morphological_cleanup(
                adaptive
            )
        )

        result = {
            "original": image,
            "resized": resized,
            "bgr": bgr,
            "gray": gray,
            "denoised": denoised,
            "blurred": blurred,
            "equalized": equalized,
            "clahe": clahe,
            "sharpened": sharpened,
            "normalized": normalized,
            "edges": edges,
            "threshold": threshold,
            "adaptive_threshold": adaptive,
            "morphology": morphology
        }

        return result


# =======================================================
# GLOBAL FUNCTIONS
# =======================================================

preprocessor = PCBPreprocessor()


def preprocess_pipeline(image):

    return preprocessor.preprocess_pipeline(
        image
    )


# =======================================================
# TESTING
# =======================================================

if __name__ == "__main__":

    # Dummy Black Image

    test_image = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )

    result = preprocess_pipeline(
        test_image
    )

    print("Preprocessing Completed")

    print(result.keys())
  
