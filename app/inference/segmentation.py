import cv2
import numpy as np


# =========================================================
# PCB SEGMENTATION MODULE
# =========================================================

class PCBSegmentation:

    def __init__(self):

        self.min_contour_area = 100

    # -----------------------------------------------------
    # PCB REGION SEGMENTATION
    # -----------------------------------------------------

    def segment_pcb_board(self, image):

        """
        Segment PCB board from background
        """

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        # Typical Green PCB Range

        lower_green = np.array(
            [35, 40, 40]
        )

        upper_green = np.array(
            [90, 255, 255]
        )

        mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        segmented = cv2.bitwise_and(
            image,
            image,
            mask=mask
        )

        return segmented, mask

    # -----------------------------------------------------
    # COPPER TRACE SEGMENTATION
    # -----------------------------------------------------

    def segment_copper_traces(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        _, thresh = cv2.threshold(
            blurred,
            120,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        morph = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel
        )

        return morph

    # -----------------------------------------------------
    # SOLDER MASK SEGMENTATION
    # -----------------------------------------------------

    def segment_solder_mask(self, image):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        lower_green = np.array(
            [35, 40, 40]
        )

        upper_green = np.array(
            [90, 255, 255]
        )

        solder_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        return solder_mask

    # -----------------------------------------------------
    # DRILL HOLE SEGMENTATION
    # -----------------------------------------------------

    def segment_drill_holes(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (7, 7),
            1
        )

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=5,
            maxRadius=30
        )

        hole_mask = np.zeros_like(gray)

        detected_holes = []

        if circles is not None:

            circles = np.round(
                circles[0, :]
            ).astype("int")

            for (x, y, r) in circles:

                detected_holes.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "radius": int(r)
                    }
                )

                cv2.circle(
                    hole_mask,
                    (x, y),
                    r,
                    255,
                    -1
                )

        return hole_mask, detected_holes

    # -----------------------------------------------------
    # EDGE-BASED DEFECT SEGMENTATION
    # -----------------------------------------------------

    def segment_edge_defects(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        morph = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel
        )

        return morph

    # -----------------------------------------------------
    # CONTOUR REGION EXTRACTION
    # -----------------------------------------------------

    def extract_contours(self, binary_image):

        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_contours = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area > self.min_contour_area:

                filtered_contours.append(
                    contour
                )

        return filtered_contours

    # -----------------------------------------------------
    # CREATE DEFECT MASK
    # -----------------------------------------------------

    def create_defect_mask(
        self,
        image,
        contours
    ):

        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            contours,
            -1,
            255,
            thickness=cv2.FILLED
        )

        return mask

    # -----------------------------------------------------
    # HEATMAP GENERATION
    # -----------------------------------------------------

    def generate_heatmap(
        self,
        image,
        defect_mask
    ):

        heatmap = cv2.applyColorMap(
            defect_mask,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image,
            0.7,
            heatmap,
            0.3,
            0
        )

        return overlay

    # -----------------------------------------------------
    # ANNOTATE SEGMENTATION
    # -----------------------------------------------------

    def annotate_segmentation(
        self,
        image,
        contours,
        label="Defect"
    ):

        annotated = image.copy()

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return annotated

    # -----------------------------------------------------
    # FULL SEGMENTATION PIPELINE
    # -----------------------------------------------------

    def run_segmentation_pipeline(
        self,
        image
    ):

        # PCB Board

        pcb_segment, pcb_mask = (
            self.segment_pcb_board(
                image
            )
        )

        # Copper Traces

        copper_mask = (
            self.segment_copper_traces(
                image
            )
        )

        # Solder Mask

        solder_mask = (
            self.segment_solder_mask(
                image
            )
        )

        # Drill Holes

        hole_mask, holes = (
            self.segment_drill_holes(
                image
            )
        )

        # Edge Defects

        edge_defects = (
            self.segment_edge_defects(
                image
            )
        )

        # Contours

        contours = self.extract_contours(
            edge_defects
        )

        # Defect Mask

        defect_mask = self.create_defect_mask(
            image,
            contours
        )

        # Heatmap

        heatmap = self.generate_heatmap(
            image,
            defect_mask
        )

        # Annotated

        annotated = (
            self.annotate_segmentation(
                image,
                contours,
                label="PCB Defect"
            )
        )

        result = {
            "pcb_segment": pcb_segment,
            "pcb_mask": pcb_mask,
            "copper_mask": copper_mask,
            "solder_mask": solder_mask,
            "hole_mask": hole_mask,
            "detected_holes": holes,
            "edge_defects": edge_defects,
            "contours": contours,
            "defect_mask": defect_mask,
            "heatmap": heatmap,
            "annotated_image": annotated
        }

        return result


# =========================================================
# GLOBAL SEGMENTATION INSTANCE
# =========================================================

segmenter = PCBSegmentation()


# =========================================================
# GLOBAL FUNCTION
# =========================================================

def run_segmentation_pipeline(image):

    return segmenter.run_segmentation_pipeline(
        image
    )


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    # Dummy Black Image

    test_image = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )

    result = run_segmentation_pipeline(
        test_image
    )

    print("Segmentation Completed")

    print(result.keys())
  
