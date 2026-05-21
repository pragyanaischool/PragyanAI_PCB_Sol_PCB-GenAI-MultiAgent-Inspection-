import cv2
import numpy as np
from PIL import Image
from llm.groq_client import analyze_with_llm


class SolderMaskAgent:

    def __init__(self):

        self.agent_name = "Solder Mask Agent"

        self.defect_type = "Solder Mask Defect"

        self.confidence_threshold = 0.50

    # -------------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------------

    def preprocess_image(self, image):

        if isinstance(image, Image.Image):
            image = np.array(image)

        # RGB → BGR

        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        hsv = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2HSV
        )

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        return image_bgr, hsv, gray, blurred

    # -------------------------------------------------------
    # SOLDER MASK DEFECT DETECTION
    # -------------------------------------------------------

    def detect_soldermask_defect(self, image):

        image_bgr, hsv, gray, blurred = (
            self.preprocess_image(image)
        )

        # Green Mask Segmentation
        # Typical PCB solder mask is green

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])

        mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        # Invert Mask

        inverted_mask = cv2.bitwise_not(mask)

        # Morphological Cleanup

        kernel = np.ones((5, 5), np.uint8)

        morph = cv2.morphologyEx(
            inverted_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        morph = cv2.morphologyEx(
            morph,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Contour Detection

        contours, _ = cv2.findContours(
            morph,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        annotated_image = image_bgr.copy()

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore very small regions

            if area < 150:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            confidence = round(
                min(area / 2000, 0.99),
                2
            )

            if confidence >= self.confidence_threshold:

                defect_type = self.classify_mask_issue(
                    w,
                    h,
                    area
                )

                detection = {
                    "defect": defect_type,
                    "confidence": confidence,
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "severity": self.classify_severity(
                        confidence
                    )
                }

                detections.append(detection)

                # Draw Bounding Box

                cv2.rectangle(
                    annotated_image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    annotated_image,
                    f"{defect_type} {confidence}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

        return detections, annotated_image

    # -------------------------------------------------------
    # CLASSIFY SOLDER MASK ISSUE
    # -------------------------------------------------------

    def classify_mask_issue(self, width, height, area):

        aspect_ratio = width / (height + 1)

        if area > 3000:
            return "Missing Solder Mask"

        elif aspect_ratio > 2.5:
            return "Solder Mask Misalignment"

        else:
            return "Solder Mask Bubble"

    # -------------------------------------------------------
    # SEVERITY CLASSIFICATION
    # -------------------------------------------------------

    def classify_severity(self, confidence):

        if confidence >= 0.90:
            return "Critical"

        elif confidence >= 0.75:
            return "High"

        elif confidence >= 0.60:
            return "Medium"

        else:
            return "Low"

    # -------------------------------------------------------
    # ROOT CAUSE ANALYSIS USING LLM
    # -------------------------------------------------------

    def root_cause_analysis(self, detections):

        if len(detections) == 0:

            return """
            No Solder Mask defects detected.
            PCB solder mask quality appears stable.
            """

        prompt = f"""
        You are a PCB manufacturing quality expert.

        Analyze the following solder mask defects:

        {detections}

        Explain:

        1. Root causes
        2. Coating process issues
        3. Alignment problems
        4. Reliability impact
        5. Electrical risks
        6. Preventive actions
        7. Corrective measures
        """

        try:

            response = analyze_with_llm(prompt)

            return response

        except Exception as e:

            return f"LLM Analysis Error: {str(e)}"

    # -------------------------------------------------------
    # RECOMMENDATIONS
    # -------------------------------------------------------

    def generate_recommendations(self):

        recommendations = [
            "Improve solder mask alignment calibration.",
            "Optimize mask coating consistency.",
            "Monitor curing temperature carefully.",
            "Inspect screen-printing precision.",
            "Improve AOI inspection frequency.",
            "Reduce contamination during coating."
        ]

        return recommendations

    # -------------------------------------------------------
    # SUMMARY GENERATION
    # -------------------------------------------------------

    def generate_summary(self, detections):

        total = len(detections)

        critical = len(
            [
                d for d in detections
                if d["severity"] == "Critical"
            ]
        )

        high = len(
            [
                d for d in detections
                if d["severity"] == "High"
            ]
        )

        medium = len(
            [
                d for d in detections
                if d["severity"] == "Medium"
            ]
        )

        low = len(
            [
                d for d in detections
                if d["severity"] == "Low"
            ]
        )

        summary = {
            "total_soldermask_defects": total,
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low
        }

        return summary

    # -------------------------------------------------------
    # MAIN ANALYSIS PIPELINE
    # -------------------------------------------------------

    def analyze(self, image):

        detections, annotated_image = (
            self.detect_soldermask_defect(image)
        )

        llm_analysis = self.root_cause_analysis(
            detections
        )

        recommendations = (
            self.generate_recommendations()
        )

        summary = self.generate_summary(
            detections
        )

        result = {
            "agent_name": self.agent_name,
            "defect_type": self.defect_type,
            "detections": detections,
            "summary": summary,
            "llm_analysis": llm_analysis,
            "recommendations": recommendations,
            "annotated_image": cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )
        }

        return result


# -------------------------------------------------------
# TESTING
# -------------------------------------------------------

if __name__ == "__main__":

    test_image = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )

    agent = SolderMaskAgent()

    result = agent.analyze(test_image)

    print(result["summary"])
