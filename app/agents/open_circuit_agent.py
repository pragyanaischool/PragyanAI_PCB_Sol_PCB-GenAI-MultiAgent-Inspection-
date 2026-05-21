import cv2
import numpy as np
from PIL import Image
from llm.groq_client import analyze_with_llm


class OpenCircuitAgent:

    def __init__(self):

        self.agent_name = "Open Circuit Agent"

        self.defect_type = "Open Circuit"

        self.confidence_threshold = 0.50

    # -------------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------------

    def preprocess_image(self, image):

        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert RGB → BGR
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
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

        return image_bgr, gray, blurred

    # -------------------------------------------------------
    # OPEN CIRCUIT DETECTION
    # -------------------------------------------------------

    def detect_open_circuit(self, image):

        image_bgr, gray, blurred = self.preprocess_image(
            image
        )

        # Edge Detection

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        # Morphological Close

        kernel = np.ones((3, 3), np.uint8)

        morph = cv2.morphologyEx(
            edges,
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

            # Ignore tiny contours

            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / (h + 1)

            # Heuristic Open Circuit Logic

            if aspect_ratio > 2.5 or aspect_ratio < 0.4:

                confidence = round(
                    min(area / 1000, 0.99),
                    2
                )

                if confidence >= self.confidence_threshold:

                    detection = {
                        "defect": "Open Circuit",
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
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        annotated_image,
                        f"Open Circuit {confidence}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        return detections, annotated_image

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
            No Open Circuit defect detected.
            PCB trace continuity appears stable.
            """

        prompt = f"""
        You are a PCB Manufacturing Expert.

        Analyze the following Open Circuit defects:

        {detections}

        Explain:

        1. Root cause
        2. Manufacturing issue
        3. Impact on PCB functionality
        4. Electrical risk
        5. Preventive methods
        6. Recommended corrective actions
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
            "Optimize etching process calibration.",
            "Inspect copper trace continuity.",
            "Reduce excessive thermal stress.",
            "Improve AOI inspection quality.",
            "Monitor copper thickness consistency.",
            "Validate PCB design spacing rules."
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
            "total_open_circuit_defects": total,
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
            self.detect_open_circuit(image)
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

    agent = OpenCircuitAgent()

    result = agent.analyze(test_image)

    print(result["summary"])
