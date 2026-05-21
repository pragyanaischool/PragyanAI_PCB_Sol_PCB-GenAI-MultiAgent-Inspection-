import cv2
import numpy as np
from PIL import Image
from llm.groq_client import analyze_with_llm


class PadDamageAgent:

    def __init__(self):

        self.agent_name = "Pad Damage Agent"

        self.defect_type = "Pad Damage"

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
    # PAD DAMAGE DETECTION
    # -------------------------------------------------------

    def detect_pad_damage(self, image):

        image_bgr, gray, blurred = (
            self.preprocess_image(image)
        )

        # Threshold

        _, thresh = cv2.threshold(
            blurred,
            120,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Morphological Operations

        kernel = np.ones((3, 3), np.uint8)

        morph = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel
        )

        morph = cv2.dilate(
            morph,
            kernel,
            iterations=1
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

            # Ignore tiny regions

            if area < 120:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter == 0:
                continue

            circularity = (
                4 * np.pi * area /
                (perimeter * perimeter)
            )

            x, y, w, h = cv2.boundingRect(contour)

            # Heuristic Pad Damage Logic
            # Pads are generally circular

            if circularity < 0.65:

                confidence = round(
                    min(area / 1500, 0.99),
                    2
                )

                if confidence >= self.confidence_threshold:

                    defect_type = (
                        self.classify_pad_damage(
                            circularity,
                            area
                        )
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
                        "circularity": round(
                            circularity,
                            2
                        ),
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
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        annotated_image,
                        f"{defect_type} {confidence}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

        return detections, annotated_image

    # -------------------------------------------------------
    # CLASSIFY PAD DAMAGE
    # -------------------------------------------------------

    def classify_pad_damage(
        self,
        circularity,
        area
    ):

        if circularity < 0.30:
            return "Missing Pad"

        elif circularity < 0.50:
            return "Lifted Pad"

        else:
            return "Cracked Pad"

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
            No Pad Damage defects detected.
            PCB pad integrity appears stable.
            """

        prompt = f"""
        You are a PCB manufacturing reliability expert.

        Analyze the following PCB pad damage defects:

        {detections}

        Explain:

        1. Root causes
        2. Soldering-related issues
        3. Mechanical stress problems
        4. Reliability impact
        5. Electrical connectivity risks
        6. Prevention methods
        7. Corrective actions
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
            "Reduce excessive soldering heat.",
            "Optimize reflow temperature profile.",
            "Inspect PCB pad adhesion quality.",
            "Reduce mechanical handling stress.",
            "Improve solder paste consistency.",
            "Increase AOI inspection coverage."
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
            "total_pad_damage_defects": total,
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
            self.detect_pad_damage(image)
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

    agent = PadDamageAgent()

    result = agent.analyze(test_image)

    print(result["summary"])
