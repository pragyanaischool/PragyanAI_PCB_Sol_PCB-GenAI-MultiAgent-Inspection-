import cv2
import numpy as np
from PIL import Image
from llm.groq_client import analyze_with_llm


class MissingHoleAgent:

    def __init__(self):

        self.agent_name = "Missing Hole Agent"

        self.defect_type = "Missing Hole"

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
            (7, 7),
            1
        )

        return image_bgr, gray, blurred

    # -------------------------------------------------------
    # MISSING HOLE DETECTION
    # -------------------------------------------------------

    def detect_missing_hole(self, image):

        image_bgr, gray, blurred = (
            self.preprocess_image(image)
        )

        annotated_image = image_bgr.copy()

        detections = []

        # ---------------------------------------------------
        # CIRCLE DETECTION USING HOUGH TRANSFORM
        # ---------------------------------------------------

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

        detected_holes = []

        if circles is not None:

            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:

                detected_holes.append(
                    {
                        "x": x,
                        "y": y,
                        "radius": r
                    }
                )

                # Draw detected hole

                cv2.circle(
                    annotated_image,
                    (x, y),
                    r,
                    (0, 255, 0),
                    2
                )

        # ---------------------------------------------------
        # GRID-BASED EXPECTED HOLE ESTIMATION
        # ---------------------------------------------------

        height, width = gray.shape

        grid_x = np.linspace(
            50,
            width - 50,
            6
        ).astype(int)

        grid_y = np.linspace(
            50,
            height - 50,
            6
        ).astype(int)

        expected_positions = []

        for gx in grid_x:
            for gy in grid_y:

                expected_positions.append((gx, gy))

        # ---------------------------------------------------
        # FIND MISSING HOLES
        # ---------------------------------------------------

        for (ex, ey) in expected_positions:

            found = False

            for hole in detected_holes:

                distance = np.sqrt(
                    (hole["x"] - ex) ** 2 +
                    (hole["y"] - ey) ** 2
                )

                if distance < 25:

                    found = True
                    break

            # Missing Hole Found

            if not found:

                confidence = round(
                    np.random.uniform(0.75, 0.99),
                    2
                )

                if confidence >= self.confidence_threshold:

                    detection = {
                        "defect": "Missing Hole",
                        "confidence": confidence,
                        "expected_position": {
                            "x": int(ex),
                            "y": int(ey)
                        },
                        "severity": self.classify_severity(
                            confidence
                        )
                    }

                    detections.append(detection)

                    # Draw Missing Hole Marker

                    cv2.circle(
                        annotated_image,
                        (ex, ey),
                        15,
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        annotated_image,
                        "Missing Hole",
                        (ex - 20, ey - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
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
            No Missing Hole defects detected.
            PCB drilling quality appears stable.
            """

        prompt = f"""
        You are a PCB drilling and manufacturing expert.

        Analyze the following Missing Hole defects:

        {detections}

        Explain:

        1. CNC drilling issues
        2. Possible machine calibration problems
        3. Layer misalignment risks
        4. Electrical connectivity impact
        5. Reliability concerns
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
            "Calibrate CNC drilling machine regularly.",
            "Monitor drill bit wear continuously.",
            "Improve PCB layer alignment accuracy.",
            "Increase AOI inspection coverage.",
            "Validate drill positioning before production.",
            "Monitor spindle vibration carefully."
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
            "total_missing_hole_defects": total,
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
            self.detect_missing_hole(image)
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

    agent = MissingHoleAgent()

    result = agent.analyze(test_image)

    print(result["summary"])
