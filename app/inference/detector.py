import cv2
import numpy as np

from inference.preprocessing import (
    preprocess_pipeline
)

from inference.segmentation import (
    run_segmentation_pipeline
)

from agents.open_circuit_agent import (
    OpenCircuitAgent
)

from agents.short_circuit_agent import (
    ShortCircuitAgent
)

from agents.soldermask_agent import (
    SolderMaskAgent
)

from agents.pad_damage_agent import (
    PadDamageAgent
)

from agents.missing_hole_agent import (
    MissingHoleAgent
)


# =========================================================
# PCB DEFECT DETECTOR
# =========================================================

class PCBDefectDetector:

    def __init__(self):

        # -------------------------------------------------
        # INITIALIZE AGENTS
        # -------------------------------------------------

        self.open_circuit_agent = (
            OpenCircuitAgent()
        )

        self.short_circuit_agent = (
            ShortCircuitAgent()
        )

        self.soldermask_agent = (
            SolderMaskAgent()
        )

        self.pad_damage_agent = (
            PadDamageAgent()
        )

        self.missing_hole_agent = (
            MissingHoleAgent()
        )

    # -----------------------------------------------------
    # PREPROCESS IMAGE
    # -----------------------------------------------------

    def preprocess(self, image):

        preprocess_result = (
            preprocess_pipeline(image)
        )

        return preprocess_result

    # -----------------------------------------------------
    # SEGMENT IMAGE
    # -----------------------------------------------------

    def segment(self, image):

        segmentation_result = (
            run_segmentation_pipeline(image)
        )

        return segmentation_result

    # -----------------------------------------------------
    # RUN ALL AGENTS
    # -----------------------------------------------------

    def run_agents(self, image):

        results = {}

        # ---------------------------------------------
        # OPEN CIRCUIT
        # ---------------------------------------------

        try:

            oc_result = (
                self.open_circuit_agent.analyze(
                    image
                )
            )

            results["open_circuit"] = (
                oc_result
            )

        except Exception as e:

            results["open_circuit"] = {
                "error": str(e)
            }

        # ---------------------------------------------
        # SHORT CIRCUIT
        # ---------------------------------------------

        try:

            sc_result = (
                self.short_circuit_agent.analyze(
                    image
                )
            )

            results["short_circuit"] = (
                sc_result
            )

        except Exception as e:

            results["short_circuit"] = {
                "error": str(e)
            }

        # ---------------------------------------------
        # SOLDER MASK
        # ---------------------------------------------

        try:

            sm_result = (
                self.soldermask_agent.analyze(
                    image
                )
            )

            results["solder_mask"] = (
                sm_result
            )

        except Exception as e:

            results["solder_mask"] = {
                "error": str(e)
            }

        # ---------------------------------------------
        # PAD DAMAGE
        # ---------------------------------------------

        try:

            pd_result = (
                self.pad_damage_agent.analyze(
                    image
                )
            )

            results["pad_damage"] = (
                pd_result
            )

        except Exception as e:

            results["pad_damage"] = {
                "error": str(e)
            }

        # ---------------------------------------------
        # MISSING HOLE
        # ---------------------------------------------

        try:

            mh_result = (
                self.missing_hole_agent.analyze(
                    image
                )
            )

            results["missing_hole"] = (
                mh_result
            )

        except Exception as e:

            results["missing_hole"] = {
                "error": str(e)
            }

        return results

    # -----------------------------------------------------
    # GLOBAL SUMMARY
    # -----------------------------------------------------

    def generate_summary(
        self,
        agent_results
    ):

        total_defects = 0

        critical = 0
        high = 0
        medium = 0
        low = 0

        defect_types = []

        recommendations = []

        for agent_name, result in (
            agent_results.items()
        ):

            if "detections" not in result:
                continue

            detections = result["detections"]

            total_defects += len(detections)

            for detection in detections:

                defect_types.append(
                    detection["defect"]
                )

                severity = detection.get(
                    "severity",
                    "Low"
                )

                if severity == "Critical":
                    critical += 1

                elif severity == "High":
                    high += 1

                elif severity == "Medium":
                    medium += 1

                else:
                    low += 1

            recommendations.extend(
                result.get(
                    "recommendations",
                    []
                )
            )

        summary = {
            "total_defects": total_defects,
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "detected_defect_types":
                list(set(defect_types))
        }

        return summary, recommendations

    # -----------------------------------------------------
    # GENERATE HEALTH SCORE
    # -----------------------------------------------------

    def calculate_health_score(
        self,
        summary
    ):

        score = 100

        score -= summary["critical"] * 20
        score -= summary["high"] * 10
        score -= summary["medium"] * 5
        score -= summary["low"] * 2

        score = max(score, 0)

        return score

    # -----------------------------------------------------
    # COMBINE ANNOTATIONS
    # -----------------------------------------------------

    def combine_annotations(
        self,
        image,
        agent_results
    ):

        combined = image.copy()

        for _, result in (
            agent_results.items()
        ):

            if "annotated_image" in result:

                overlay = result[
                    "annotated_image"
                ]

                combined = cv2.addWeighted(
                    combined,
                    0.75,
                    overlay,
                    0.25,
                    0
                )

        return combined

    # -----------------------------------------------------
    # GENERATE AI SUMMARY
    # -----------------------------------------------------

    def generate_ai_summary(
        self,
        summary
    ):

        total = summary["total_defects"]

        if total == 0:

            return """
            PCB inspection completed successfully.
            No major manufacturing defects detected.
            PCB quality appears healthy.
            """

        ai_summary = f"""
        PCB inspection detected
        {total} manufacturing defects.

        Critical Defects:
        {summary['critical']}

        High Severity:
        {summary['high']}

        Medium Severity:
        {summary['medium']}

        Low Severity:
        {summary['low']}

        Defect Types:
        {summary['detected_defect_types']}

        Manufacturing optimization and
        quality control improvements
        are recommended.
        """

        return ai_summary

    # -----------------------------------------------------
    # MAIN DETECTION PIPELINE
    # -----------------------------------------------------

    def detect(self, image):

        # ---------------------------------------------
        # PREPROCESSING
        # ---------------------------------------------

        preprocess_result = (
            self.preprocess(image)
        )

        resized_image = preprocess_result[
            "resized"
        ]

        # ---------------------------------------------
        # RGB → BGR
        # ---------------------------------------------

        bgr_image = cv2.cvtColor(
            resized_image,
            cv2.COLOR_RGB2BGR
        )

        # ---------------------------------------------
        # SEGMENTATION
        # ---------------------------------------------

        segmentation_result = (
            self.segment(bgr_image)
        )

        # ---------------------------------------------
        # RUN AGENTS
        # ---------------------------------------------

        agent_results = self.run_agents(
            resized_image
        )

        # ---------------------------------------------
        # SUMMARY
        # ---------------------------------------------

        summary, recommendations = (
            self.generate_summary(
                agent_results
            )
        )

        # ---------------------------------------------
        # HEALTH SCORE
        # ---------------------------------------------

        health_score = (
            self.calculate_health_score(
                summary
            )
        )

        # ---------------------------------------------
        # AI SUMMARY
        # ---------------------------------------------

        ai_summary = (
            self.generate_ai_summary(
                summary
            )
        )

        # ---------------------------------------------
        # COMBINE ANNOTATIONS
        # ---------------------------------------------

        annotated_image = (
            self.combine_annotations(
                resized_image,
                agent_results
            )
        )

        # ---------------------------------------------
        # FINAL OUTPUT
        # ---------------------------------------------

        final_output = {
            "summary": summary,
            "health_score": health_score,
            "recommendations":
                list(set(recommendations)),
            "agent_results":
                agent_results,
            "segmentation_result":
                segmentation_result,
            "annotated_image":
                annotated_image,
            "llm_analysis":
                ai_summary
        }

        return final_output


# =========================================================
# GLOBAL DETECTOR INSTANCE
# =========================================================

pcb_detector = PCBDefectDetector()


# =========================================================
# GLOBAL FUNCTION
# =========================================================

def detect_defects(image):

    return pcb_detector.detect(image)


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    # Dummy Test Image

    test_image = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )

    result = detect_defects(
        test_image
    )

    print("\nPCB Inspection Completed\n")

    print("Summary:")
    print(result["summary"])

    print("\nHealth Score:")
    print(result["health_score"])

    print("\nRecommendations:")
    print(result["recommendations"])
  
