import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PCB VISUALIZATION UTILITIES
# =========================================================

class PCBVisualizer:

    def __init__(self):

        self.font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

    # -----------------------------------------------------
    # DRAW BOUNDING BOX
    # -----------------------------------------------------

    def draw_bounding_box(
        self,
        image,
        bbox,
        label="Defect",
        color=(0, 0, 255)
    ):

        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            image,
            label,
            (x, y - 10),
            self.font,
            0.5,
            color,
            2
        )

        return image

    # -----------------------------------------------------
    # DRAW MULTIPLE DETECTIONS
    # -----------------------------------------------------

    def draw_detections(
        self,
        image,
        detections
    ):

        annotated = image.copy()

        for detection in detections:

            bbox = detection[
                "bounding_box"
            ]

            label = (
                f"{detection['defect']} "
                f"{detection['confidence']}"
            )

            severity = detection.get(
                "severity",
                "Low"
            )

            color = self.get_color(
                severity
            )

            annotated = (
                self.draw_bounding_box(
                    annotated,
                    bbox,
                    label,
                    color
                )
            )

        return annotated

    # -----------------------------------------------------
    # SEVERITY COLORS
    # -----------------------------------------------------

    def get_color(self, severity):

        if severity == "Critical":
            return (0, 0, 255)

        elif severity == "High":
            return (0, 165, 255)

        elif severity == "Medium":
            return (0, 255, 255)

        else:
            return (0, 255, 0)

    # -----------------------------------------------------
    # GENERATE HEATMAP
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
    # DISPLAY IMAGE
    # -----------------------------------------------------

    def display_image(
        self,
        image,
        title="PCB Image"
    ):

        plt.figure(figsize=(10, 8))

        plt.imshow(image)

        plt.title(title)

        plt.axis("off")

        plt.show()

    # -----------------------------------------------------
    # DISPLAY MULTIPLE IMAGES
    # -----------------------------------------------------

    def display_multiple_images(
        self,
        images,
        titles
    ):

        total = len(images)

        plt.figure(
            figsize=(5 * total, 5)
        )

        for i in range(total):

            plt.subplot(1, total, i + 1)

            plt.imshow(images[i])

            plt.title(titles[i])

            plt.axis("off")

        plt.show()

    # -----------------------------------------------------
    # PLOT DEFECT DISTRIBUTION
    # -----------------------------------------------------

    def plot_defect_distribution(
        self,
        defect_counts
    ):

        labels = list(
            defect_counts.keys()
        )

        values = list(
            defect_counts.values()
        )

        plt.figure(figsize=(8, 6))

        plt.bar(labels, values)

        plt.xlabel("Defect Type")

        plt.ylabel("Count")

        plt.title(
            "PCB Defect Distribution"
        )

        plt.xticks(rotation=30)

        plt.show()

    # -----------------------------------------------------
    # PLOT SEVERITY DISTRIBUTION
    # -----------------------------------------------------

    def plot_severity_distribution(
        self,
        severity_counts
    ):

        labels = list(
            severity_counts.keys()
        )

        values = list(
            severity_counts.values()
        )

        plt.figure(figsize=(6, 6))

        plt.pie(
            values,
            labels=labels,
            autopct="%1.1f%%"
        )

        plt.title(
            "Defect Severity Distribution"
        )

        plt.show()

    # -----------------------------------------------------
    # CREATE DASHBOARD IMAGE
    # -----------------------------------------------------

    def create_dashboard_overlay(
        self,
        image,
        summary
    ):

        overlay = image.copy()

        text_lines = [

            f"Total Defects: "
            f"{summary['total_defects']}",

            f"Critical: "
            f"{summary['critical']}",

            f"High: "
            f"{summary['high']}",

            f"Medium: "
            f"{summary['medium']}",

            f"Low: "
            f"{summary['low']}",

            f"Health Score: "
            f"{summary['health_score']}%"
        ]

        y = 30

        for line in text_lines:

            cv2.putText(
                overlay,
                line,
                (20, y),
                self.font,
                0.7,
                (255, 255, 255),
                2
            )

            y += 35

        return overlay

    # -----------------------------------------------------
    # SAVE VISUALIZATION
    # -----------------------------------------------------

    def save_visualization(
        self,
        image,
        path
    ):

        cv2.imwrite(path, image)

    # -----------------------------------------------------
    # CREATE COMPARISON VIEW
    # -----------------------------------------------------

    def comparison_view(
        self,
        original,
        annotated
    ):

        comparison = np.hstack(
            (original, annotated)
        )

        return comparison


# =========================================================
# GLOBAL INSTANCE
# =========================================================

visualizer = PCBVisualizer()


# =========================================================
# GLOBAL FUNCTIONS
# =========================================================

def draw_detections(
    image,
    detections
):

    return visualizer.draw_detections(
        image,
        detections
    )


def generate_heatmap(
    image,
    defect_mask
):

    return visualizer.generate_heatmap(
        image,
        defect_mask
    )


def plot_defect_distribution(
    defect_counts
):

    visualizer.plot_defect_distribution(
        defect_counts
    )


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    dummy_image = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )

    detections = [
        {
            "defect": "Open Circuit",
            "confidence": 0.95,
            "severity": "Critical",
            "bounding_box": {
                "x": 100,
                "y": 100,
                "width": 120,
                "height": 80
            }
        }
    ]

    annotated = draw_detections(
        dummy_image,
        detections
    )

    print(
        "Visualization Completed"
    )
  
