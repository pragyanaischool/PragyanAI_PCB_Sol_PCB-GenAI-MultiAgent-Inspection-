import streamlit as st
from PIL import Image
import numpy as np
from inference.detector import detect_defects

def upload_page():

    st.header("PragyanAI PCB Upload & Inspection")

    st.write(
        """
        Upload PCB image for AI-powered defect detection.
        Supported defects:
        - Open Circuit
        - Short Circuit
        - Missing Hole
        - Spur
        - Mouse Bite
        - Solder Mask Defects
        """
    )

    uploaded_file = st.file_uploader(
        "Upload PCB Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Uploaded PCB Image")

        st.image(
            image,
            use_container_width=True
        )

        image_np = np.array(image)

        if st.button("Run AI Inspection of PCB "):

            with st.spinner("Running Multi-Agent PCB Analysis..."):

                result = detect_defects(image_np)

            st.success("Inspection Completed")

            st.subheader(" Detection Summary")

            st.json(result["summary"])

            st.subheader(" AI Root Cause Analysis")

            st.write(result["llm_analysis"])

            if "annotated_image" in result:

                st.subheader(" Annotated Detection Output")

                st.image(
                    result["annotated_image"],
                    use_container_width=True
                )

            st.subheader("⚠️ Severity Analysis")

            severity = result.get("severity", {})

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Critical",
                    severity.get("critical", 0)
                )

            with col2:
                st.metric(
                    "Medium",
                    severity.get("medium", 0)
                )

            with col3:
                st.metric(
                    "Low",
                    severity.get("low", 0)
                )

            st.subheader(" Recommended Actions")

            recommendations = result.get(
                "recommendations",
                []
            )

            for rec in recommendations:
                st.info(rec)
