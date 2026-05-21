import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def analytics_page():

    st.header("PragyanAI -PCB Manufacturing Analytics")

    st.write(
        """
        AI-powered defect trend analytics and manufacturing insights.
        """
    )

    # Sample Data

    data = pd.DataFrame({
        "Month": [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun"
        ],
        "Open Circuit": [5, 7, 8, 12, 10, 14],
        "Short Circuit": [3, 5, 4, 7, 6, 9],
        "Solder Mask": [2, 3, 5, 4, 6, 8]
    })

    st.subheader(" Defect Trends")

    fig, ax = plt.subplots()

    ax.plot(
        data["Month"],
        data["Open Circuit"],
        marker="o",
        label="Open Circuit"
    )

    ax.plot(
        data["Month"],
        data["Short Circuit"],
        marker="o",
        label="Short Circuit"
    )

    ax.plot(
        data["Month"],
        data["Solder Mask"],
        marker="o",
        label="Solder Mask"
    )

    ax.legend()

    ax.set_xlabel("Month")
    ax.set_ylabel("Defect Count")

    st.pyplot(fig)

    st.divider()

    st.subheader(" Root Cause Analysis")

    root_causes = pd.DataFrame({
        "Defect": [
            "Open Circuit",
            "Short Circuit",
            "Solder Mask"
        ],
        "Root Cause": [
            "Over-Etching",
            "Copper Bridging",
            "Alignment Error"
        ]
    })

    st.dataframe(
        root_causes,
        use_container_width=True
    )

    st.divider()

    st.subheader(" Manufacturing Recommendations")

    recommendations = [
        "Optimize etching process.",
        "Improve solder mask alignment calibration.",
        "Increase AOI inspection frequency.",
        "Monitor copper deposition consistency.",
        "Improve drill machine calibration."
    ]

    for rec in recommendations:
        st.success(rec)

    st.divider()

    st.subheader(" AI Predictive Insights")

    prediction = """
    AI predicts a 15% increase in solder mask defects
    in upcoming production batches if calibration
    drift is not corrected.
    """

    st.info(prediction)
