import streamlit as st

# UI Pages
from ui.upload import upload_page
from ui.dashboard import dashboard_page
from ui.analytics import analytics_page
from ui.chatbot import chatbot_page


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="PCB GenAI Multi-Agent Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    .title-text {
        font-size: 42px;
        font-weight: bold;
        color: #00BFFF;
    }

    .sub-text {
        font-size: 18px;
        color: #CCCCCC;
    }

    .feature-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    """
    <div class="title-text">
        🔍 PCB Defect Detection + GenAI Multi-Agent System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-text">
        AI-powered PCB Manufacturing Inspection using
        Deep Learning, LangGraph, LangChain, and GROQ LLM.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title(" PragyanAI - PCB - Navigation")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Upload PCB",
        "Dashboard",
        "Analytics",
        "AI Assistant"
    ]
)


# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if menu == "Home":

    st.subheader("PragyanAI - Industrial AI PCB Inspection Platform")

    st.write(
        """
        This platform provides intelligent PCB defect detection
        using Deep Learning + Generative AI + Multi-Agent Systems.
        """
    )

    st.divider()

    # FEATURES

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
                <h3> PragyanAI - AI Defect Detection</h3>
                <p>
                Detect Open Circuit, Short Circuit,
                Missing Hole, Spur, Pin Hole,
                Solder Mask Defects and more.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3> PragyanAI - PCB - Multi-Agent AI</h3>
                <p>
                Dedicated intelligent agents analyze
                each PCB issue independently.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>PCB - Manufacturing Analytics</h3>
                <p>
                Visualize defect trends,
                production quality,
                and process insights.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
                <h3>⚡ GROQ LLM Reasoning</h3>
                <p>
                AI explains root causes,
                severity,
                and preventive actions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>🔗 LangGraph Workflow</h3>
                <p>
                Intelligent orchestration
                of multiple AI agents
                using LangGraph.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>PCB - Industrial QA Reports</h3>
                <p>
                Automated AI-powered
                PCB inspection reports
                and recommendations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # SYSTEM WORKFLOW

    st.subheader("PCB - System Workflow")

    workflow = """
    PCB Upload
        ↓
    Image Preprocessing
        ↓
    Deep Learning Detection
        ↓
    Multi-Agent AI Analysis
        ↓
    GROQ LLM Root Cause Analysis
        ↓
    Severity Classification
        ↓
    Dashboard + Reports
    """

    st.code(workflow)

    st.divider()

    # SUPPORTED DEFECTS

    st.subheader(" Supported PCB Defects")

    defects = [
        "Open Circuit",
        "Short Circuit",
        "Missing Hole",
        "Spur",
        "Mouse Bite",
        "Pin Hole",
        "Under-Etching",
        "Over-Etching",
        "Solder Mask Defects",
        "Pad Damage"
    ]

    cols = st.columns(2)

    for i, defect in enumerate(defects):

        with cols[i % 2]:
            st.success(defect)

    st.divider()

    # TECHNOLOGY STACK

    st.subheader(" Technology Stack")

    tech_stack = {
        "Frontend": "Streamlit",
        "Computer Vision": "OpenCV",
        "Deep Learning": "PyTorch + YOLOv8",
        "LLM": "GROQ Llama 3",
        "Agents": "LangChain",
        "Workflow": "LangGraph",
        "Visualization": "Matplotlib",
        "Deployment": "Streamlit Cloud"
    }

    st.json(tech_stack)

    st.divider()

    # FOOTER

    st.info(
        """
        Developed for Industrial AI, Semiconductor AI,
        PCB Manufacturing Inspection, and Smart Factory Applications.
        """
    )


# ---------------------------------------------------
# PAGE ROUTING
# ---------------------------------------------------

elif menu == "Upload PCB":

    upload_page()


elif menu == "Dashboard":

    dashboard_page()


elif menu == "Analytics":

    analytics_page()


elif menu == "AI Assistant":

    chatbot_page()
