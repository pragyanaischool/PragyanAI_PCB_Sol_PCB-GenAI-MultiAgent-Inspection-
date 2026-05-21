import streamlit as st
from llm.groq_client import analyze_with_llm


def chatbot_page():

    st.header("PragyanAI - PCB AI Assistant")

    st.write(
        """
        Ask PCB manufacturing, defect analysis,
        or process optimization questions.
        """
    )

    # Session Chat History

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # User Input

    prompt = st.chat_input(
        "Ask your PCB manufacturing question..."
    )

    if prompt:

        # Store User Message

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Display User Message

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Response

        with st.chat_message("assistant"):

            with st.spinner("Analyzing..."):

                response = analyze_with_llm(prompt)

                st.markdown(response)

        # Save AI Response

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    st.divider()

    st.subheader("💡 Example Questions")

    examples = [
        "Why does open circuit occur in PCB?",
        "How to reduce solder mask defects?",
        "What causes under-etching?",
        "How does AOI detect PCB defects?",
        "Explain short circuit detection."
    ]

    for example in examples:
        st.info(example)
