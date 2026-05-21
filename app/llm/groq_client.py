import streamlit as st

from langchain_groq import ChatGroq

from langchain.schema import (
    HumanMessage,
    SystemMessage
)


# =========================================================
# GROQ CLIENT
# =========================================================

class GroqLLMClient:

    def __init__(self):

        self.model_name = (
            "llama3-70b-8192"
        )

        self.temperature = 0.2

        self.max_tokens = 2048

        self.llm = self.initialize_llm()

    # -----------------------------------------------------
    # INITIALIZE GROQ MODEL
    # -----------------------------------------------------

    def initialize_llm(self):

        try:

            api_key = st.secrets[
                "GROQ_API_KEY"
            ]

        except Exception:

            api_key = None

        if api_key is None:

            raise ValueError(
                """
                GROQ_API_KEY not found.
                Add it in Streamlit secrets.
                """
            )

        llm = ChatGroq(
            model=self.model_name,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return llm

    # -----------------------------------------------------
    # SIMPLE PROMPT
    # -----------------------------------------------------

    def generate_response(
        self,
        prompt
    ):

        try:

            response = self.llm.invoke(
                prompt
            )

            return response.content

        except Exception as e:

            return (
                f"Groq LLM Error: {str(e)}"
            )

    # -----------------------------------------------------
    # SYSTEM + USER PROMPT
    # -----------------------------------------------------

    def generate_chat_response(
        self,
        system_prompt,
        user_prompt
    ):

        try:

            messages = [

                SystemMessage(
                    content=system_prompt
                ),

                HumanMessage(
                    content=user_prompt
                )
            ]

            response = self.llm.invoke(
                messages
            )

            return response.content

        except Exception as e:

            return (
                f"Groq Chat Error: {str(e)}"
            )

    # -----------------------------------------------------
    # PCB ROOT CAUSE ANALYSIS
    # -----------------------------------------------------

    def analyze_pcb_defect(
        self,
        defect_type,
        detection_details
    ):

        system_prompt = """
        You are an expert PCB manufacturing,
        semiconductor quality control,
        and electronics reliability engineer.

        Your role is to analyze PCB defects,
        identify root causes,
        explain manufacturing impact,
        and provide corrective actions.
        """

        user_prompt = f"""
        PCB Defect Detected:
        {defect_type}

        Detection Details:
        {detection_details}

        Please explain:

        1. Root cause
        2. Manufacturing issues
        3. Electrical risks
        4. Reliability concerns
        5. Prevention methods
        6. Corrective actions
        7. Industrial best practices
        """

        return self.generate_chat_response(
            system_prompt,
            user_prompt
        )

    # -----------------------------------------------------
    # PCB QA ANALYSIS
    # -----------------------------------------------------

    def pcb_quality_analysis(
        self,
        inspection_summary
    ):

        system_prompt = """
        You are an Industrial PCB QA Expert.
        """

        user_prompt = f"""
        PCB Inspection Summary:

        {inspection_summary}

        Provide:

        1. Manufacturing quality analysis
        2. Defect severity assessment
        3. Process optimization suggestions
        4. Future defect prevention
        """

        return self.generate_chat_response(
            system_prompt,
            user_prompt
        )

    # -----------------------------------------------------
    # AI MANUFACTURING INSIGHTS
    # -----------------------------------------------------

    def manufacturing_insights(
        self,
        defect_statistics
    ):

        system_prompt = """
        You are an AI manufacturing consultant
        specializing in PCB fabrication,
        smart factories,
        and semiconductor production.
        """

        user_prompt = f"""
        PCB Manufacturing Statistics:

        {defect_statistics}

        Generate:

        1. Manufacturing insights
        2. Production risks
        3. Trend analysis
        4. Predictive maintenance suggestions
        5. AI optimization recommendations
        """

        return self.generate_chat_response(
            system_prompt,
            user_prompt
        )

    # -----------------------------------------------------
    # GENERAL PCB CHATBOT
    # -----------------------------------------------------

    def pcb_chatbot(
        self,
        question
    ):

        system_prompt = """
        You are an intelligent PCB manufacturing
        and electronics engineering assistant.

        Answer clearly and technically.
        """

        return self.generate_chat_response(
            system_prompt,
            question
        )


# =========================================================
# GLOBAL CLIENT INSTANCE
# =========================================================

groq_client = GroqLLMClient()


# =========================================================
# GLOBAL FUNCTIONS
# =========================================================

def analyze_with_llm(prompt):

    return groq_client.generate_response(
        prompt
    )


def analyze_pcb_defect(
    defect_type,
    detection_details
):

    return groq_client.analyze_pcb_defect(
        defect_type,
        detection_details
    )


def pcb_quality_analysis(
    inspection_summary
):

    return groq_client.pcb_quality_analysis(
        inspection_summary
    )


def manufacturing_insights(
    defect_statistics
):

    return groq_client.manufacturing_insights(
        defect_statistics
    )


def pcb_chatbot(question):

    return groq_client.pcb_chatbot(
        question
    )


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    try:

        result = analyze_with_llm(
            """
            Explain PCB Open Circuit defect.
            """
        )

        print("\nLLM Response:\n")

        print(result)

    except Exception as e:

        print(
            f"Error: {str(e)}"
        )
      
