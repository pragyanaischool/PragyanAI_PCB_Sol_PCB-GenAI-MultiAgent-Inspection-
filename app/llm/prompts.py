# =========================================================
# PCB GENAI PROMPTS
# =========================================================

# ---------------------------------------------------------
# GLOBAL SYSTEM PROMPT
# ---------------------------------------------------------

GLOBAL_SYSTEM_PROMPT = """
You are an expert PCB manufacturing,
electronics reliability,
AI inspection,
and semiconductor quality control engineer.

You specialize in:

- PCB defect analysis
- Manufacturing optimization
- Semiconductor QA
- Root cause analysis
- Industrial AI inspection
- Smart factory automation
- Reliability engineering

Your responses should be:
- Technical
- Accurate
- Structured
- Industrial-grade
- Easy to understand
"""

# =========================================================
# OPEN CIRCUIT PROMPT
# =========================================================

OPEN_CIRCUIT_PROMPT = """
Analyze the following PCB Open Circuit defect.

Explain:

1. Root cause
2. Manufacturing process issue
3. Electrical impact
4. Reliability concern
5. Possible failure modes
6. Prevention techniques
7. Corrective actions
8. Industrial best practices

Detection Details:
{detection_details}
"""

# =========================================================
# SHORT CIRCUIT PROMPT
# =========================================================

SHORT_CIRCUIT_PROMPT = """
Analyze the following PCB Short Circuit defect.

Explain:

1. Root cause
2. Copper bridging causes
3. Electrical risk
4. Fire or overheating risk
5. Manufacturing issue
6. Prevention methods
7. Corrective actions
8. Quality inspection recommendations

Detection Details:
{detection_details}
"""

# =========================================================
# SOLDER MASK PROMPT
# =========================================================

SOLDER_MASK_PROMPT = """
Analyze the following PCB Solder Mask defect.

Explain:

1. Root cause
2. Coating process issue
3. Misalignment causes
4. Reliability impact
5. Soldering impact
6. Manufacturing risks
7. Prevention methods
8. Corrective actions

Detection Details:
{detection_details}
"""

# =========================================================
# PAD DAMAGE PROMPT
# =========================================================

PAD_DAMAGE_PROMPT = """
Analyze the following PCB Pad Damage defect.

Explain:

1. Root cause
2. Mechanical stress issue
3. Soldering problems
4. Electrical connectivity risks
5. Reliability impact
6. Prevention techniques
7. Corrective actions
8. Industrial quality improvements

Detection Details:
{detection_details}
"""

# =========================================================
# MISSING HOLE PROMPT
# =========================================================

MISSING_HOLE_PROMPT = """
Analyze the following PCB Missing Hole defect.

Explain:

1. CNC drilling issue
2. Machine calibration problems
3. Layer misalignment risks
4. Electrical connectivity impact
5. Reliability concerns
6. Prevention techniques
7. Corrective actions
8. Industrial QA recommendations

Detection Details:
{detection_details}
"""

# =========================================================
# UNDER ETCHING PROMPT
# =========================================================

UNDER_ETCHING_PROMPT = """
Analyze the following PCB Under-Etching defect.

Explain:

1. Etching process issue
2. Excess copper causes
3. Electrical risks
4. Manufacturing impact
5. Prevention methods
6. Corrective actions
7. Process optimization techniques

Detection Details:
{detection_details}
"""

# =========================================================
# OVER ETCHING PROMPT
# =========================================================

OVER_ETCHING_PROMPT = """
Analyze the following PCB Over-Etching defect.

Explain:

1. Excess copper removal causes
2. Trace weakening impact
3. Reliability concerns
4. Electrical performance risks
5. Prevention methods
6. Corrective actions
7. Industrial optimization methods

Detection Details:
{detection_details}
"""

# =========================================================
# SPUR PROMPT
# =========================================================

SPUR_PROMPT = """
Analyze the following PCB Spur defect.

Explain:

1. Copper protrusion causes
2. Short circuit risks
3. Manufacturing issue
4. Reliability impact
5. Prevention methods
6. Corrective actions
7. Industrial quality improvements

Detection Details:
{detection_details}
"""

# =========================================================
# PIN HOLE PROMPT
# =========================================================

PIN_HOLE_PROMPT = """
Analyze the following PCB Pin Hole defect.

Explain:

1. Copper void causes
2. Conductivity issues
3. Reliability concerns
4. Manufacturing risks
5. Prevention methods
6. Corrective actions
7. Inspection improvements

Detection Details:
{detection_details}
"""

# =========================================================
# MOUSE BITE PROMPT
# =========================================================

MOUSE_BITE_PROMPT = """
Analyze the following PCB Mouse Bite defect.

Explain:

1. Over-etching causes
2. Edge damage impact
3. Mechanical reliability issues
4. Manufacturing concerns
5. Prevention methods
6. Corrective actions
7. Process optimization

Detection Details:
{detection_details}
"""

# =========================================================
# PCB QUALITY ANALYSIS PROMPT
# =========================================================

PCB_QUALITY_ANALYSIS_PROMPT = """
Analyze the following PCB inspection report.

Provide:

1. Overall manufacturing quality
2. Severity assessment
3. Most critical risks
4. Root cause trends
5. Process optimization suggestions
6. Industrial QA improvements
7. Reliability recommendations
8. Smart factory recommendations

Inspection Summary:
{inspection_summary}
"""

# =========================================================
# MANUFACTURING INSIGHTS PROMPT
# =========================================================

MANUFACTURING_INSIGHTS_PROMPT = """
Analyze the following PCB manufacturing statistics.

Provide:

1. Manufacturing insights
2. Defect trend analysis
3. Production bottlenecks
4. Predictive maintenance insights
5. AI optimization opportunities
6. Smart factory recommendations
7. Yield improvement techniques
8. Industrial automation suggestions

Statistics:
{defect_statistics}
"""

# =========================================================
# AI CHATBOT PROMPT
# =========================================================

PCB_CHATBOT_PROMPT = """
You are a PCB manufacturing
and semiconductor AI assistant.

Answer the following question clearly,
technically,
and professionally.

Question:
{question}
"""

# =========================================================
# RAG DOCUMENT PROMPT
# =========================================================

RAG_PROMPT = """
Use the following PCB manufacturing documents
and IPC standards to answer the question.

Documents:
{context}

Question:
{question}

Provide:
- Technical explanation
- Standards reference
- Best practices
- Manufacturing recommendations
"""

# =========================================================
# FACTORY QA COPILOT PROMPT
# =========================================================

FACTORY_QA_PROMPT = """
You are an Industrial Smart Factory QA Copilot.

Analyze the PCB production issue below.

Provide:

1. Root cause
2. Process failure analysis
3. Yield impact
4. Cost impact
5. Production optimization
6. Preventive maintenance
7. AI automation opportunities

Issue:
{issue}
"""

# =========================================================
# PREDICTIVE MAINTENANCE PROMPT
# =========================================================

PREDICTIVE_MAINTENANCE_PROMPT = """
Analyze PCB production defects
and predict future manufacturing risks.

Provide:

1. Future defect trends
2. Machine failure possibilities
3. Tool wear analysis
4. Maintenance recommendations
5. AI monitoring suggestions

Production Data:
{production_data}
"""

# =========================================================
# DEFECT SEVERITY PROMPT
# =========================================================

DEFECT_SEVERITY_PROMPT = """
Analyze PCB defect severity.

Defect Details:
{defect_details}

Classify:
- Critical
- High
- Medium
- Low

Explain:
1. Risk level
2. Reliability impact
3. Production impact
4. Recommended priority
"""

# =========================================================
# SMART FACTORY PROMPT
# =========================================================

SMART_FACTORY_PROMPT = """
Suggest Smart Factory AI improvements
for PCB manufacturing.

Focus on:

1. AI inspection
2. Predictive maintenance
3. Robotics integration
4. Process automation
5. Yield optimization
6. Real-time monitoring
7. Digital twin integration
8. Multi-agent AI systems

Factory Details:
{factory_details}
"""

# =========================================================
# REPORT GENERATION PROMPT
# =========================================================

REPORT_GENERATION_PROMPT = """
Generate a professional PCB inspection report.

Include:

1. Executive summary
2. Defect analysis
3. Severity assessment
4. Root cause analysis
5. Manufacturing impact
6. Recommendations
7. Corrective actions
8. Future prevention methods

Inspection Data:
{inspection_data}
"""

# =========================================================
# DASHBOARD INSIGHTS PROMPT
# =========================================================

DASHBOARD_INSIGHTS_PROMPT = """
Generate industrial AI dashboard insights
for PCB manufacturing analytics.

Analytics Data:
{analytics_data}

Provide:
1. Key insights
2. Defect trends
3. Yield risks
4. Production recommendations
5. AI optimization opportunities
"""
