from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, END

from workflows.router import router

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

from inference.preprocessing import (
    preprocess_pipeline
)

from inference.segmentation import (
    run_segmentation_pipeline
)


# =========================================================
# GRAPH STATE
# =========================================================

class PCBGraphState(TypedDict):

    image: Any

    preprocessing_result: Dict

    segmentation_result: Dict

    selected_agents: list

    agent_outputs: Dict

    final_summary: Dict

    recommendations: list


# =========================================================
# INITIALIZE AGENTS
# =========================================================

open_circuit_agent = OpenCircuitAgent()

short_circuit_agent = ShortCircuitAgent()

soldermask_agent = SolderMaskAgent()

pad_damage_agent = PadDamageAgent()

missing_hole_agent = MissingHoleAgent()


# =========================================================
# PREPROCESSING NODE
# =========================================================

def preprocessing_node(state):

    image = state["image"]

    preprocess_result = (
        preprocess_pipeline(image)
    )

    state["preprocessing_result"] = (
        preprocess_result
    )

    return state


# =========================================================
# SEGMENTATION NODE
# =========================================================

def segmentation_node(state):

    preprocess_result = (
        state["preprocessing_result"]
    )

    resized_image = preprocess_result[
        "resized"
    ]

    segmentation_result = (
        run_segmentation_pipeline(
            resized_image
        )
    )

    state["segmentation_result"] = (
        segmentation_result
    )

    return state


# =========================================================
# ROUTER NODE
# =========================================================

def router_node(state):

    preprocessing_result = (
        state["preprocessing_result"]
    )

    segmentation_result = (
        state["segmentation_result"]
    )

    selected_agents = router.route_agents(
        preprocessing_result,
        segmentation_result
    )

    prioritized_agents = (
        router.prioritize_agents(
            selected_agents
        )
    )

    state["selected_agents"] = (
        prioritized_agents
    )

    return state


# =========================================================
# OPEN CIRCUIT NODE
# =========================================================

def open_circuit_node(state):

    image = state["image"]

    outputs = state.get(
        "agent_outputs",
        {}
    )

    try:

        result = (
            open_circuit_agent.analyze(
                image
            )
        )

        outputs["open_circuit"] = result

    except Exception as e:

        outputs["open_circuit"] = (
            router.handle_agent_failure(
                "open_circuit",
                e
            )
        )

    state["agent_outputs"] = outputs

    return state


# =========================================================
# SHORT CIRCUIT NODE
# =========================================================

def short_circuit_node(state):

    image = state["image"]

    outputs = state.get(
        "agent_outputs",
        {}
    )

    try:

        result = (
            short_circuit_agent.analyze(
                image
            )
        )

        outputs["short_circuit"] = result

    except Exception as e:

        outputs["short_circuit"] = (
            router.handle_agent_failure(
                "short_circuit",
                e
            )
        )

    state["agent_outputs"] = outputs

    return state


# =========================================================
# SOLDER MASK NODE
# =========================================================

def soldermask_node(state):

    image = state["image"]

    outputs = state.get(
        "agent_outputs",
        {}
    )

    try:

        result = (
            soldermask_agent.analyze(
                image
            )
        )

        outputs["solder_mask"] = result

    except Exception as e:

        outputs["solder_mask"] = (
            router.handle_agent_failure(
                "solder_mask",
                e
            )
        )

    state["agent_outputs"] = outputs

    return state


# =========================================================
# PAD DAMAGE NODE
# =========================================================

def pad_damage_node(state):

    image = state["image"]

    outputs = state.get(
        "agent_outputs",
        {}
    )

    try:

        result = (
            pad_damage_agent.analyze(
                image
            )
        )

        outputs["pad_damage"] = result

    except Exception as e:

        outputs["pad_damage"] = (
            router.handle_agent_failure(
                "pad_damage",
                e
            )
        )

    state["agent_outputs"] = outputs

    return state


# =========================================================
# MISSING HOLE NODE
# =========================================================

def missing_hole_node(state):

    image = state["image"]

    outputs = state.get(
        "agent_outputs",
        {}
    )

    try:

        result = (
            missing_hole_agent.analyze(
                image
            )
        )

        outputs["missing_hole"] = result

    except Exception as e:

        outputs["missing_hole"] = (
            router.handle_agent_failure(
                "missing_hole",
                e
            )
        )

    state["agent_outputs"] = outputs

    return state


# =========================================================
# FINAL SUMMARY NODE
# =========================================================

def summary_node(state):

    outputs = state["agent_outputs"]

    total_defects = 0

    recommendations = []

    defect_types = []

    for _, result in outputs.items():

        detections = result.get(
            "detections",
            []
        )

        total_defects += len(
            detections
        )

        recommendations.extend(
            result.get(
                "recommendations",
                []
            )
        )

        for d in detections:

            defect_types.append(
                d["defect"]
            )

    summary = {
        "total_defects": total_defects,
        "detected_defects":
            list(set(defect_types)),
        "agents_executed":
            list(outputs.keys())
    }

    state["final_summary"] = summary

    state["recommendations"] = (
        list(set(recommendations))
    )

    return state


# =========================================================
# BUILD LANGGRAPH
# =========================================================

workflow = StateGraph(
    PCBGraphState
)

# ---------------------------------------------------------
# ADD NODES
# ---------------------------------------------------------

workflow.add_node(
    "preprocessing",
    preprocessing_node
)

workflow.add_node(
    "segmentation",
    segmentation_node
)

workflow.add_node(
    "router",
    router_node
)

workflow.add_node(
    "open_circuit",
    open_circuit_node
)

workflow.add_node(
    "short_circuit",
    short_circuit_node
)

workflow.add_node(
    "solder_mask",
    soldermask_node
)

workflow.add_node(
    "pad_damage",
    pad_damage_node
)

workflow.add_node(
    "missing_hole",
    missing_hole_node
)

workflow.add_node(
    "summary",
    summary_node
)

# ---------------------------------------------------------
# GRAPH FLOW
# ---------------------------------------------------------

workflow.set_entry_point(
    "preprocessing"
)

workflow.add_edge(
    "preprocessing",
    "segmentation"
)

workflow.add_edge(
    "segmentation",
    "router"
)

# Parallel Agent Execution

workflow.add_edge(
    "router",
    "open_circuit"
)

workflow.add_edge(
    "router",
    "short_circuit"
)

workflow.add_edge(
    "router",
    "solder_mask"
)

workflow.add_edge(
    "router",
    "pad_damage"
)

workflow.add_edge(
    "router",
    "missing_hole"
)

# Merge to Summary

workflow.add_edge(
    "open_circuit",
    "summary"
)

workflow.add_edge(
    "short_circuit",
    "summary"
)

workflow.add_edge(
    "solder_mask",
    "summary"
)

workflow.add_edge(
    "pad_damage",
    "summary"
)

workflow.add_edge(
    "missing_hole",
    "summary"
)

workflow.add_edge(
    "summary",
    END
)

# ---------------------------------------------------------
# COMPILE GRAPH
# ---------------------------------------------------------

pcb_graph = workflow.compile()


# =========================================================
# RUN WORKFLOW
# =========================================================

def run_workflow(image):

    initial_state = {
        "image": image,
        "preprocessing_result": {},
        "segmentation_result": {},
        "selected_agents": [],
        "agent_outputs": {},
        "final_summary": {},
        "recommendations": []
    }

    result = pcb_graph.invoke(
        initial_state
    )

    return result


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    import numpy as np

    dummy_image = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )

    result = run_workflow(
        dummy_image
    )

    print("\nLangGraph Workflow Completed\n")

    print(result["final_summary"])
  
