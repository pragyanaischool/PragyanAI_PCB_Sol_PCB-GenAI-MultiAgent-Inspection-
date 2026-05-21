from typing import Dict, List


# =========================================================
# PCB MULTI-AGENT ROUTER
# =========================================================

class PCBRouter:

    """
    Intelligent routing engine for
    LangGraph Multi-Agent PCB System
    """

    def __init__(self):

        self.available_agents = {
            "open_circuit": True,
            "short_circuit": True,
            "solder_mask": True,
            "pad_damage": True,
            "missing_hole": True
        }

    # -----------------------------------------------------
    # ROUTE BASED ON IMAGE CHARACTERISTICS
    # -----------------------------------------------------

    def route_agents(
        self,
        preprocessing_result,
        segmentation_result
    ):

        selected_agents = []

        # -------------------------------------------------
        # ANALYZE SEGMENTATION RESULTS
        # -------------------------------------------------

        copper_mask = segmentation_result.get(
            "copper_mask"
        )

        solder_mask = segmentation_result.get(
            "solder_mask"
        )

        detected_holes = segmentation_result.get(
            "detected_holes",
            []
        )

        defect_mask = segmentation_result.get(
            "defect_mask"
        )

        # -------------------------------------------------
        # OPEN CIRCUIT ROUTING
        # -------------------------------------------------

        if self.should_run_open_circuit(
            copper_mask,
            defect_mask
        ):

            selected_agents.append(
                "open_circuit"
            )

        # -------------------------------------------------
        # SHORT CIRCUIT ROUTING
        # -------------------------------------------------

        if self.should_run_short_circuit(
            copper_mask
        ):

            selected_agents.append(
                "short_circuit"
            )

        # -------------------------------------------------
        # SOLDER MASK ROUTING
        # -------------------------------------------------

        if self.should_run_soldermask(
            solder_mask
        ):

            selected_agents.append(
                "solder_mask"
            )

        # -------------------------------------------------
        # PAD DAMAGE ROUTING
        # -------------------------------------------------

        if self.should_run_pad_damage(
            defect_mask
        ):

            selected_agents.append(
                "pad_damage"
            )

        # -------------------------------------------------
        # MISSING HOLE ROUTING
        # -------------------------------------------------

        if self.should_run_missing_hole(
            detected_holes
        ):

            selected_agents.append(
                "missing_hole"
            )

        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        if len(selected_agents) == 0:

            selected_agents = [
                "open_circuit",
                "short_circuit"
            ]

        return selected_agents

    # -----------------------------------------------------
    # OPEN CIRCUIT LOGIC
    # -----------------------------------------------------

    def should_run_open_circuit(
        self,
        copper_mask,
        defect_mask
    ):

        if copper_mask is None:
            return False

        non_zero = copper_mask.sum()

        if non_zero > 10000:
            return True

        return False

    # -----------------------------------------------------
    # SHORT CIRCUIT LOGIC
    # -----------------------------------------------------

    def should_run_short_circuit(
        self,
        copper_mask
    ):

        if copper_mask is None:
            return False

        density = copper_mask.mean()

        if density > 20:
            return True

        return False

    # -----------------------------------------------------
    # SOLDER MASK LOGIC
    # -----------------------------------------------------

    def should_run_soldermask(
        self,
        solder_mask
    ):

        if solder_mask is None:
            return False

        mask_pixels = solder_mask.sum()

        if mask_pixels > 5000:
            return True

        return False

    # -----------------------------------------------------
    # PAD DAMAGE LOGIC
    # -----------------------------------------------------

    def should_run_pad_damage(
        self,
        defect_mask
    ):

        if defect_mask is None:
            return False

        edges = defect_mask.sum()

        if edges > 8000:
            return True

        return False

    # -----------------------------------------------------
    # MISSING HOLE LOGIC
    # -----------------------------------------------------

    def should_run_missing_hole(
        self,
        detected_holes
    ):

        if len(detected_holes) < 5:
            return True

        return False

    # -----------------------------------------------------
    # PRIORITY ORDERING
    # -----------------------------------------------------

    def prioritize_agents(
        self,
        selected_agents
    ):

        priority_order = [
            "open_circuit",
            "short_circuit",
            "missing_hole",
            "pad_damage",
            "solder_mask"
        ]

        prioritized = []

        for agent in priority_order:

            if agent in selected_agents:

                prioritized.append(agent)

        return prioritized

    # -----------------------------------------------------
    # FAILURE HANDLER
    # -----------------------------------------------------

    def handle_agent_failure(
        self,
        agent_name,
        error
    ):

        print(
            f"[Router] Agent Failure: "
            f"{agent_name} -> {error}"
        )

        return {
            "agent": agent_name,
            "status": "failed",
            "error": str(error)
        }

    # -----------------------------------------------------
    # AGENT STATUS
    # -----------------------------------------------------

    def get_agent_status(self):

        return self.available_agents


# =========================================================
# GLOBAL ROUTER INSTANCE
# =========================================================

router = PCBRouter()
