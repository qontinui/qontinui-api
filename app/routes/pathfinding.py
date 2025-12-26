"""Pathfinding validation endpoints.

This router provides stateless pathfinding validation:
- Validate if multiple target states can be reached simultaneously
- Uses multistate library for graph-based pathfinding

NO backward compatibility - clean FastAPI code.
"""

import logging
import os
import sys
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Add multistate to path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../../../multistate/src"),
)

from multistate.manager import StateManager, StateManagerConfig  # noqa: E402
from multistate.pathfinding.multi_target import SearchStrategy  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class TransitionDefinition(BaseModel):
    """Transition definition from frontend."""

    id: str
    from_state: str = Field(alias="fromState")
    activate_states: list[str] = Field(default_factory=list, alias="activateStates")
    deactivate_states: list[str] = Field(default_factory=list, alias="deactivateStates")
    stays_visible: bool = Field(default=True, alias="staysVisible")

    class Config:
        populate_by_name = True


class StateDefinition(BaseModel):
    """State definition from frontend."""

    id: str
    name: str
    is_initial: bool = Field(default=False, alias="isInitial")
    is_blocking: bool = Field(default=False, alias="isBlocking")

    class Config:
        populate_by_name = True


class PathValidationRequest(BaseModel):
    """Request to validate if target states can be reached."""

    states: list[StateDefinition]
    transitions: list[TransitionDefinition]
    from_states: list[str] = Field(
        default_factory=list,
        alias="fromStates",
        description="Current/starting state IDs. If empty, uses initial states.",
    )
    target_states: list[str] = Field(
        alias="targetStates", description="Target state IDs that must ALL be reached"
    )

    class Config:
        populate_by_name = True


class PathStep(BaseModel):
    """A step in the path."""

    transition_id: str
    from_states: list[str]
    to_states: list[str]


class PathValidationResponse(BaseModel):
    """Response from path validation."""

    reachable: bool
    path: list[PathStep] | None = None
    reason: str | None = None
    details: dict[str, Any] | None = None


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/pathfinding", tags=["pathfinding"])


@router.post("/validate", response_model=PathValidationResponse)
async def validate_path(request: PathValidationRequest) -> PathValidationResponse:
    """Validate if all target states can be reached simultaneously.

    This endpoint uses the multistate library to check if there exists a path
    from the starting states to a configuration where ALL target states are active.

    Args:
        request: Validation request with states, transitions, and targets

    Returns:
        PathValidationResponse with reachability status and path if found
    """
    try:
        # Validate input
        if not request.states:
            return PathValidationResponse(reachable=False, reason="No states provided")

        if not request.target_states:
            return PathValidationResponse(reachable=False, reason="No target states provided")

        if len(request.target_states) == 1:
            # Single target - always potentially reachable if state exists
            target_id = request.target_states[0]
            state_ids = {s.id for s in request.states}
            if target_id not in state_ids:
                return PathValidationResponse(
                    reachable=False,
                    reason=f"Target state '{target_id}' not found in state definitions",
                )
            return PathValidationResponse(
                reachable=True,
                reason="Single target state - reachability depends on transitions",
            )

        # Build multistate graph
        config = StateManagerConfig(
            default_search_strategy=SearchStrategy.BFS,
            max_path_depth=50,
        )
        manager = StateManager(config)

        # Register states
        state_id_set = set()
        for state_def in request.states:
            manager.add_state(
                id=state_def.id,
                name=state_def.name,
                blocking=state_def.is_blocking,
            )
            state_id_set.add(state_def.id)

        # Validate target states exist
        missing_targets = set(request.target_states) - state_id_set
        if missing_targets:
            return PathValidationResponse(
                reachable=False,
                reason=f"Target states not found: {', '.join(missing_targets)}",
            )

        # Register transitions
        for trans_def in request.transitions:
            # Skip transitions with invalid states
            if trans_def.from_state and trans_def.from_state not in state_id_set:
                logger.warning(
                    f"Skipping transition {trans_def.id}: from_state '{trans_def.from_state}' not found"
                )
                continue

            invalid_activate = set(trans_def.activate_states) - state_id_set
            if invalid_activate:
                logger.warning(
                    f"Skipping transition {trans_def.id}: activate_states {invalid_activate} not found"
                )
                continue

            manager.add_transition(
                id=trans_def.id,
                name=f"Transition {trans_def.id}",
                from_states=[trans_def.from_state] if trans_def.from_state else [],
                activate_states=trans_def.activate_states,
                exit_states=trans_def.deactivate_states,
            )

        # Determine starting states
        from_states: set[str] = set()
        if request.from_states:
            from_states = set(request.from_states) & state_id_set
        else:
            # Use initial states
            from_states = {s.id for s in request.states if s.is_initial}

        if not from_states:
            # Fall back to first state
            from_states = {request.states[0].id}

        # Find path to ALL target states
        logger.info(f"Finding path from {from_states} to targets {request.target_states}")

        path = manager.find_path_to(
            target_state_ids=list(request.target_states),
            from_states=from_states,
        )

        if path is None:
            # Analyze why path doesn't exist
            reachable_from_start = manager.get_reachable_states(max_depth=20)
            unreachable_targets = set(request.target_states) - reachable_from_start

            if unreachable_targets:
                return PathValidationResponse(
                    reachable=False,
                    reason=f"States unreachable from starting position: {', '.join(unreachable_targets)}",
                    details={
                        "starting_states": list(from_states),
                        "reachable_states": list(reachable_from_start),
                        "unreachable_targets": list(unreachable_targets),
                    },
                )
            else:
                return PathValidationResponse(
                    reachable=False,
                    reason="All targets are individually reachable, but cannot all be active simultaneously",
                    details={
                        "starting_states": list(from_states),
                        "target_states": request.target_states,
                        "note": "The targets may be mutually exclusive or require conflicting transitions",
                    },
                )

        # Convert path to response format
        path_steps = []
        for i, transition in enumerate(path.transitions_sequence):
            # Get states before and after this transition
            states_before = path.states_sequence[i] if i < len(path.states_sequence) else set()
            states_after = (
                path.states_sequence[i + 1] if i + 1 < len(path.states_sequence) else set()
            )

            path_steps.append(
                PathStep(
                    transition_id=transition.id,
                    from_states=[s.id for s in states_before],
                    to_states=[s.id for s in states_after],
                )
            )

        return PathValidationResponse(
            reachable=True,
            path=path_steps,
            reason=f"Path found with {len(path_steps)} transitions",
            details={
                "total_cost": path.total_cost,
                "starting_states": list(from_states),
                "final_states": (
                    [s.id for s in path.states_sequence[-1]] if path.states_sequence else []
                ),
            },
        )

    except Exception as e:
        logger.error(f"Path validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Path validation failed: {str(e)}") from e


@router.post("/analyze-reachability")
async def analyze_reachability(
    request: PathValidationRequest,
) -> dict[str, Any]:
    """Analyze reachability of all states from starting position.

    This is a more detailed analysis than validate, returning information
    about which states are reachable and how.

    Args:
        request: Same as validation request

    Returns:
        Detailed reachability analysis
    """
    try:
        if not request.states:
            return {"error": "No states provided"}

        # Build multistate graph
        config = StateManagerConfig(
            default_search_strategy=SearchStrategy.BFS,
            max_path_depth=50,
        )
        manager = StateManager(config)

        # Register states
        state_id_set = set()
        for state_def in request.states:
            manager.add_state(
                id=state_def.id,
                name=state_def.name,
                blocking=state_def.is_blocking,
            )
            state_id_set.add(state_def.id)

        # Register transitions
        for trans_def in request.transitions:
            if trans_def.from_state and trans_def.from_state not in state_id_set:
                continue
            invalid_activate = set(trans_def.activate_states) - state_id_set
            if invalid_activate:
                continue

            manager.add_transition(
                id=trans_def.id,
                name=f"Transition {trans_def.id}",
                from_states=[trans_def.from_state] if trans_def.from_state else [],
                activate_states=trans_def.activate_states,
                exit_states=trans_def.deactivate_states,
            )

        # Determine starting states
        from_states: set[str] = set()
        if request.from_states:
            from_states = set(request.from_states) & state_id_set
        else:
            from_states = {s.id for s in request.states if s.is_initial}
        if not from_states and request.states:
            from_states = {request.states[0].id}

        # Activate starting states
        manager.activate_states(from_states)

        # Get reachability info
        reachable = manager.get_reachable_states(max_depth=30)
        available_transitions = manager.get_available_transitions()

        # Analyze each target
        target_analysis = {}
        for target in request.target_states:
            if target not in state_id_set:
                target_analysis[target] = {
                    "status": "not_found",
                    "reachable": False,
                }
            elif target in reachable:
                # Try to find path to just this target
                path = manager.find_path_to([target], from_states=from_states)
                target_analysis[target] = {
                    "status": "reachable",
                    "reachable": True,
                    "path_length": len(path.transitions_sequence) if path else 0,
                }
            else:
                target_analysis[target] = {
                    "status": "unreachable",
                    "reachable": False,
                }

        return {
            "starting_states": list(from_states),
            "all_states": list(state_id_set),
            "reachable_states": list(reachable),
            "unreachable_states": list(state_id_set - reachable),
            "available_transitions": available_transitions,
            "target_analysis": target_analysis,
            "complexity": manager.analyze_complexity(),
        }

    except Exception as e:
        logger.error(f"Reachability analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Reachability analysis failed: {str(e)}"
        ) from e
