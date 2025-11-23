"""
State Adapter for converting frontend state format to qontinui State format.

This adapter bridges the gap between the frontend's state representation
and the qontinui library's State class requirements.
"""

import base64
from io import BytesIO
from typing import Any

from qontinui.model.element import Image as QontinuiImage
from qontinui.model.state import State as QontinuiState


def convert_frontend_state_to_qontinui(frontend_state: dict[str, Any]) -> QontinuiState:
    """
    Convert frontend state definition to qontinui State object.

    Frontend format:
    {
        "id": "login_screen",
        "name": "Login Screen",
        "images": [{"data": "base64...", "id": "img1"}],
        "stateImages": [{"image": "base64...", "name": "login_button"}],
        "transitions": ["logged_in", "error"]
    }

    Qontinui format:
    State(name, state_enum, images, transitions)

    Args:
        frontend_state: Dictionary containing frontend state definition

    Returns:
        QontinuiState object ready for use with qontinui library
    """
    # Use state name as identifier (qontinui uses names, not IDs)
    state_name = frontend_state.get("name", frontend_state.get("id", ""))

    # Convert base64 images to qontinui Image objects
    qontinui_images = []

    # Handle both "images" and "stateImages" formats
    images_list = frontend_state.get("stateImages", [])
    if not images_list:
        images_list = frontend_state.get("images", [])

    for img_data in images_list:
        if isinstance(img_data, dict):
            # Get image data from various possible field names
            base64_str = img_data.get("image") or img_data.get("data") or img_data.get("imageData")

            if base64_str:
                try:
                    # Handle base64 encoded images
                    if "," in base64_str:
                        # Remove data URL prefix (e.g., "data:image/png;base64,")
                        base64_str = base64_str.split(",", 1)[1]

                    # Convert base64 to qontinui Image
                    image = QontinuiImage.from_base64(base64_str)
                    qontinui_images.append(image)
                except Exception as e:
                    # Log error but continue processing other images
                    print(f"Warning: Failed to convert image in state '{state_name}': {e}")
                    continue

    # Create State object
    # Note: qontinui State uses 'name' as the primary identifier
    state = QontinuiState(name=state_name)

    # Add images to state if any were successfully converted
    # Note: The exact method for adding images may vary based on qontinui State API
    # For now, we'll store them as an attribute
    if qontinui_images:
        # Depending on qontinui's State class API, this might need adjustment
        state.images = qontinui_images

    # Store transitions if provided
    transitions = frontend_state.get("transitions", [])
    if transitions:
        state.transitions = transitions

    return state


def convert_multiple_states(frontend_states: list[dict[str, Any]]) -> list[QontinuiState]:
    """
    Convert list of frontend states to qontinui States.

    Args:
        frontend_states: List of frontend state definitions

    Returns:
        List of QontinuiState objects
    """
    converted_states = []

    for state_dict in frontend_states:
        try:
            qontinui_state = convert_frontend_state_to_qontinui(state_dict)
            converted_states.append(qontinui_state)
        except Exception as e:
            # Log error but continue processing other states
            state_id = state_dict.get("id", state_dict.get("name", "unknown"))
            print(f"Error converting state '{state_id}': {e}")
            continue

    return converted_states
