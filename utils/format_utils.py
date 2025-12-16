def clean_json_string(raw_str: str) -> str:
    """
    Remove leading/trailing Markdown code fences (```json ... ```)
    and return a clean JSON string.
    """
    # Strip leading/trailing whitespace
    cleaned = raw_str.strip()

    # Remove starting ```json or ``` if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].lstrip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].lstrip()

    # Remove ending ``` if present
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")].rstrip()

    return cleaned



def normalize_bounding_boxes(data, width, height):
    """
    Normalize all bounding boxes in the given dictionary structure.

    Args:
        data (dict): Dictionary containing objects and object_details
        width (int/float): Width of the image/frame
        height (int/float): Height of the image/frame

    Returns:
        dict: Dictionary with normalized bounding boxes (values between 0 and 1)
    """
    # Create a deep copy to avoid modifying the original data
    import copy
    normalized_data = copy.deepcopy(data)

    # Check if object_details exists in the data
    if "object_details" not in normalized_data:
        return normalized_data

    # Iterate through all objects in object_details
    for obj_name, obj_info in normalized_data["object_details"].items():
        if "bounding_box" in obj_info:
            bbox = obj_info["bounding_box"]

            # Ensure bounding_box has 4 coordinates
            if len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox

                # Normalize coordinates by dividing by width and height
                normalized_bbox = [
                    x_min / width,  # x_min normalized
                    y_min / height, # y_min normalized
                    x_max / width,  # x_max normalized
                    y_max / height  # y_max normalized
                ]

                # Update the bounding box with normalized values
                obj_info["bounding_box"] = normalized_bbox

    return normalized_data
