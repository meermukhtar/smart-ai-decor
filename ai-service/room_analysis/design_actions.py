def match_design_action(action, detections):
    """
    Match a Gemini action with an actually detected
    furniture object using catalog_name.
    """

    catalog_name = action.get("catalog_name")

    if not catalog_name:
        return None

    for detection in detections:

        if detection.get("catalog_name") == catalog_name:
            return detection

    return None


def match_all_actions(actions, detections):
    """
    Match all Gemini actions with YOLO detections.

    Returns:
        [
            {
                "action": {...},
                "detection": {...}
            }
        ]
    """

    matched_actions = []

    for action in actions:

        detection = match_design_action(
            action,
            detections
        )

        matched_actions.append({
            "action": action,
            "detection": detection
        })

    return matched_actions
