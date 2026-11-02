"""Qt-independent status of a saved detection run against saved weights."""


def describe_run_model(run, *, current_model_id=None, model_state="unknown"):
    """Classify a scan's model version; never equate a filename with a version."""
    if model_state == "ready":
        return (
            "Current model — predictions match the saved fitted weights."
            if current_model_id == run.model_id else
            "Historical model — predictions were made with different fitted weights."
        )
    if model_state == "missing":
        return "Historical snapshot — no fitted weights are currently saved."
    if model_state == "checking":
        return "Checking saved model version…"
    if model_state == "error":
        return "Could not verify saved model version; predictions remain browsable."
    return "Model version has not been checked."
