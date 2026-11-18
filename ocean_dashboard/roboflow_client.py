"""OCEAN AI / Roboflow SAM 3 workflow client.

Preserves the original run_sam3(image_bytes, api_key) two-value interface
for older scripts while offering all three published Workflow outputs.
"""
import base64
import binascii
import requests

WORKFLOW_URL = (
    "https://serverless.roboflow.com/"
    "leos-workspace-jcehb/workflows/ocean-ai-sam3-corrosion"
)
PROMPT = "localized rust patches, corroded metal areas, corrosion pits"


def _image_bytes(value, field):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Roboflow output {field!r} is not a base64 image")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"Could not decode Roboflow {field!r}") from exc


def run_sam3_full(image_bytes, api_key):
    """Return {'clean', 'labeled', 'predictions'} from one Roboflow request.

    Optional image outputs are kept distinct: never pretend a labeled overlay
    is a clean visualization. The published Workflow must expose clean_preview.
    """
    if not api_key:
        raise ValueError("Roboflow API key is missing")
    if not image_bytes:
        raise ValueError("Image is empty")

    payload = {
        "inputs": {
            "image": {
                "type": "base64",
                "value": base64.b64encode(image_bytes).decode("ascii"),
            },
            "classes": PROMPT,
        }
    }
    response = requests.post(
        WORKFLOW_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # Roboflow's workflow preview can return the output list directly.
        output = data[0]
    elif isinstance(data, dict):
        outputs = data.get("outputs", data)
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            output = outputs[0]
        elif isinstance(outputs, dict):
            output = outputs
        else:
            raise ValueError("Unexpected Roboflow workflow response (no outputs)")
    else:
        raise ValueError("Unexpected Roboflow workflow response (no outputs)")

    clean = _image_bytes(output.get("clean_preview"), "clean_preview")
    labeled = _image_bytes(output.get("annotated_image"), "annotated_image")

    prediction_field = output.get("predictions", {})
    if isinstance(prediction_field, dict):
        predictions = prediction_field.get("predictions", [])
    elif isinstance(prediction_field, list):
        predictions = prediction_field
    else:
        predictions = []
    if not isinstance(predictions, list):
        predictions = []

    if clean is None and labeled is None:
        raise ValueError("Workflow provided neither clean_preview nor annotated_image")

    return {
        "clean": clean,
        "labeled": labeled,
        "predictions": predictions,
    }


def run_sam3(image_bytes, api_key):
    """Backwards-compatible two-value API; prefers clean overlays."""
    results = run_sam3_full(image_bytes, api_key)
    return results["clean"] or results["labeled"], results["predictions"]


def run_sam3_batch(image_bytes, api_key):
    """Batch API: preferred image, predictions, optional labeled image."""
    results = run_sam3_full(image_bytes, api_key)
    return (results["clean"] or results["labeled"],
            results["predictions"], results["labeled"])
