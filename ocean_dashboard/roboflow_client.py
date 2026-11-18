import base64
import requests

WORKFLOW_URL = (
    "https://serverless.roboflow.com/"
    "leos-workspace-jcehb/workflows/ocean-ai-sam3-corrosion"
)

PROMPT = (
    "localized rust patches, "
    "corroded metal areas, "
    "corrosion pits"
)

def run_sam3(image_bytes, api_key):
    if not api_key:
        raise ValueError("Roboflow API key is missing.")

    payload = {
        "inputs": {
            "image": {
                "type": "base64",
                "value": base64.b64encode(
                    image_bytes
                ).decode("ascii")
            },
            "classes": PROMPT
        }
    }

    response = requests.post(
        WORKFLOW_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    data = response.json()

    output = data["outputs"][0]

    annotated = output["annotated_image"]["value"]

    if annotated.startswith("data:"):
        annotated = annotated.split(",", 1)[1]

    annotated_bytes = base64.b64decode(annotated)

    predictions = output["predictions"]["predictions"]

    return annotated_bytes, predictions
