"""
comfy_agent_demo.py — an LLM agent that drives ComfyUI through its HTTP API.

Setup:
  1. Start ComfyUI (default address http://127.0.0.1:8188).
  2. In ComfyUI: Workflow menu -> Export (API) -> save as workflow_api.json next to this script.
     (Use the default text-to-image workflow, or any graph with one KSampler.)
  3. pip install anthropic requests python-dotenv
  4. Put ANTHROPIC_API_KEY=sk-ant-... in a .env file next to this script

Run:
  python comfy_agent_demo.py "Design four logo ideas for MassQL, a SaaS tool for drawing SQL database diagrams"
"""

import json
import random
import sys
import time
import uuid
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

# Read ANTHROPIC_API_KEY from the .env file next to this script (if present).
# Variables already set in the terminal take priority over the file.
load_dotenv(Path(__file__).resolve().parent / ".env")

COMFY_URL = "http://127.0.0.1:8188"
SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW_FILE = SCRIPT_DIR / "workflow_api.json"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are an art director working with an image generator.
Use the generate_image tool to create images for the user's brief.
Write detailed, visual prompts: subject, style, composition, colours, background.
Image models match words to visuals, so describe instead of instructing.
For logos: ONE single centred symbol per image, flat vector style, on a plain white background.
Never describe a scene, a mockup, a table, wood, a room, or several objects. No lettering (image models garble text).
Use a different seed for each new idea; reuse the same seed when refining an idea.
Always act, never just plan: call generate_image for every idea in the brief before you reply.
Do not ask for permission and do not say what you would do. Run the tool, then explain the results.
When you are done, briefly explain each image and why you chose it."""

TOOLS = [
    {
        "name": "generate_image",
        "description": "Render one image with ComfyUI from a text prompt. Returns the saved file path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed visual description of the image.",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Things to avoid, e.g. 'text, letters, blurry'.",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed. Same seed + same prompt = same image.",
                },
            },
            "required": ["prompt"],
        },
    }
]


# ---------- ComfyUI side ----------


def find_node(workflow: dict, class_type: str) -> str:
    """Return the id of the first node with this class_type (ids differ between exports)."""
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id
    raise KeyError(f"No {class_type} node in {WORKFLOW_FILE}")


def generate_image(
    prompt: str,
    negative_prompt: str = "text, letters, watermark, blurry, wood, texture, table, mockup, "
    "multiple objects, collage, cropped, shadow, photo, 3d",
    seed: int | None = None,
) -> str:
    workflow = json.loads(WORKFLOW_FILE.read_text())
    sampler_id = find_node(workflow, "KSampler")
    sampler = workflow[sampler_id]["inputs"]

    # Follow the wires: the sampler's positive/negative inputs point at the CLIP Text Encode nodes.
    positive_id = sampler["positive"][0]
    negative_id = sampler["negative"][0]
    workflow[positive_id]["inputs"]["text"] = prompt
    workflow[negative_id]["inputs"]["text"] = negative_prompt
    sampler["seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)

    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]
    print(f"  queued {prompt_id} (seed {sampler['seed']})")

    # Poll /history until the job shows outputs.
    deadline = time.time() + 600
    while time.time() < deadline:
        history = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30).json()
        if prompt_id in history:
            break
        time.sleep(1)
    else:
        return "Error: generation timed out."

    OUTPUT_DIR.mkdir(exist_ok=True)
    saved = []
    for node_output in history[prompt_id]["outputs"].values():
        for img in node_output.get("images", []):
            params = {
                "filename": img["filename"],
                "subfolder": img["subfolder"],
                "type": img["type"],
            }
            data = requests.get(f"{COMFY_URL}/view", params=params, timeout=60).content
            path = OUTPUT_DIR / img["filename"]
            path.write_bytes(data)
            saved.append(str(path))
    return f"Saved: {', '.join(saved)} (seed {sampler['seed']})"


# ---------- Agent side ----------


def run_agent(brief: str) -> None:
    client = anthropic.Anthropic()
    task = f"{brief}\n\nGenerate the images now with the generate_image tool."
    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\nAgent: {block.text}")

        if response.stop_reason != "tool_use":
            break

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n-> generate_image: {block.input.get('prompt')}")
                try:
                    output = generate_image(**block.input)
                except (
                    Exception
                ) as exc:  # report errors back to the agent instead of crashing
                    output = f"Error: {exc}"
                print(f"  {output}")
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    if not WORKFLOW_FILE.exists():
        sys.exit(
            f"Missing {WORKFLOW_FILE.name}: export your workflow with Workflow > Export (API) into {SCRIPT_DIR}"
        )
    brief = (
        " ".join(sys.argv[1:])
        or "Design four logo ideas for MassQL, a SaaS tool for drawing SQL database diagrams."
    )
    run_agent(brief)
