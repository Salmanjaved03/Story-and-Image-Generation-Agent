"""
MCP Tool Implementations
=========================
Defines and registers all tools that agents use via MCP discovery.
Each tool is a standalone function registered with the MCP registry.

Tools:
  - generate_script_segment: LLM-powered script generation
  - validate_script_structure: Script structure validation
  - extract_characters: Character extraction from scripts
  - generate_character_image: Image generation via HuggingFace
  - commit_memory: Store data to ChromaDB
  - query_memory: Retrieve data from ChromaDB
"""

import json
import os
import re
import requests
import uuid
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

import config
from mcp.server import get_mcp_registry
from memory.vector_store import get_memory_store


def _get_llm() -> ChatGroq:
    """Get configured Groq LLM instance."""
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model_name=config.GROQ_MODEL,
        temperature=0.7,
        max_tokens=4096
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 1: Generate Script Segment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_script_segment(prompt: str, num_scenes: int = 3, genre: str = "drama") -> Dict[str, Any]:
    """
    Generate a structured screenplay segment from a prompt using LLM.
    Returns a JSON-compatible dict with scenes, dialogues, and visual cues.
    """
    llm = _get_llm()

    system_prompt = """You are an expert screenwriter. Generate a structured screenplay based on the user's prompt.

OUTPUT FORMAT - Return ONLY valid JSON with this exact structure:
{
  "title": "Story Title",
  "genre": "genre",
  "total_scenes": <number>,
  "scenes": [
    {
      "scene_number": 1,
      "heading": "INT./EXT. LOCATION - TIME",
      "setting": "Description of the setting",
      "characters_present": ["Character1", "Character2"],
      "action": "What happens in this scene",
      "dialogue": [
        {
          "character": "CHARACTER_NAME",
          "line": "What they say",
          "direction": "(emotional direction)"
        }
      ],
      "visual_cues": {
        "lighting": "Description of lighting",
        "camera_angle": "Description of camera angle",
        "mood": "Overall visual mood",
        "color_palette": "Dominant colors"
      },
      "transition": "CUT TO / FADE TO / etc."
    }
  ]
}

RULES:
- Generate exactly the requested number of scenes
- Every scene MUST have at least one dialogue exchange
- Every scene MUST have visual cues
- Characters must be consistent across scenes
- Return ONLY the JSON, no markdown formatting, no code blocks"""

    user_prompt = f"""Create a {genre} screenplay with {num_scenes} scenes based on this concept:

{prompt}

Remember: Return ONLY valid JSON, no markdown."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    # Parse the response
    content = response.content.strip()
    # Remove any markdown code block markers if present
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    try:
        script_data = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            script_data = json.loads(json_match.group())
        else:
            script_data = {
                "title": "Generated Script",
                "genre": genre,
                "total_scenes": 0,
                "scenes": [],
                "error": "Failed to parse LLM response",
                "raw_response": content[:500]
            }

    return script_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 2: Validate Script Structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_script_structure(script_text: str) -> Dict[str, Any]:
    """
    Validate a manually provided script for required structure.
    Checks: scene headers, dialogue labels, action descriptions.
    Returns validation result with errors/suggestions.
    """
    llm = _get_llm()

    system_prompt = """You are a script validation expert. Analyze the provided script and validate its structure.

Check for:
1. Scene headings (INT./EXT. LOCATION - TIME format)
2. Dialogue labels (CHARACTER NAME followed by their line)
3. Action/stage descriptions
4. Visual cue possibilities

OUTPUT FORMAT - Return ONLY valid JSON:
{
  "is_valid": true/false,
  "total_scenes_found": <number>,
  "errors": ["list of structural errors found"],
  "warnings": ["list of suggestions for improvement"],
  "parsed_scenes": [
    {
      "scene_number": 1,
      "heading": "Scene heading as found or inferred",
      "setting": "Setting description",
      "characters_present": ["Character1"],
      "action": "Action description",
      "dialogue": [
        {
          "character": "CHARACTER",
          "line": "Their dialogue",
          "direction": "(direction if any)"
        }
      ],
      "visual_cues": {
        "lighting": "Inferred lighting",
        "camera_angle": "Suggested camera angle",
        "mood": "Scene mood",
        "color_palette": "Suggested colors"
      },
      "transition": "Transition type"
    }
  ]
}

Return ONLY the JSON, no markdown."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Validate this script:\n\n{script_text}")
    ])

    content = response.content.strip()
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "is_valid": False,
                "errors": ["Could not parse validation response"],
                "warnings": [],
                "parsed_scenes": []
            }

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 3: Extract Characters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_characters(script_data: dict) -> Dict[str, Any]:
    """
    Extract and formalize character identities from a script.
    Returns structured character database with appearance, traits, etc.
    """
    llm = _get_llm()

    system_prompt = """You are a character design expert. Analyze the screenplay and extract detailed character profiles.

For each character mentioned in the script, create a comprehensive profile.

OUTPUT FORMAT - Return ONLY valid JSON:
{
  "characters": [
    {
      "id": "char_001",
      "name": "Character Name",
      "role": "protagonist/antagonist/supporting/minor",
      "personality_traits": ["trait1", "trait2", "trait3"],
      "appearance": {
        "age_range": "20-30",
        "build": "slim/athletic/stocky/etc",
        "hair": "color and style",
        "eyes": "color",
        "skin_tone": "description",
        "distinguishing_features": "any notable features",
        "typical_clothing": "what they usually wear"
      },
      "background": "Brief character background",
      "motivation": "What drives this character",
      "reference_style": "Art style reference for visual generation",
      "scenes_appeared": [1, 2, 3],
      "image_prompt": "Detailed prompt for generating this character's portrait"
    }
  ],
  "total_characters": <number>,
  "relationships": [
    {
      "between": ["Character1", "Character2"],
      "type": "friends/enemies/family/romantic/professional"
    }
  ]
}

Return ONLY the JSON, no markdown."""

    script_json = json.dumps(script_data, indent=2)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract characters from this screenplay:\n\n{script_json}")
    ])

    content = response.content.strip()
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "characters": [],
                "total_characters": 0,
                "relationships": [],
                "error": "Failed to parse character extraction response"
            }

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 4: Generate Character Image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import time


def _try_huggingface_model(model_id: str, prompt: str, headers: dict, max_retries: int = 2) -> Optional[bytes]:
    """
    Try generating an image with a specific HuggingFace model.
    Handles 503 (model loading) with retries and wait_for_model flag.
    Returns image bytes on success, None on failure.
    """
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"

    payload = {
        "inputs": prompt,
        "options": {
            "wait_for_model": True,
            "use_cache": True
        }
    }

    for attempt in range(max_retries + 1):
        try:
            print(f"      [Attempt {attempt + 1}] Trying model: {model_id}...")
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=180
            )

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "image" in content_type or len(response.content) > 1000:
                    print(f"      ✅ Success with model: {model_id}")
                    return response.content
                else:
                    print(f"      ⚠ Got non-image response from {model_id}")
                    return None

            elif response.status_code == 503:
                # Model is loading - wait and retry
                try:
                    error_data = response.json()
                    wait_time = error_data.get("estimated_time", 30)
                except Exception:
                    wait_time = 30
                print(f"      ⏳ Model loading, waiting {min(wait_time, 60):.0f}s...")
                time.sleep(min(wait_time, 60))
                continue

            elif response.status_code == 429:
                # Rate limited
                print(f"      ⚠ Rate limited on {model_id}, waiting 10s...")
                time.sleep(10)
                continue

            else:
                error_text = ""
                try:
                    error_text = response.json().get("error", response.text[:200])
                except Exception:
                    error_text = response.text[:200]
                print(f"      ❌ Model {model_id} returned {response.status_code}: {error_text}")
                return None

        except requests.exceptions.Timeout:
            print(f"      ⏱ Timeout on {model_id} (attempt {attempt + 1})")
            continue
        except Exception as e:
            print(f"      ❌ Error with {model_id}: {e}")
            return None

    return None


def generate_character_image(
    character_name: str,
    image_prompt: str,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Generate a character portrait image using HuggingFace Stable Diffusion API.
    Tries multiple models as fallback. Saves the image to the output directory.
    """
    if output_dir is None:
        output_dir = config.IMAGE_ASSETS_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Enhance the prompt for better character portraits
    enhanced_prompt = (
        f"Professional character portrait, {image_prompt}, "
        f"high quality, detailed, cinematic lighting, "
        f"concept art style"
    )

    headers = {
        "Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"
    }

    safe_name = re.sub(r'[^\w\-_]', '_', character_name.lower())
    filename = f"{safe_name}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(output_dir, filename)

    # Try each fallback model
    models_to_try = config.HUGGINGFACE_FALLBACK_MODELS
    print(f"    🔄 Attempting image generation with {len(models_to_try)} model(s)...")

    for model_id in models_to_try:
        image_bytes = _try_huggingface_model(model_id, enhanced_prompt, headers)
        if image_bytes:
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            return {
                "success": True,
                "character_name": character_name,
                "image_path": filepath,
                "prompt_used": enhanced_prompt,
                "model_used": model_id
            }

    # All models failed — create a high-quality placeholder
    print(f"    ⚠ All API models failed. Creating placeholder image...")
    return _create_placeholder_image(character_name, filepath, enhanced_prompt)


def _create_placeholder_image(
    character_name: str, filepath: str, prompt: str
) -> Dict[str, Any]:
    """Create a styled placeholder image when all APIs are unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random
        import hashlib

        # Use character name to seed consistent colors
        name_hash = int(hashlib.md5(character_name.encode()).hexdigest()[:8], 16)
        hue = name_hash % 360

        # Create a gradient-like background
        img = Image.new('RGB', (512, 512), color=(20, 20, 35))
        draw = ImageDraw.Draw(img)

        # Draw decorative elements based on character name hash
        r = (hue * 3) % 180 + 40
        g = (hue * 7) % 150 + 40
        b = (hue * 11) % 200 + 55

        # Outer frame
        draw.rectangle([(8, 8), (503, 503)], outline=(r, g, b), width=3)
        # Inner frame
        draw.rectangle([(20, 20), (491, 491)], outline=(r // 2, g // 2, b // 2), width=1)

        # Silhouette circle
        cx, cy = 256, 200
        for radius in range(80, 40, -5):
            alpha = int((radius - 40) / 40 * 60) + 30
            draw.ellipse(
                [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                outline=(r, g, b),
                width=1
            )

        # Draw a simple silhouette head/shoulders
        draw.ellipse([(226, 150), (286, 210)], fill=(r // 3, g // 3, b // 3), outline=(r, g, b), width=2)
        draw.arc([(196, 210), (316, 300)], start=0, end=180, fill=(r, g, b), width=2)

        # Load fonts
        try:
            font_large = ImageFont.truetype("arial.ttf", 26)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 13)
        except OSError:
            font_large = ImageFont.load_default()
            font_medium = font_large
            font_small = font_large

        # Character name
        draw.text((256, 330), character_name, fill=(220, 225, 240), font=font_large, anchor="mm")

        # Decorative line
        draw.line([(120, 355), (392, 355)], fill=(r, g, b), width=1)

        # Labels
        draw.text((256, 380), "CHARACTER REFERENCE", fill=(r, g, b), font=font_medium, anchor="mm")
        draw.text((256, 410), "PROJECT MONTAGE", fill=(120, 130, 150), font=font_small, anchor="mm")

        # Prompt snippet at bottom
        short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        draw.text((256, 460), short_prompt, fill=(80, 90, 110), font=font_small, anchor="mm")

        img.save(filepath, quality=95)
        return {
            "success": True,
            "character_name": character_name,
            "image_path": filepath,
            "prompt_used": prompt,
            "note": "Placeholder image created (API models unavailable). Replace HuggingFace API key for real generation."
        }
    except Exception as e:
        return {
            "success": False,
            "character_name": character_name,
            "image_path": None,
            "error": f"Image generation failed and placeholder creation also failed: {e}"
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 5: Commit to Memory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def commit_memory(data: dict, data_type: str, metadata: dict = None) -> Dict[str, Any]:
    """
    Commit data to the persistent vector memory store.
    Supports: script_history, character_metadata, image_reference
    """
    store = get_memory_store()
    doc_id = f"{data_type}_{uuid.uuid4().hex[:8]}"

    text_content = json.dumps(data, indent=2)
    meta = {
        "type": data_type,
        "doc_id": doc_id,
        **(metadata or {})
    }

    store.add_document(
        doc_id=doc_id,
        content=text_content,
        metadata=meta
    )

    return {
        "success": True,
        "doc_id": doc_id,
        "data_type": data_type,
        "message": f"Committed {data_type} to memory store"
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 6: Query Memory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def query_memory(query: str, data_type: str = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Query the persistent vector memory store.
    Optionally filter by data_type.
    """
    store = get_memory_store()
    results = store.query(query=query, top_k=top_k, filter_type=data_type)
    return {
        "success": True,
        "query": query,
        "results": results,
        "total_results": len(results)
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REGISTRATION: Register all tools with MCP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def register_all_tools():
    """Register all tools with the MCP registry for dynamic discovery."""
    registry = get_mcp_registry()

    print("\n╔══════════════════════════════════════════╗")
    print("║   MCP Tool Registration                  ║")
    print("╚══════════════════════════════════════════╝")

    # 1. Script Generation
    registry.register_tool(
        name="generate_script_segment",
        description="Generate a structured screenplay segment from a text prompt using LLM",
        handler=generate_script_segment,
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Creative prompt for script generation"},
                "num_scenes": {"type": "integer", "description": "Number of scenes to generate", "default": 3},
                "genre": {"type": "string", "description": "Genre of the screenplay", "default": "drama"}
            },
            "required": ["prompt"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "scenes": {"type": "array"}
            }
        },
        category="scriptwriting"
    )

    # 2. Script Validation
    registry.register_tool(
        name="validate_script_structure",
        description="Validate the structure of a manually provided screenplay",
        handler=validate_script_structure,
        input_schema={
            "type": "object",
            "properties": {
                "script_text": {"type": "string", "description": "Raw script text to validate"}
            },
            "required": ["script_text"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "errors": {"type": "array"},
                "parsed_scenes": {"type": "array"}
            }
        },
        category="validation"
    )

    # 3. Character Extraction
    registry.register_tool(
        name="extract_characters",
        description="Extract and formalize character identities from a structured screenplay",
        handler=extract_characters,
        input_schema={
            "type": "object",
            "properties": {
                "script_data": {"type": "object", "description": "Structured script data with scenes"}
            },
            "required": ["script_data"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "characters": {"type": "array"},
                "total_characters": {"type": "integer"}
            }
        },
        category="character_design"
    )

    # 4. Image Generation
    registry.register_tool(
        name="generate_character_image",
        description="Generate a character portrait image using Stable Diffusion via HuggingFace",
        handler=generate_character_image,
        input_schema={
            "type": "object",
            "properties": {
                "character_name": {"type": "string", "description": "Name of the character"},
                "image_prompt": {"type": "string", "description": "Detailed visual description for image generation"},
                "output_dir": {"type": "string", "description": "Directory to save the image"}
            },
            "required": ["character_name", "image_prompt"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "image_path": {"type": "string"}
            }
        },
        category="image_synthesis"
    )

    # 5. Memory Commit
    registry.register_tool(
        name="commit_memory",
        description="Store data persistently in the vector memory database",
        handler=commit_memory,
        input_schema={
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "Data to store"},
                "data_type": {"type": "string", "description": "Type: script_history, character_metadata, image_reference"},
                "metadata": {"type": "object", "description": "Optional metadata"}
            },
            "required": ["data", "data_type"]
        },
        category="memory"
    )

    # 6. Memory Query
    registry.register_tool(
        name="query_memory",
        description="Query the vector memory database for stored data",
        handler=query_memory,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "data_type": {"type": "string", "description": "Filter by type"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5}
            },
            "required": ["query"]
        },
        category="memory"
    )

    print(f"\n  [MCP] Total tools registered: {len(registry.list_tool_names())}")
    print(f"  [MCP] Available tools: {registry.list_tool_names()}\n")
