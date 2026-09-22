"""
Image Synthesizer Agent
========================
Generates visual representations of characters.

Implementation:
  - Stable Diffusion via HuggingFace Inference API
  - Accessed via MCP (no hardcoded API calls)

Output:
  - Character reference images saved to image_assets/
"""

import json
import os
from typing import Any, Dict, List

import config
from graph.state import WriterRoomState
from mcp.server import get_mcp_registry


class ImageSynthesizerAgent:
    """
    Image Synthesizer Agent - Generates character visual representations.
    
    Uses Stable Diffusion via HuggingFace API (through MCP discovery)
    to create character reference images.
    """

    def __init__(self):
        self.name = "Image Synthesizer Agent"
        self.role = "Generate visual representations of characters"
        self.mcp = get_mcp_registry()

    def _discover_tools(self):
        """Discover image synthesis tools via MCP registry."""
        tools = self.mcp.discover_tools(category="image_synthesis")
        memory_tools = self.mcp.discover_tools(category="memory")
        all_tools = tools + memory_tools
        print(f"  [{self.name}] Discovered {len(all_tools)} tools via MCP")
        for tool in all_tools:
            print(f"    → {tool['name']}: {tool['description']}")
        return all_tools

    def _build_image_prompt(self, character: Dict[str, Any]) -> str:
        """Build a detailed image prompt from character data."""
        # Use the image_prompt if available from character extraction
        if character.get("image_prompt"):
            return character["image_prompt"]

        # Build from appearance data
        appearance = character.get("appearance", {})
        name = character.get("name", "Character")

        parts = [f"Portrait of {name}"]

        if appearance.get("age_range"):
            parts.append(f"age {appearance['age_range']}")
        if appearance.get("build"):
            parts.append(f"{appearance['build']} build")
        if appearance.get("hair"):
            parts.append(f"{appearance['hair']} hair")
        if appearance.get("eyes"):
            parts.append(f"{appearance['eyes']} eyes")
        if appearance.get("skin_tone"):
            parts.append(f"{appearance['skin_tone']} skin")
        if appearance.get("typical_clothing"):
            parts.append(f"wearing {appearance['typical_clothing']}")
        if appearance.get("distinguishing_features"):
            parts.append(appearance["distinguishing_features"])

        # Add style reference
        ref_style = character.get("reference_style", "cinematic concept art")
        parts.append(f"in {ref_style} style")

        return ", ".join(parts)

    def run(self, state: WriterRoomState) -> Dict[str, Any]:
        """
        Execute image synthesis reasoning loop:
        1. Discover MCP tools
        2. For each character, generate a portrait image
        3. Save images to image_assets/
        4. Commit image references to memory
        """
        print(f"\n{'='*60}")
        print(f"  🖼️  {self.name} - Starting Image Generation")
        print(f"{'='*60}")

        # Step 1: Discover tools
        self._discover_tools()

        character_db = state.get("character_db")
        if not character_db:
            print("  ⚠ No character database available")
            return {
                "image_results": [],
                "errors": state.get("errors", []) + ["Image Synthesizer: No character data"],
                "current_stage": "image_generation_failed",
                "status": "failed"
            }

        characters = character_db.get("characters", [])
        print(f"\n  [Step 1] Generating images for {len(characters)} characters")

        # Ensure output directory exists
        os.makedirs(config.IMAGE_ASSETS_DIR, exist_ok=True)

        # Step 2: Generate image for each character
        image_results: List[Dict[str, Any]] = []

        for i, character in enumerate(characters, 1):
            name = character.get("name", f"character_{i}")
            print(f"\n  [Step 2.{i}] Generating portrait for: {name}")

            # Build the image prompt
            image_prompt = self._build_image_prompt(character)
            print(f"    Prompt: {image_prompt[:80]}...")

            # Generate via MCP tool
            result = self.mcp.invoke_tool(
                "generate_character_image",
                character_name=name,
                image_prompt=image_prompt,
                output_dir=config.IMAGE_ASSETS_DIR
            )

            image_results.append(result)

            if result.get("success"):
                print(f"    ✅ Image saved: {result.get('image_path', 'N/A')}")
                if result.get("note"):
                    print(f"    ℹ  {result['note'][:80]}")
            else:
                print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")

        # Step 3: Commit image references to memory
        print(f"\n  [Step 3] Committing image references to memory")
        for result in image_results:
            if result.get("success"):
                self.mcp.invoke_tool(
                    "commit_memory",
                    data={
                        "character_name": result.get("character_name"),
                        "image_path": result.get("image_path"),
                        "prompt_used": result.get("prompt_used")
                    },
                    data_type="image_reference",
                    metadata={"character": result.get("character_name", "")}
                )

        successful = sum(1 for r in image_results if r.get("success"))
        print(f"\n  ✅ {self.name} completed - {successful}/{len(characters)} images generated\n")

        return {
            "image_results": image_results,
            "current_stage": "images_generated",
            "status": "in_progress"
        }


def image_node(state: WriterRoomState) -> Dict[str, Any]:
    """LangGraph node function for the Image Synthesizer Agent."""
    agent = ImageSynthesizerAgent()
    return agent.run(state)
