"""
Character Designer Agent
=========================
Extracts and formalizes character identities from scripts.

Outputs:
  - Name
  - Personality traits
  - Appearance description
  - Reference style

Key Feature:
  - Maintains identity consistency across scenes

MCP Tools Used:
  - extract_characters
  - commit_memory
"""

import json
from typing import Any, Dict

from graph.state import WriterRoomState
from mcp.server import get_mcp_registry


class CharacterDesignerAgent:
    """
    Character Designer Agent - Extracts and formalizes character identities.
    
    Analyzes the script to create comprehensive character profiles
    with visual descriptions suitable for image generation.
    """

    def __init__(self):
        self.name = "Character Designer Agent"
        self.role = "Extract and formalize character identities from scripts"
        self.mcp = get_mcp_registry()

    def _discover_tools(self):
        """Discover character design tools via MCP registry."""
        tools = self.mcp.discover_tools(category="character_design")
        memory_tools = self.mcp.discover_tools(category="memory")
        all_tools = tools + memory_tools
        print(f"  [{self.name}] Discovered {len(all_tools)} tools via MCP")
        for tool in all_tools:
            print(f"    → {tool['name']}: {tool['description']}")
        return all_tools

    def run(self, state: WriterRoomState) -> Dict[str, Any]:
        """
        Execute character design reasoning loop:
        1. Discover MCP tools
        2. Extract characters from script
        3. Formalize character profiles
        4. Commit to memory
        """
        print(f"\n{'='*60}")
        print(f"  🎨 {self.name} - Starting Character Extraction")
        print(f"{'='*60}")

        # Step 1: Discover tools
        self._discover_tools()

        script_data = state.get("scene_manifest") or state.get("raw_script")

        if not script_data:
            print("  ⚠ No script data available")
            return {
                "character_db": None,
                "errors": state.get("errors", []) + ["Character Designer: No script data"],
                "current_stage": "character_extraction_failed",
                "status": "failed"
            }

        # Step 2: Extract characters via MCP
        print(f"\n  [Step 1] Extracting characters via MCP tool: extract_characters")
        character_data = self.mcp.invoke_tool(
            "extract_characters",
            script_data=script_data
        )

        characters = character_data.get("characters", [])
        total = character_data.get("total_characters", len(characters))

        print(f"\n  [Step 2] Characters Extracted: {total}")
        for char in characters:
            name = char.get("name", "Unknown")
            role = char.get("role", "unknown")
            traits = ", ".join(char.get("personality_traits", [])[:3])
            print(f"    → {name} ({role}): {traits}")

            appearance = char.get("appearance", {})
            if appearance:
                print(f"      Appearance: {appearance.get('hair', 'N/A')} hair, "
                      f"{appearance.get('build', 'N/A')} build, "
                      f"age {appearance.get('age_range', 'N/A')}")

        # Step 3: Ensure identity consistency across scenes
        print(f"\n  [Step 3] Verifying identity consistency across scenes...")
        scenes = script_data.get("scenes", [])
        for char in characters:
            char_name = char.get("name", "")
            scenes_with_char = []
            for scene in scenes:
                chars_in_scene = scene.get("characters_present", [])
                if char_name.lower() in [c.lower() for c in chars_in_scene]:
                    scenes_with_char.append(scene.get("scene_number", 0))
            char["scenes_appeared"] = scenes_with_char
            print(f"    {char_name}: appears in scenes {scenes_with_char}")

        # Step 4: Commit to memory
        print(f"\n  [Step 4] Committing character data to memory via MCP")
        self.mcp.invoke_tool(
            "commit_memory",
            data=character_data,
            data_type="character_metadata",
            metadata={"source": "character_designer_agent"}
        )

        # Build final character DB
        character_db = {
            "characters": characters,
            "total_characters": total,
            "relationships": character_data.get("relationships", []),
            "source_script": script_data.get("title", "Unknown")
        }

        print(f"\n  ✅ {self.name} completed - {total} characters profiled\n")

        return {
            "character_db": character_db,
            "current_stage": "characters_extracted",
            "status": "in_progress"
        }


def character_node(state: WriterRoomState) -> Dict[str, Any]:
    """LangGraph node function for the Character Designer Agent."""
    agent = CharacterDesignerAgent()
    return agent.run(state)
