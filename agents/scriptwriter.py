"""
Scriptwriter Agent
===================
Transforms abstract prompts into structured, production-ready scripts.

Reasoning Loop:
  1. Interpret the user's prompt
  2. Decompose into scenes
  3. Generate dialogue
  4. Attach visual context

MCP Tools Used:
  - generate_script_segment
  - commit_memory
"""

import json
from typing import Any, Dict

from graph.state import WriterRoomState
from mcp.server import get_mcp_registry


class ScriptwriterAgent:
    """
    Scriptwriter Agent - Creates structured screenplays from prompts.
    
    This agent transforms abstract user prompts into multi-scene,
    production-ready scripts with dialogues and visual cues.
    All tool access is through MCP discovery.
    """

    def __init__(self):
        self.name = "Scriptwriter Agent"
        self.role = "Transform abstract prompts into structured, production-ready scripts"
        self.mcp = get_mcp_registry()

    def _discover_tools(self):
        """Discover available tools via MCP registry."""
        tools = self.mcp.discover_tools(category="scriptwriting")
        print(f"  [{self.name}] Discovered {len(tools)} scriptwriting tools via MCP")
        for tool in tools:
            print(f"    → {tool['name']}: {tool['description']}")
        return tools

    def run(self, state: WriterRoomState) -> Dict[str, Any]:
        """
        Execute the scriptwriter reasoning loop:
        1. Discover available MCP tools
        2. Interpret the user prompt
        3. Generate structured screenplay via MCP tool
        4. Return updated state
        """
        print(f"\n{'='*60}")
        print(f"  🎬 {self.name} - Starting Reasoning Loop")
        print(f"{'='*60}")

        # Step 1: Discover tools via MCP
        self._discover_tools()

        prompt = state.get("user_input", "")
        genre = state.get("genre", "drama")
        num_scenes = state.get("num_scenes", 3)

        print(f"\n  [Step 1] Interpreting prompt:")
        print(f"    Prompt: {prompt[:100]}...")
        print(f"    Genre: {genre}")
        print(f"    Scenes: {num_scenes}")

        # Step 2: Generate script via MCP tool invocation
        print(f"\n  [Step 2] Generating screenplay via MCP tool: generate_script_segment")
        script_data = self.mcp.invoke_tool(
            "generate_script_segment",
            prompt=prompt,
            num_scenes=num_scenes,
            genre=genre
        )

        if isinstance(script_data, dict) and "error" in script_data and "scenes" not in script_data:
            print(f"  [ERROR] Script generation failed: {script_data['error']}")
            return {
                "raw_script": None,
                "errors": state.get("errors", []) + [f"Scriptwriter: {script_data['error']}"],
                "current_stage": "scriptwriter_failed",
                "status": "failed"
            }

        total_scenes = len(script_data.get("scenes", []))
        title = script_data.get("title", "Untitled")

        print(f"\n  [Step 3] Script generated successfully:")
        print(f"    Title: {title}")
        print(f"    Total Scenes: {total_scenes}")
        print(f"    Genre: {script_data.get('genre', genre)}")

        # Preview scenes
        for scene in script_data.get("scenes", [])[:3]:
            print(f"    Scene {scene.get('scene_number', '?')}: {scene.get('heading', 'N/A')}")

        # Step 3: Attach visual context verification
        print(f"\n  [Step 4] Verifying visual cues attached to all scenes...")
        for scene in script_data.get("scenes", []):
            if "visual_cues" not in scene or not scene["visual_cues"]:
                scene["visual_cues"] = {
                    "lighting": "Natural",
                    "camera_angle": "Medium shot",
                    "mood": "Neutral",
                    "color_palette": "Earth tones"
                }

        print(f"  ✅ {self.name} completed successfully\n")

        return {
            "raw_script": script_data,
            "scene_manifest": script_data,
            "current_stage": "script_generated",
            "status": "in_progress"
        }


def scriptwriter_node(state: WriterRoomState) -> Dict[str, Any]:
    """LangGraph node function for the Scriptwriter Agent."""
    agent = ScriptwriterAgent()
    return agent.run(state)
