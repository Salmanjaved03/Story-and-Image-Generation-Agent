"""
Script Validator Agent
=======================
Ensures correctness of manually provided scripts.

Validation Checks:
  - Scene headers exist
  - Dialogue is labeled
  - Actions are structured

Failure Handling:
  - Reject invalid scripts with specific errors
  - Suggest corrections

MCP Tools Used:
  - validate_script_structure
"""

import json
from typing import Any, Dict

from graph.state import WriterRoomState
from mcp.server import get_mcp_registry


class ScriptValidatorAgent:
    """
    Script Validator Agent - Validates manually provided scripts.
    
    Checks structure, dialogue labels, scene headers, and
    provides actionable correction suggestions.
    """

    def __init__(self):
        self.name = "Script Validator Agent"
        self.role = "Ensure correctness of manually provided scripts"
        self.mcp = get_mcp_registry()

    def _discover_tools(self):
        """Discover validation tools via MCP registry."""
        tools = self.mcp.discover_tools(category="validation")
        print(f"  [{self.name}] Discovered {len(tools)} validation tools via MCP")
        for tool in tools:
            print(f"    → {tool['name']}: {tool['description']}")
        return tools

    def run(self, state: WriterRoomState) -> Dict[str, Any]:
        """
        Execute validation reasoning loop:
        1. Discover MCP tools
        2. Validate script structure
        3. Parse into standardized format or reject with suggestions
        """
        print(f"\n{'='*60}")
        print(f"  ✓ {self.name} - Starting Validation")
        print(f"{'='*60}")

        # Step 1: Discover tools
        self._discover_tools()

        script_text = state.get("user_input", "")
        print(f"\n  [Step 1] Received script ({len(script_text)} characters)")

        # Step 2: Validate via MCP tool
        print(f"  [Step 2] Validating structure via MCP tool: validate_script_structure")
        validation_result = self.mcp.invoke_tool(
            "validate_script_structure",
            script_text=script_text
        )

        is_valid = validation_result.get("is_valid", False)
        errors = validation_result.get("errors", [])
        warnings = validation_result.get("warnings", [])
        parsed_scenes = validation_result.get("parsed_scenes", [])

        print(f"\n  [Step 3] Validation Result:")
        print(f"    Valid: {'✅ Yes' if is_valid else '❌ No'}")
        print(f"    Scenes Found: {validation_result.get('total_scenes_found', len(parsed_scenes))}")

        if errors:
            print(f"    Errors:")
            for err in errors:
                print(f"      ✗ {err}")

        if warnings:
            print(f"    Warnings:")
            for warn in warnings:
                print(f"      ⚠ {warn}")

        if not is_valid and not parsed_scenes:
            print(f"\n  ❌ Script rejected - cannot proceed")
            return {
                "validation_result": validation_result,
                "raw_script": None,
                "errors": state.get("errors", []) + errors,
                "current_stage": "validation_failed",
                "status": "failed"
            }

        # Build standardized script structure from parsed scenes
        script_data = {
            "title": "User Provided Script",
            "genre": state.get("genre", "drama"),
            "total_scenes": len(parsed_scenes),
            "scenes": parsed_scenes,
            "source": "manual_input",
            "validation": {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings
            }
        }

        print(f"\n  ✅ {self.name} completed - Script {'accepted' if is_valid else 'accepted with warnings'}\n")

        return {
            "validation_result": validation_result,
            "raw_script": script_data,
            "scene_manifest": script_data,
            "current_stage": "script_validated",
            "status": "in_progress"
        }


def validator_node(state: WriterRoomState) -> Dict[str, Any]:
    """LangGraph node function for the Script Validator Agent."""
    agent = ScriptValidatorAgent()
    return agent.run(state)
