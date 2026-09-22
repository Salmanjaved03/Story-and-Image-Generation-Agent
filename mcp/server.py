"""
MCP Server - Model Context Protocol Tool Registry
===================================================
Implements a local MCP tool registry that allows agents to discover
and invoke tools dynamically at runtime. No hardcoded API calls.

All tools are registered with:
  - name: unique tool identifier
  - description: what the tool does
  - input_schema: expected input parameters
  - output_schema: expected output format
  - handler: the actual function to execute
"""

import json
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """Schema definition for an MCP-registered tool."""
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="What this tool does")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input parameter schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output format schema")
    category: str = Field(default="general", description="Tool category for discovery")


class MCPToolRegistry:
    """
    MCP Tool Registry - Central hub for dynamic tool discovery.
    
    Agents query this registry at runtime to discover available tools.
    Tools are invoked via structured schemas, never hardcoded.
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        category: str = "general"
    ) -> None:
        """Register a new tool in the MCP registry."""
        schema = ToolSchema(
            name=name,
            description=description,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            category=category
        )
        self._tools[name] = schema.model_dump()
        self._handlers[name] = handler
        print(f"  [MCP] Registered tool: {name} ({category})")

    def discover_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Discover available tools, optionally filtered by category.
        This is the primary MCP discovery endpoint agents use.
        """
        if category:
            return [
                tool for tool in self._tools.values()
                if tool["category"] == category
            ]
        return list(self._tools.values())

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the full schema for a specific tool."""
        return self._tools.get(tool_name)

    def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Invoke a tool by name with the given parameters.
        Tools are always invoked through this interface, never directly.
        """
        if tool_name not in self._handlers:
            raise ValueError(
                f"Tool '{tool_name}' not found in MCP registry. "
                f"Available tools: {list(self._tools.keys())}"
            )

        handler = self._handlers[tool_name]
        print(f"  [MCP] Invoking tool: {tool_name}")
        try:
            result = handler(**kwargs)
            return result
        except Exception as e:
            return {"error": str(e), "tool": tool_name}

    def list_tool_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tools_summary(self) -> str:
        """Get a formatted summary of all available tools for agent prompts."""
        lines = ["Available MCP Tools:"]
        for name, schema in self._tools.items():
            lines.append(f"  - {name}: {schema['description']}")
            if schema['input_schema']:
                params = ", ".join(schema['input_schema'].get('properties', {}).keys())
                lines.append(f"    Parameters: {params}")
        return "\n".join(lines)


# ─── Global MCP Registry Singleton ──────────────────────────────────────────
_global_registry: Optional[MCPToolRegistry] = None


def get_mcp_registry() -> MCPToolRegistry:
    """Get or create the global MCP tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MCPToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None
