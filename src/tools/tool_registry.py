from __future__ import annotations
from typing import Dict, Callable, Any, Tuple
from pydantic import ValidationError
from tool_definitions import ToolSpec, get_default_tool_specs, as_openai_tool_spec
import json


class ToolRegistry:
    """A simple registry that maps tool name -> (ToolSpec)."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register_tool(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list_openai_tools(self):
        """Return OpenAI-compatible tools[] array."""
        return [as_openai_tool_spec(spec) for spec in self._tools.values()]

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        spec = self.get(name)
        try:
            return spec.handler(args)
        except ValidationError as ve:
            # normalize pydantic errors
            return {"error": "validation_error", "details": ve.errors()}
        except Exception as e:
            return {"error": "runtime_error", "details": str(e)}
        
    def specs_for_prompt(self) -> str:
        """Return tool specs formatted for inclusion in a prompt."""
        lines = []
        for t in self._tools.values():
            schema = t.input_model.model_json_schema()["properties"]
            lines.append(f"- {t.name}: {t.description}\n  params: {json.dumps(schema, ensure_ascii=False)}")
        return "\n".join(lines)

def register_default_tools() -> ToolRegistry:  
    reg = ToolRegistry()
    for spec in get_default_tool_specs():
        reg.register_tool(spec)
    return reg