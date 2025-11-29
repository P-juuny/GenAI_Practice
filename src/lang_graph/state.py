from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# ===================
# State 정의 (Short Term Memory)
# ===================

class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]  # 대화 히스토리 자동 누적
    tool_calls: List[dict] | None  # tool 호출 정보