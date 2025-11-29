import sys
from pathlib import Path

# 경로 설정
lang_graph_path = Path(__file__).resolve().parent
tools_path = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(lang_graph_path))
sys.path.insert(0, str(tools_path))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import llm_node, tool_node

# ===================
# 라우터
# ===================

def route(state: AgentState) -> str:
    """tool_calls 있으면 tool로, 없으면 END"""
    if state.get("tool_calls"):
        return "tool"
    return END

# ===================
# 그래프 생성
# ===================

def create_graph():
    """LangGraph 에이전트 생성"""
    
    builder = StateGraph(AgentState)
    
    # 노드 추가
    builder.add_node("llm", llm_node)
    builder.add_node("tool", tool_node)
    
    # 엣지 연결
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", route)
    builder.add_edge("tool", "llm")
    
    # Checkpointer (Short Term Memory)
    memory = MemorySaver()
    
    # 컴파일
    graph = builder.compile(checkpointer=memory)
    
    return graph