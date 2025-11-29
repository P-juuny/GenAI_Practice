import sys
from pathlib import Path

# 경로 설정
lang_graph_path = Path(__file__).resolve().parent
tools_path = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(lang_graph_path))
sys.path.insert(0, str(tools_path))

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.types import interrupt

from state import AgentState
from tool_registry import register_default_tools

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
registry = register_default_tools()

# ===================
# System Prompt
# ===================

SYSTEM_PROMPT = """
You are an AI assistant that uses tools (functions), RAG, and memory.

# High-level behavior
- Be helpful, honest, and concise.
- Answer primarily in Korean unless the user clearly wants another language.
- Think step by step internally, but do NOT expose chain-of-thought.
- When tools are available and helpful, call them instead of guessing.

# Tools available:
- get_time: Get current time in a timezone
- calculate: Basic arithmetic
- google_search: Search the web
- rag_search: Search documents in ChromaDB
- read_memory: Recall past information (장기 기억)
- write_memory: Store important information (장기 기억)

# Memory usage guidelines
- Call `read_memory` when user mentions "지난 번", "이전에", "저번에" etc.
- Call `write_memory` for user preferences, long-term goals, important facts.
- Do NOT write temporary info like today's lunch menu.

# Answer style
- Default: Korean, 친절하지만 군더더기 없이.
- 현재 대화 내역(messages)을 참고하여 "방금", "아까" 등의 질문에 답변하세요.
"""

# 위험한 tool 목록 (사람 확인 필요)
DANGEROUS_TOOLS = ["google_search", "write_memory"]

# ===================
# 메시지 변환 헬퍼
# ===================

def convert_messages(messages):
    """LangGraph 메시지 객체를 OpenAI 형식 dict로 변환"""
    converted = []
    for m in messages:
        if hasattr(m, "type"):  # HumanMessage, AIMessage 등
            role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
            role = role_map.get(m.type, m.type)
            msg_dict = {"role": role, "content": m.content or ""}
            
            # tool_calls가 있으면 추가 (AIMessage)
            if hasattr(m, "tool_calls") and m.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.get("id") or tc.get("tool_call_id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": tc.get("args", "") if isinstance(tc.get("args"), str) else json.dumps(tc.get("args", {}))
                        }
                    }
                    for tc in m.tool_calls
                ]
            
            # tool_call_id가 있으면 추가 (ToolMessage)
            if hasattr(m, "tool_call_id") and m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            
            converted.append(msg_dict)
        elif isinstance(m, dict):
            converted.append(m)
        else:
            converted.append({"role": "user", "content": str(m)})
    return converted

# ===================
# LLM Node
# ===================

def llm_node(state: AgentState) -> dict:
    """LLM 호출하는 노드"""
    
    messages = list(state["messages"])
    
    # 메시지 변환
    converted_messages = convert_messages(messages)
    
    # system prompt 확인
    has_system = converted_messages and converted_messages[0].get("role") == "system"
    if not has_system:
        converted_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + converted_messages
    
    # OpenAI Tool Spec
    tools = registry.list_openai_tools()
    
    # LLM 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=converted_messages,
        tools=tools,
        temperature=0
    )
    
    msg = response.choices[0].message
    
    # tool_calls 있는지 확인
    if msg.tool_calls:
        tool_calls_data = []
        for tc in msg.tool_calls:
            tool_calls_data.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments or "{}")
            })
        
        # OpenAI 메시지 → dict 변환
        msg_dict = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        }
        
        return {
            "messages": [msg_dict],
            "tool_calls": tool_calls_data
        }
    else:
        # OpenAI 메시지 → dict 변환
        msg_dict = {
            "role": "assistant",
            "content": msg.content or ""
        }
        
        return {
            "messages": [msg_dict],
            "tool_calls": None
        }

# ===================
# Tool Node (with Interrupt)
# ===================

def tool_node(state: AgentState) -> dict:
    """Tool 실행하는 노드 (위험한 tool은 interrupt)"""
    
    tool_calls = state["tool_calls"]
    
    if not tool_calls:
        return {"messages": [], "tool_calls": None}
    
    tool_messages = []
    
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["arguments"]
        tool_id = tc["id"]
        
        # 위험한 tool이면 사람 확인
        if tool_name in DANGEROUS_TOOLS:
            confirm = interrupt(f"'{tool_name}' 실행할까요? 인자: {tool_args}")
            
            if confirm != "y":
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps({"status": "cancelled", "reason": "사용자가 취소함"})
                })
                continue
        
        print(f"[Tool 실행] {tool_name}({tool_args})")
        
        # Tool 실행
        result = registry.call(tool_name, tool_args)
        observation = json.dumps(result, ensure_ascii=False)
        
        print(f"[Tool 결과] {observation[:200]}...")
        
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": observation
        })
    
    return {
        "messages": tool_messages,
        "tool_calls": None
    }