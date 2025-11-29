import os
import json
import random
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from tool_registry import register_default_tools
from tool_definitions import cleanup_memories

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 툴 로드
registry = register_default_tools()

# ===================
# 데이터 모델
# ===================

@dataclass
class Trace:
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None

@dataclass
class Trajectory:
    question: str
    traces: List[Trace] = field(default_factory=list)
    final_answer: Optional[str] = None

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
# Tools and ReAct-style behavior
- You may call tools such as:
- read_memory: to recall important past information about the user or past sessions.
- write_memory: to store new, useful information about the user or this conversation.
- retrieve_docs or other RAG tools: to look up information in external knowledge bases.
- Use tools when:
- You lack required factual details.
- You need to recall prior user preferences, past discussions, or long-term context.
- You need domain knowledge stored in a vector database or document store.
- After receiving a tool result, incorporate it into your reasoning and produce a final answer.
Copyright 2025. Korea Aerospace University. All rights reserved.
# Memory usage guidelines
- Memory is not magic; you must explicitly call `read_memory` or `write_memory` to use it.
- Call `read_memory` when:
- The user refers to “지난 번”, “이전에 말했듯이”, “저번에 만들던 코드” 등 과거 내용.
- The answer clearly depends on the user’s preferences, profile, or long-term history.
- Call `write_memory` when:
- The user shares stable personal preferences (e.g., 좋아하는 스타일, 선호 옵션).
- The user states long-term goals, ongoing projects, or recurring topics.
- The user corrects you or provides important facts that will be useful later.
- Do NOT write memory for:
- Short-lived, one-off facts (예: 오늘 점심 메뉴).
- Extremely detailed logs that are unlikely to be reused.
- Sensitive personal data, unless the user explicitly requests you to remember it.
# RAG usage guidelines
- Call retrieval tools (예: retrieve_docs) when:
- The user asks for factual information that may be in an external KB.
- You need more detailed or authoritative content (e.g., 긴 기술 설명, 강의자료).
- When you get retrieved documents, read them and synthesize a clear, concise answer.
# Answer style
- Default: Korean, 친절하지만 군더더기 없이.
- Provide structure (번호, 소제목) for teaching/explaining technical concepts.
- If the user is building a system or code, show step-by-step reasoning in high level,
but do NOT output low-level hidden chain-of-thought or internal scratch work.
# Safety
- If a user asks you to perform unsafe, illegal,
or harmful actions, politely refuse.
- If you’re unsure, say so and explain what
additional information would be needed.

- Examples for `read_memory`:
- User: "지난 번에 설명해 준 HNSW ef 파라미터 다시 정리해 주세요."
-> Call read_memory with query like "HNSW ef parameter explanation last session".
- User: "우리가 전에 만들던 RAG 코드 이어서 해볼까요?"
-> Call read_memory with query describing "previous RAG code we wrote".
- Examples for `write_memory`:
- User: "앞으로 나를 부를 때는 '교수님'이라고 불러 주세요."
-> Call write_memory with memory_type="profile", tags=["name_preference"].
- User: "내 장기 목표는 'Agentic AI' 강의 전체 커리큘럼을 완성하는 것입니다."
-> Call write_memory with memory_type="profile" or "episodic",
tags=["long_term_goal", "agentic_ai_course"], importance=4 or 5.
"""



MEMORY_EXTRACTOR_PROMPT = """
다음 대화를 보고 장기 기억으로 저장할 가치가 있는 정보가 있는지 판단하세요.

저장해야 할 것:
- 사용자 선호 (이름, 스타일)
- 장기 목표, 프로젝트
- 나중에 유용할 정보

저장하면 안 되는 것:
- 일회성 정보 (오늘 점심)
- 너무 상세한 로그

JSON으로 응답:
{
    "should_write": true/false,
    "memory_type": "profile" | "episodic" | "knowledge",
    "importance": 1~5,
    "content": "저장할 내용",
    "tags": ["태그1", "태그2"]
}

저장할 거 없으면:
{"should_write": false}
"""

def extract_and_save_memory(question: str, answer: str):
    snippet = f"User: {question}\nAssistant: {answer}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
            {"role": "user", "content": snippet}
        ],
        temperature=0
    )
    
    decision = json.loads(response.choices[0].message.content)
    
    if decision.get("should_write"):
        registry.call("write_memory", {
            "content": decision["content"],
            "memory_type": decision["memory_type"],
            "importance": decision["importance"],
            "tags": decision.get("tags", [])
        })
        print(f"[Memory Saved] {decision['content'][:50]}...")
    

# ===================
# Agent Loop (Tool Calling)
# ===================
def run_react_agent(question: str, max_cycles: int = 6, verbose: bool = True) -> Trajectory:
    traj = Trajectory(question=question)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    # OpenAI Tool Spec 가져오기
    tools = registry.list_openai_tools()
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print('='*50)
    
    for cycle in range(max_cycles):
        if verbose:
            print(f"\n--- Cycle {cycle + 1} ---")
        
        # 1. LLM 호출 (tools 포함)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0
        )
        
        msg = response.choices[0].message
        tool_calls = msg.tool_calls
        
        # 2. Tool Call 없으면 → 최종 답변
        if not tool_calls:
            traj.final_answer = msg.content or ""
            if verbose:
                print(f"\n[Final Answer]\n{traj.final_answer}")

            extract_and_save_memory(question, traj.final_answer)

            if random.randint(1, 30) == 1:
                cleanup_memories()

            return traj
        
        # 3. assistant 메시지 추가 (tool_calls 포함)
        messages.append(msg)
        
        # 4. 각 Tool Call 실행
        for tc in tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments or "{}")
            
            if verbose:
                print(f"Tool Call: {tool_name}({tool_args})")
            
            # Tool 실행 (registry.call 사용)
            result = registry.call(tool_name, tool_args)
            observation = json.dumps(result, ensure_ascii=False)
            
            if verbose:
                print(f"Observation: {observation[:300]}...")
            
            # Trace 저장
            traj.traces.append(Trace(
                tool_name=tool_name,
                tool_args=tool_args,
                observation=observation
            ))
            
            # 5. Tool 결과를 messages에 추가
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": observation
            })
    
    traj.final_answer = "최대 단계 초과"
    return traj

# ===================
# 실행
# ===================

if __name__ == "__main__":
    run_react_agent("요즘 generative 논문중에서 attention is all you need랑 비슷한 논문 뭐가 있음?")

    run_react_agent("내 이름은 박성준이고, 생성형 AI 팀플을 하고 있어. 기억해줘.")

    run_react_agent("지난 번에 내가 말한 내 이름이 뭐였지?")

    run_react_agent("나는 Python이랑 PyTorch 주로 써.")

    run_react_agent("우리가 전에 만들던 RAG 코드 이어서 해볼까요?")