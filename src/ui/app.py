import sys
from pathlib import Path
import uuid

project_root = Path(__file__).resolve().parent.parent.parent
lang_graph_path = project_root / "src" / "lang_graph"
tools_path = project_root / "src" / "tools"

sys.path.insert(0, str(lang_graph_path))
sys.path.insert(0, str(tools_path))

import gradio as gr
from langgraph.types import Command
from graph import create_graph
from memory import extract_and_save_memory

# 그래프 전역
graph = create_graph()

# 세션별 thread_id 저장
session_threads = {}

def chat(message: str, history: list, request: gr.Request):
    """Gradio 채팅 함수"""
    
    # 세션 ID 가져오기 (Gradio가 자동 생성)
    session_id = request.session_hash
    
    # 세션별 thread_id 생성/조회
    if session_id not in session_threads:
        session_threads[session_id] = str(uuid.uuid4())
    
    thread_id = session_threads[session_id]
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"[SESSION] {session_id} -> thread: {thread_id}")
    
    initial_state = {
        "messages": [{"role": "user", "content": message}],
        "tool_calls": None
    }
    
    final_answer = ""
    
    yield "🤔 생각 중..."
    
    try:
        while True:
            for event in graph.stream(initial_state, config, stream_mode="updates"):
                for node_name, value in event.items():
                    if node_name == "llm":
                        msgs = value.get("messages", [])
                        for m in msgs:
                            if isinstance(m, dict) and m.get("tool_calls"):
                                tool_names = [tc['function']['name'] for tc in m['tool_calls']]
                                yield f"🔧 도구 사용 중: {', '.join(tool_names)}"
                            if isinstance(m, dict) and m.get("content"):
                                final_answer = m["content"]
                                yield final_answer
                    
                    elif node_name == "tool":
                        yield "⚙️ 도구 실행 완료"
            
            current_state = graph.get_state(config)
            
            if not current_state.next:
                break
            
            if current_state.tasks:
                task = current_state.tasks[0]
                if hasattr(task, 'interrupts') and task.interrupts:
                    print(f"[AUTO APPROVE] {task.interrupts[0].value}")
                    initial_state = Command(resume="y")
                    continue
            
            initial_state = None
        
        if not final_answer:
            final_state = graph.get_state(config)
            final_messages = final_state.values.get("messages", [])
            if final_messages:
                last_msg = final_messages[-1]
                if hasattr(last_msg, "content"):
                    final_answer = last_msg.content
                elif isinstance(last_msg, dict):
                    final_answer = last_msg.get("content", "")
                if final_answer:
                    yield final_answer
        
        if not final_answer:
            yield "❌ 응답을 받지 못했습니다."
        else:
            extract_and_save_memory(message, final_answer)
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        yield f"❌ 오류 발생: {str(e)}"

# Gradio UI
demo = gr.ChatInterface(
    fn=chat,
    title="📚 논문 검색 AI",
    description="Generative AI 관련 논문을 검색하고 질문에 답변해드립니다.",
)

if __name__ == "__main__":
    demo.launch()