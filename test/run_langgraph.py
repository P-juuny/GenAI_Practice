import sys
from pathlib import Path

# 경로 설정 (test 폴더 기준)
project_root = Path(__file__).resolve().parent.parent
lang_graph_path = project_root / "src" / "lang_graph"
tools_path = project_root / "src" / "tools"

sys.path.insert(0, str(lang_graph_path))
sys.path.insert(0, str(tools_path))

from langgraph.types import Command
from graph import create_graph
from memory import extract_and_save_memory

# ===================
# 실행 함수 (Stream 사용)
# ===================

def run_agent(question: str, thread_id: str = "default", auto_approve: bool = True):
    """에이전트 실행 (Stream 모드 + Interrupt)"""
    
    graph = create_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "tool_calls": None
    }
    
    print(f"\n{'='*50}")
    print(f"질문: {question}")
    print(f"세션: {thread_id}")
    print('='*50)
    
    final_answer = None
    
    while True:
        # Stream으로 실행
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, value in event.items():
                print(f"\n[{node_name} 노드 실행됨]")
                
                if node_name == "llm":
                    msgs = value.get("messages", [])
                    for m in msgs:
                        if isinstance(m, dict) and m.get("content"):
                            print(f"LLM: {m['content'][:300]}...")
                            final_answer = m["content"]
                        if isinstance(m, dict) and m.get("tool_calls"):
                            print(f"Tool 호출 예정: {[tc['function']['name'] for tc in m['tool_calls']]}")
        
        # 현재 상태 확인
        current_state = graph.get_state(config)
        
        # END 도달하면 종료
        if not current_state.next:
            print("\n[완료]")
            if final_answer:
                extract_and_save_memory(question, final_answer)
            break
        
        # interrupt 상태 확인 (올바른 방법)
        if current_state.tasks:
            task = current_state.tasks[0]
            if hasattr(task, 'interrupts') and task.interrupts:
                interrupt_info = task.interrupts[0].value
                
                if auto_approve:
                    print(f"\n[Tool 자동 승인] {interrupt_info}")
                    initial_state = Command(resume="y")
                else:
                    print(f"\n[Interrupt] {interrupt_info}")
                    approval = input("승인? (y/n): ").strip().lower()
                    initial_state = Command(resume=approval)
                continue
        
        # interrupt 아니면 이어서 실행
        initial_state = None
    
    # 최종 답변 출력
    final_messages = current_state.values.get("messages", [])
    if final_messages:
        last_msg = final_messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        print(f"\n[최종 답변]\n{content}")
    
    return current_state

# ===================
# 간단 실행 함수 (Interrupt 없이)
# ===================

def run_simple(question: str, thread_id: str = "default"):
    """Interrupt 없이 간단하게 실행"""
    
    graph = create_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "tool_calls": None
    }
    
    print(f"\n{'='*50}")
    print(f"질문: {question}")
    print(f"세션: {thread_id}")
    print('='*50)
    
    final_answer = None
    
    # Stream으로 실행
    for event in graph.stream(initial_state, config, stream_mode="updates"):
        for node_name, value in event.items():
            print(f"[{node_name}] 실행됨")
            
            if node_name == "llm":
                msgs = value.get("messages", [])
                for m in msgs:
                    if hasattr(m, "content") and m.content:
                        final_answer = m.content
    
    # 최종 상태
    final_state = graph.get_state(config)
    final_messages = final_state.values.get("messages", [])
    
    if final_messages:
        last_msg = final_messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        print(f"\n[최종 답변]\n{content}")
        
        # 자동 메모리 저장
        extract_and_save_memory(question, content)
    
    return final_state

# ===================
# 테스트
# ===================

if __name__ == "__main__":
    session = "test_session_1"
    
    # 테스트 1: 이름 기억 (Long Term Memory 저장)
    run_agent("내 이름은 박성준이고, 생성형 AI 팀플을 하고 있어. 기억해줘.", thread_id=session, auto_approve=False)
    
    # 테스트 2: Short Term Memory (방금 대화)
    run_simple("방금 내가 뭐라고 했지?", thread_id=session)
    
    # 테스트 3: 계산 (Tool 사용)
    run_simple("123 * 456 계산해줘", thread_id=session)
    
    # 테스트 4: Long Term Memory (지난 번)
    run_simple("지난 번에 내가 말한 내 이름이 뭐였지?", thread_id=session)
    
    # 테스트 5: Interrupt 있는 버전 (수동 승인)
    # run_agent("오늘 날씨 검색해줘", thread_id=session, auto_approve=False)