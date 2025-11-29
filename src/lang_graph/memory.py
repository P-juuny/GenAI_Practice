import sys
from pathlib import Path

# 경로 설정
tools_path = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(tools_path))

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from tool_registry import register_default_tools

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
registry = register_default_tools()

# ===================
# Reflection Prompt
# ===================

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

# ===================
# 자동 메모리 저장 (Reflection)
# ===================

def extract_and_save_memory(question: str, answer: str):
    """대화가 끝나면 자동으로 장기 기억 저장 여부 판단"""
    
    snippet = f"User: {question}\nAssistant: {answer}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
            {"role": "user", "content": snippet}
        ],
        temperature=0
    )
    
    try:
        decision = json.loads(response.choices[0].message.content)
        
        if decision.get("should_write"):
            registry.call("write_memory", {
                "content": decision["content"],
                "memory_type": decision["memory_type"],
                "importance": decision["importance"],
                "tags": decision.get("tags", [])
            })
            print(f"[Memory Saved] {decision['content'][:50]}...")
    except:
        pass  # JSON 파싱 실패 시 무시