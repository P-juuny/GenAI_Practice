from __future__ import annotations
from typing import Any, Dict, Callable, Type, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from dateutil import tz
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
import requests
import os
import chromadb

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
collection = chromadb_client.get_collection(name="paper_rag_db")
memory_collection = chromadb_client.get_or_create_collection(name="memory_db")

# Tool Definitions
# -----------------

class GetTimeInput(BaseModel):
    timezone: str = Field(..., description="IANA timezone name, e.g., 'Asia/Seoul'")

def get_time(input: GetTimeInput) -> Dict[str, Any]:
    try:
        target_tz = tz.gettz(input.timezone)
        if target_tz is None:
            raise ValueError(f"Invalid timezone: {input.timezone}")
        now = datetime.now(tz=target_tz)
        return {
            "timezone": input.timezone,
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
        }
    except Exception as e:
        return {"error": str(e)}
    
class CalculaterInput(BaseModel):
    num1: float = Field(..., description="The first number")
    num2: float = Field(..., description="The second number")
    op: str = Field(..., description="The operation to perform: add, subtract, multiply, divide")

def calculate(input: CalculaterInput) -> Dict[str, Any]:
    try:
        if input.op == "add":
            result = input.num1 + input.num2
        elif input.op == "subtract":
            result = input.num1 - input.num2
        elif input.op == "multiply":
            result = input.num1 * input.num2
        elif input.op == "divide":
            if input.num2 == 0:
                raise ValueError("Division by zero is not allowed.")
            result = input.num1 / input.num2
        else:
            raise ValueError(f"Invalid operation: {input.op}")
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
    
class GoogleSearchInput(BaseModel):
    query: str = Field(..., description="The search query string")
    num_results: int = Field(5, ge=1, le=10, description="Number of search results to return")

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be empty")
        return v

def google_search(input: GoogleSearchInput) -> Dict[str, Any]:
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        cx = os.getenv("GOOGLE_CX")
        if not api_key or not cx:
            raise EnvironmentError("Google API key and CX must be set in environment variables.")
        
        url = "https://www.googleapis.com/customsearch/v1"

        params = {
            "key": api_key,
            "cx": cx,
            "q": input.query,
            "num": input.num_results
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
        
        return {"results": results, "source": "google_cse"}
    except Exception as e:
        return {"error": str(e)}
    
# Have to change after RAG finished + Add Reranking module
class RAGSearchInput(BaseModel):
    query: str = Field(..., description="The search query string")
    n_results: int = Field(5, ge=1, le=20, description="Number of search results to return")

def rag_search(input: RAGSearchInput) -> Dict[str, Any]:
    try:
        results = collection.query(
            query_texts=[input.query],
            n_results=input.n_results,
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return {"results": [], "total": 0}
        
        # 2. 리랭킹 (쿼리 + 문서 pair)
        pairs = [[input.query, doc] for doc in documents]
        scores = reranker.predict(pairs)
        
        # 3. 점수순 정렬 후 top_k개만
        ranked = sorted(
            zip(documents, metadatas, scores),
            key=lambda x: x[2],
            reverse=True
        )[:input.n_results]
        
        # 4. 결과 포맷팅
        results_list = []
        for i, (doc, metadata, score) in enumerate(ranked):
            results_list.append({
                "rank": i + 1,
                "content": doc,
                "metadata": metadata if metadata else {},
                "score": round(float(score), 4)
            })
        
        return {"results": results_list, "source": "chroma_rag"}
    except Exception as e:
        return {"error": str(e)}
    
class ReadMemoryInput(BaseModel):
    query: str = Field(..., description="검색할 키워드나 질문")
    memory_type: str = Field("all", description="메모리 타입: 'all', 'profile', 'episodic', 'knowledge'")
    top_k: int = Field(5, ge=1, le=10, description="반환할 결과 수")

def read_memory(input: ReadMemoryInput) -> Dict[str, Any]:
    try:
        # 검색 조건 설정
        where_filter = None
        if input.memory_type != "all":
            where_filter = {"memory_type": input.memory_type}
        
        results = memory_collection.query(
            query_texts=[input.query],
            n_results=input.top_k,
            where=where_filter
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return {"results": [], "message": "관련 기억을 찾지 못했습니다."}
        
        results_list = []
        for doc, metadata in zip(documents, metadatas):
            results_list.append({
                "content": doc,
                "memory_type": metadata.get("memory_type", "unknown"),
                "importance": metadata.get("importance", 0),
                "tags": metadata.get("tags", []),
                "created_at": metadata.get("created_at", "unknown")
            })
        
        return {"results": results_list, "count": len(results_list)}
    except Exception as e:
        return {"error": str(e)}
    
class WriteMemoryInput(BaseModel):
    content: str = Field(..., description="저장할 내용")
    memory_type: str = Field("episodic", description="메모리 타입: 'profile', 'episodic', 'knowledge'")
    importance: int = Field(3, ge=1, le=5, description="중요도 1(낮음) ~ 5(높음)")
    tags: List[str] = Field(default=[], description="태그 목록")

def write_memory(input: WriteMemoryInput) -> Dict[str, Any]:
    try:
        from datetime import datetime
        
        # 고유 ID 생성
        memory_id = f"mem_{datetime.now().timestamp()}"
        
        # 메타데이터
        metadata = {
            "memory_type": input.memory_type,
            "importance": input.importance,
            "tags": ",".join(input.tags),  # ChromaDB는 list 지원 안 함
            "created_at": datetime.now().isoformat()
        }
        
        # 저장
        memory_collection.add(
            documents=[input.content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        return {
            "status": "saved",
            "memory_id": memory_id,
            "content": input.content,
            "memory_type": input.memory_type
        }
    except Exception as e:
        return {"error": str(e)}
    
def cleanup_memories(max_count: int = 500):
    """오래되고 중요도 낮은 메모리 정리"""
    
    all_data = memory_collection.get()
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    
    if len(ids) <= max_count:
        return  # 정리 필요 없음
    
    # (id, importance, created_at) 리스트 만들기
    memory_info = []
    for id, meta in zip(ids, metadatas):
        memory_info.append({
            "id": id,
            "importance": meta.get("importance", 3),
            "created_at": meta.get("created_at", "")
        })
    
    # 중요도 낮고 오래된 순으로 정렬
    memory_info.sort(key=lambda x: (x["importance"], x["created_at"]))
    
    # 초과분 삭제
    to_delete = len(ids) - max_count
    delete_ids = [m["id"] for m in memory_info[:to_delete]]
    
    memory_collection.delete(ids=delete_ids)
    print(f"[Memory Cleanup] {len(delete_ids)}개 삭제됨")
    

# Tool Spec
# -----------------

class ToolSpec(BaseModel):
    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Callable[[Any], Dict[str, Any]]

def as_openai_tool_spec(spec: ToolSpec) -> Dict[str, Any]:
    """Return OpenAI tools[] spec for function calling (JSON Schema)."""
    schema = spec.input_model.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": schema,
        },
    }

def get_default_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="get_time",
            description="Get the current time in a specified timezone.",
            input_model=GetTimeInput,
            handler=lambda args: get_time(GetTimeInput(**args)),
        ),
        ToolSpec(
            name="calculate",
            description="Perform basic arithmetic operations between two numbers.",
            input_model=CalculaterInput,
            handler=lambda args: calculate(CalculaterInput(**args)),
        ),
        ToolSpec(
            name="google_search",
            description="Perform a Google search and return the top results.",
            input_model=GoogleSearchInput,
            handler=lambda args: google_search(GoogleSearchInput(**args)),
        ),
        ToolSpec(
            name="rag_search",
            description="Search documents from ChromaDB with reranking. Use for questions about papers, research, or stored documents.",
            input_model=RAGSearchInput,
            handler=lambda args: rag_search(RAGSearchInput(**args)),
        ),
        ToolSpec(
            name="read_memory",
            description="과거 대화나 사용자 정보를 검색합니다. '지난 번', '이전에 말했듯이' 등 과거 내용 언급 시 사용.",
            input_model=ReadMemoryInput,
            handler=lambda args: read_memory(ReadMemoryInput(**args)),
        ),
        ToolSpec(
            name="write_memory",
            description="중요한 정보를 메모리에 저장합니다. 사용자 선호, 장기 목표, 프로젝트 정보 등.",
            input_model=WriteMemoryInput,
            handler=lambda args: write_memory(WriteMemoryInput(**args)),
        ),
    ]