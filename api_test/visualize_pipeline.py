import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from PIL import Image

# -----------------------------------------------
# 1. 상태 정의 (파이프라인을 따라 이동하는 데이터)
# -----------------------------------------------
class PipelineState(TypedDict):
    original_path: str
    cached_empty_path: str
    step3_cache_path: str
    step4_cache_path: str
    
    style_prompt: str
    furniture_paths: List[str]
    refinement_prompt: str
    
    run_only_step_4: bool # 4단계만 실행할지 여부
    
    # 중간 결과물 상태
    step1_image: Image.Image | None 
    step3_image: Image.Image | None

# -----------------------------------------------
# 2. 노드(Node) 정의 (빈 함수)
# -----------------------------------------------

# ❗️❗️❗️ 1. 시작점 라우터 노드 추가 ❗️❗️❗️
def entry_point_router(state: PipelineState) -> PipelineState:
    """
    (노드) 실행 모드를 결정하기 위한 진입점입니다.
    (실제 작업은 하지 않고 상태만 통과시킵니다.)
    """
    print("➡️ (노드) 실행 모드 결정 중...")
    return state

def check_step1_cache(state: PipelineState) -> PipelineState:
    print("➡️ (노드) 1단계 캐시 확인")
    if os.path.exists(state["cached_empty_path"]):
        print("   -> 1단계 캐시 찾음")
        state["step1_image"] = Image.new("RGB", (1,1)) # (임시) 찾았다고 표시
    else:
        print("   -> 1단계 캐시 없음")
        state["step1_image"] = None
    return state

def run_step1_openai(state: PipelineState) -> PipelineState:
    print("➡️ (노드) 1단계 실행: OpenAI 가구 제거")
    state["step1_image"] = Image.new("RGB", (1,1))
    return state

def check_step3_cache(state: PipelineState) -> PipelineState:
    print("➡️ (노드) 3단계 캐시 확인")
    if os.path.exists(state["step3_cache_path"]):
        print("   -> 3단계 캐시 찾음")
        state["step3_image"] = Image.new("RGB", (1,1))
    else:
        print("   -> 3단계 캐시 없음")
        state["step3_image"] = None
    return state

def run_step3_gemini(state: PipelineState) -> PipelineState:
    print("➡️ (노드) 3단계 실행: Gemini 스타일+가구 배치")
    state["step3_image"] = Image.new("RGB", (1,1))
    return state

def check_step4_prerequisite(state: PipelineState) -> PipelineState:
    """(4단계 전용) 3단계 캐시 확인 노드 (check_step3_cache와 동일한 함수 사용)"""
    print("➡️ (노드) 4단계 전제조건 (3단계 캐시) 확인")
    if os.path.exists(state["step3_cache_path"]):
        print("   -> 3단계 캐시 찾음")
        state["step3_image"] = Image.new("RGB", (1,1))
    else:
        print("   -> 3단계 캐시 없음")
        state["step3_image"] = None
    return state

def run_step4_openai(state: PipelineState) -> PipelineState:
    print("➡️ (노드) 4단계 실행: OpenAI 부분 수정")
    return state

def handle_error(state: PipelineState) -> PipelineState:
    print("➡️ (노드) 오류 발생 및 중지")
    return state

# -----------------------------------------------
# 3. 엣지(Edge) 정의 (분기 로직)
# -----------------------------------------------

def should_run_only_step4(state: PipelineState) -> str:
    """(시작점) 4단계만 실행할지, 전체 파이프라인을 실행할지 결정"""
    if state["run_only_step_4"]:
        print("   -> (분기) 4단계 '수정' 모드")
        return "run_step4_only"
    else:
        print("   -> (분기) '1~3단계 생성' 모드")
        return "run_full_pipeline"

def should_run_step1(state: PipelineState) -> str:
    """1단계 캐시를 확인하고, 1단계를 실행할지 건너뛸지 결정"""
    if state["step1_image"]:
        print("   -> (분기) 1단계 건너뛰기")
        return "skip_step1"
    else:
        print("   -> (분기) 1단계 실행")
        return "run_step1"

def should_run_step3(state: PipelineState) -> str:
    """1단계가 끝난 후, 3단계 캐시를 확인하고 실행할지 결정 (4단계로 안 감)"""
    if state["step1_image"] is None:
        print("   -> (분기) 1단계 실패로 중지")
        return "error"
        
    if state["step3_image"]:
        print("   -> (분기) 3단계 캐시 찾음 (종료)")
        return "end_pipeline"
    else:
        print("   -> (분기) 3단계 실행")
        return "run_step3"

def check_step3_result(state: PipelineState) -> str:
    """3단계가 끝난 후, 성공/실패 여부만 판단 (4단계로 안 감)"""
    if state["step3_image"] is None:
        print("   -> (분기) 3단계 실패로 중지")
        return "error"
    else:
        print("   -> (분기) 3단계 성공으로 종료")
        return "end_pipeline"

def check_step4_prerequisite_logic(state: PipelineState) -> str:
    """4단계 '수정' 모드 진입 시, 3단계 캐시가 있는지 확인"""
    if state["step3_image"]:
        print("   -> (분기) 4단계 실행")
        return "run_step4"
    else:
        print("   -> (분기) 4단계 전제조건 실패로 중지")
        return "error"

# -----------------------------------------------
# 4. 그래프(Graph) 생성 (❗️❗️❗️ 이 부분이 수정되었습니다 ❗️❗️❗️)
# -----------------------------------------------
print("--- 📊 LangGraph 파이프라인 그래프 생성 시작 ---")

builder = StateGraph(PipelineState)

# 1. 노드 추가
builder.add_node("entry_point_router", entry_point_router) # ❗️ 2. 시작 라우터 노드 추가
builder.add_node("check_step1_cache", check_step1_cache)
builder.add_node("run_step1_openai", run_step1_openai)
builder.add_node("check_step3_cache", check_step3_cache)
builder.add_node("run_step3_gemini", run_step3_gemini)
builder.add_node("check_step4_prerequisite", check_step4_prerequisite) # 4단계 전용 캐시 확인 노드
builder.add_node("run_step4_openai", run_step4_openai)
builder.add_node("handle_error", handle_error)

# 2. 엣지(흐름) 연결
# 시작점
builder.set_entry_point("entry_point_router") # ❗️ 3. 시작점 변경

# 메인 분기 (4단계만? or 1~3단계?)
builder.add_conditional_edges(
    "entry_point_router", # ❗️ 4. 출발지 변경
    should_run_only_step4,
    {
        "run_full_pipeline": "check_step1_cache",      # '1~3단계 생성' 모드
        "run_step4_only": "check_step4_prerequisite" # '4단계 수정' 모드
    }
)

# --- '1~3단계 생성' 파이프라인 로직 ---
builder.add_conditional_edges(
    "check_step1_cache", should_run_step1,
    {"skip_step1": "check_step3_cache", "run_step1": "run_step1_openai"}
)
builder.add_edge("run_step1_openai", "check_step3_cache")

builder.add_conditional_edges(
    "check_step3_cache", should_run_step3,
    {"end_pipeline": END, "run_step3": "run_step3_gemini", "error": "handle_error"}
)
builder.add_conditional_edges(
    "run_step3_gemini", check_step3_result,
    {"end_pipeline": END, "error": "handle_error"}
)

# --- '4단계 수정' 파이프라인 로직 ---
builder.add_conditional_edges(
    "check_step4_prerequisite", # 3단계 캐시 확인 후
    check_step4_prerequisite_logic,
    {
        "run_step4": "run_step4_openai",
        "error": "handle_error"
    }
)

# 종료점
builder.add_edge("run_step4_openai", END)
builder.add_edge("handle_error", END)

# -----------------------------------------------
# 5. 그래프 컴파일 및 시각화
# -----------------------------------------------
app = builder.compile()

try:
    # 그래프를 PNG 파일로 저장
    img_data = app.get_graph().draw_png()
    with open("pipeline_flow.png", "wb") as f:
        f.write(img_data)
    print("\n--- ✅ 파이프라인 시각화 완료! 'pipeline_flow.png' 파일을 확인하세요. ---")
    
    # 

except ImportError:
    print("\n--- ❗️ 시각화 실패 ---")
    print("PNG 파일을 생성하려면 'pygraphviz' 라이브러리가 필요합니다.")
    print("터미널에서 `pip install pygraphviz` 또는 `pip install langgraph[draw]`를 실행하세요.")
except Exception as e:
    print(f"\n--- ❗️ 시각화 중 오류 발생: {e} ---")