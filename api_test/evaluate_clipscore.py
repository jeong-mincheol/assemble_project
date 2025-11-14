import torch
from PIL import Image
from clip_score import clip_score

# -----------------------------------------------------------
# ❗️ 1. 평가할 이미지와 프롬프트를 여기에 하드코딩합니다.
# -----------------------------------------------------------

# 3단계(Gemini) 결과물을 평가할 경우
IMAGE_TO_TEST = "result_3_final_with_furniture.webp"
TEXT_PROMPT_1 = """
- 스타일: 매우 모던하고 미니멀한 화이트톤
- 벽: 깨끗한 흰색
- 바닥: 밝은 회색빛이 도는 원목 마루
- 조명: 따뜻한 색의 은은한 간접 조명
- 배치된 가구: sofa.jpg, table.webp
"""

# 4단계(OpenAI) 결과물을 평가할 경우
# IMAGE_TO_TEST = "result_4_final_refined.webp"
# TEXT_PROMPT_1 = """
# - 스타일: 매우 모던하고 미니멀한 화이트톤, 흰 벽, 원목 마루, 간접 조명
# - 배치된 가구: sofa.jpg, table.webp
# - 추가 사항: 오른쪽 벽면에 심플한 원형 시계가 걸려있음
# """

# (참고) 여러 개의 프롬프트로 한 번에 테스트할 수도 있습니다.
TEXT_PROMPTS = [TEXT_PROMPT_1]

# -----------------------------------------------------------

def evaluate_image_clipscore(image_path: str, prompts: list[str]):
    """
    주어진 이미지와 텍스트 프롬프트 간의 CLIPScore를 계산합니다.
    """
    print(f"⏳ CLIPScore 평가 시작...")
    print(f"   - 이미지: {image_path}")
    print(f"   - 프롬프트: \"{prompts[0][:50]}...\"")

    try:
        # 1. 이미지 로드
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"❌ 평가 실패: 이미지 파일({image_path})을 찾을 수 없습니다.")
        print("   -> (먼저 `run_pipeline.py`를 실행해 이미지를 생성해야 합니다.)")
        return
    except Exception as e:
        print(f"❌ 평가 실패: 이미지 로드 중 오류: {e}")
        return

    # 2. CLIP 모델 로드 및 장치(GPU/CPU) 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   - (디바이스 {device}에서 CLIP 모델 로딩 중... 잠시 기다리세요)")

    try:
        # 3. 점수 계산
        # clip_score() 함수가 내부적으로 모델을 다운로드하고 이미지를 처리합니다.
        # (이미지, 텍스트 리스트) -> 점수 딕셔너리 반환
        scores = clip_score(image, prompts, device=device)
        
        # 4. 결과 출력
        # 점수는 (W)eighted (Sim)ilarity의 약자인 'wsim' 키에 저장됩니다.
        final_score = scores.get('wsim', 0.0)

        print("\n--- ✅ CLIPScore 평가 완료 ---")
        print(f"  - 이미지: {image_path}")
        print(f"  - 점수 (0~1): {final_score:.4f}")
        print("  - (1에 가까울수록 이미지와 텍스트가 잘 일치합니다)")

    except Exception as e:
        print(f"❌ CLIPScore 계산 중 심각한 오류 발생: {e}")
        print("   -> (라이브러리 설치를 확인하세요: `pip install clip-score torch transformers`)")

# =====================================
# 🚀 평가 스크립트 실행
# =====================================
if __name__ == "__main__":
    evaluate_image_clipscore(IMAGE_TO_TEST, TEXT_PROMPTS)