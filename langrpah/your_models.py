import time
import concurrent.futures

# --- AI 모델 시뮬레이션 함수 ---

def model_1_image_to_empty_room(original_image):
    """모델 1: 원본 이미지를 받아 빈 방 이미지로 변환하는 모델"""
    print("🚀 모델 1 시작: 원본 이미지를 빈 방으로 변환 중...")
    time.sleep(5)  # 모델 추론에 5초가 걸린다고 가정
    empty_room_image = f"'{original_image}'의_빈_방_이미지.jpg"
    print("✅ 모델 1 완료!")
    return empty_room_image

def model_2_text_to_prompt(user_requirements):
    """모델 2: 사용자 요구사항 텍스트를 LLM 프롬프트로 변환하는 모델"""
    print("🚀 모델 2 시작: 요구사항을 프롬프트로 변환 중...")
    time.sleep(3)  # 모델 추론에 3초가 걸린다고 가정
    generated_prompt = f"'{user_requirements}'를 반영한 프롬프트: 'A modern, minimalist living room with oak wood floors and a large window.'"
    print("✅ 모델 2 완료!")
    return generated_prompt

def model_3_image_to_image(empty_room_image, prompt):
    """모델 3: 빈 방 이미지와 프롬프트를 받아 최종 인테리어 이미지를 생성하는 모델"""
    print("\n🚀 모델 3 시작: 빈 방에 프롬프트를 적용하여 최종 이미지 생성 중...")
    print(f"   - 입력 이미지: {empty_room_image}")
    print(f"   - 적용 프롬프트: {prompt}")
    time.sleep(7)  # 모델 추론에 7초가 걸린다고 가정
    final_image = "최종_인테리어_결과물.jpg"
    print("🎉 모델 3 완료!")
    return final_image

# --- 전체 파이프라인을 실행하는 메인 로직 ---

def run_interior_design_pipeline(original_image, user_requirements):
    """
    모델 1과 2를 병렬로 실행하고, 두 결과가 나오면 모델 3를 실행하는 파이프라인
    """
    # ThreadPoolExecutor를 사용해 병렬 작업을 관리합니다.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        print("--- 인테리어 디자인 파이프라인 시작 ---")
        
        # 1. 모델 1과 모델 2를 동시에 실행 (Fork)
        future_model_1 = executor.submit(model_1_image_to_empty_room, original_image)
        future_model_2 = executor.submit(model_2_text_to_prompt, user_requirements)
        
        # 2. 두 모델의 결과가 나올 때까지 대기 (Join)
        # .result()를 호출하면 해당 작업이 끝날 때까지 기다린 후 결과를 반환합니다.
        empty_room_image_result = future_model_1.result()
        generated_prompt_result = future_model_2.result()
        
        print("\n--- 모델 1과 2의 작업이 모두 완료되었습니다. ---")

    # 3. 모델 1과 2의 결과를 입력으로 하여 모델 3 실행
    final_result = model_3_image_to_image(empty_room_image_result, generated_prompt_result)
    
    print("\n--- ✨ 모든 파이프라인이 종료되었습니다. ---")
    print(f"최종 결과물: {final_result}")


# --- 파이프라인 실행 ---
if __name__ == "__main__":
    # 입력 데이터 정의
    my_room_image = "내_방_사진.jpg"
    my_requirements = "모던하고 미니멀한 스타일로 바꿔줘"
    
    # 실행!
    run_interior_design_pipeline(my_room_image, my_requirements)