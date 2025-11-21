import base64
import io
import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import google.generativeai as genai
from PIL import Image

from interior.services.catalog_poc_injector import inject_with_gemini
from interior.services.catalog_poc_furniture_repository import FurnitureItem

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_MIMETYPES = ("image/jpeg", "image/png", "image/webp")

DEFAULT_EMPTY_ROOM_PROMPT = """
# Your Mission
- Remove all furniture, decorations, and objects from the image, EXCEPT for the elements listed under 'Elements to Keep'.
# Elements to Keep (DO NOT CHANGE):
- The entire structure of the room's walls, including columns, corners, ceiling, and floor shape.
- The original design of window frames and doors.
- The original material and texture of the walls and floor.
- The EXACT camera angle, perspective, and viewpoint of the original photo.
- The lighting conditions and overall atmosphere.
# Actions to AVOID (DO NOT DO):
- Do not demolish or create new walls.
- Do not change the size or shape of the windows.
- Do not alter the room's layout or structure in any way.
- CRITICAL: Do not change the camera angle, perspective, or composition of the photo.
""".strip()


class PipelineStepError(RuntimeError):
    """Raised when a pipeline step fails."""


def _ensure_source_exists(path: str) -> None:
    if not os.path.exists(path):
        raise PipelineStepError(f"이미지 파일을 찾을 수 없습니다: {path}")


def _detect_mimetype(path: str) -> str:
    mimetype, _ = mimetypes.guess_type(path)
    if mimetype not in SUPPORTED_IMAGE_MIMETYPES:
        raise PipelineStepError(
            f"지원하지 않는 이미지 형식입니다: {mimetype or 'unknown'} "
            f"(지원 형식: {', '.join(SUPPORTED_IMAGE_MIMETYPES)})"
        )
    return mimetype


def _decode_image_bytes(encoded: str) -> bytes:
    try:
        return base64.b64decode(encoded)
    except (TypeError, ValueError) as exc:
        raise PipelineStepError("이미지 데이터를 디코딩하지 못했습니다.") from exc


def generate_empty_room(
    original_image_path: str,
    openai_client,
    *,
    prompt: Optional[str] = None,
    size: str = "1024x1024",
) -> Image.Image:
    """Generate an empty room image from the original input."""
    logger.info("1단계 시작: 빈 방 이미지 생성 (%s)", original_image_path)
    _ensure_source_exists(original_image_path)
    mimetype = _detect_mimetype(original_image_path)

    with open(original_image_path, "rb") as img_file:
        image_data = img_file.read()

    payload_prompt = prompt or DEFAULT_EMPTY_ROOM_PROMPT

    try:
        response = openai_client.images.edit(
            model="gpt-image-1",
            image=(os.path.basename(original_image_path), image_data, mimetype),
            prompt=payload_prompt,
            size=size,
        )
    except Exception as exc:
        raise PipelineStepError("OpenAI 이미지 편집 API 호출에 실패했습니다.") from exc

    try:
        encoded = response.data[0].b64_json
    except (AttributeError, IndexError, KeyError) as exc:
        raise PipelineStepError("OpenAI 응답에 이미지 데이터가 포함되어 있지 않습니다.") from exc

    image_bytes = _decode_image_bytes(encoded)
    result = Image.open(io.BytesIO(image_bytes))
    result.load()
    logger.info("1단계 완료: 빈 방 이미지 생성 성공")
    return result


def step3_add_local_furniture(
    empty_room_image: Image.Image,
    style_prompt: str,
    furniture_paths: Optional[Sequence[str]],
    *,
    generative_model,
    upload_file_func: Optional[Callable[[io.BytesIO, str], object]] = None,
    timeout: int = 180,
    variation_index: int = 0,
) -> Image.Image:
    """Apply style and furniture to the empty room using Gemini."""
    if generative_model is None:
        raise PipelineStepError("Gemini 모델 인스턴스가 필요합니다.")

    upload_func = upload_file_func or genai.upload_file

    # 다양한 가구 조합 및 배치를 위한 variation 설정
    variation_instructions = [
        "다양한 색상의 가구들을 사용하여 활기찬 분위기를 연출하세요. 가구 배치는 대칭적으로 정돈된 느낌으로 하세요.",
        "차분하고 모던한 색상(회색, 베이지, 화이트)의 가구를 선택하고, 비대칭 배치로 역동적인 느낌을 주세요.",
        "우드 톤의 자연스러운 가구들을 중심으로 배치하고, 식물과 같은 자연 요소를 추가하세요.",
        "밝은 파스텔 톤의 가구들을 사용하여 부드럽고 따뜻한 분위기를 연출하세요. 소품도 다양하게 배치하세요.",
        "대담한 악센트 컬러(네이비, 그린, 버건디 등)의 가구를 포인트로 사용하고, 나머지는 중성 색상으로 조화롭게 배치하세요.",
        "미니멀한 디자인의 가구를 최소한으로 배치하고, 여백을 살려 공간감을 강조하세요.",
        "다양한 텍스처(벨벳, 리넨, 가죽)의 가구들을 혼합하여 풍부한 질감을 표현하세요.",
        "빈티지 또는 레트로 스타일의 가구들을 선택하고, 독특한 소품들로 개성을 더하세요.",
        "기능적이면서도 스타일리시한 수납가구들을 포함하여 실용적인 공간을 연출하세요. 다양한 형태와 크기의 가구를 조합하세요.",
    ]

    variation_guide = variation_instructions[variation_index % len(variation_instructions)]

    logger.info("3단계 시작: 스타일 적용 및 가구 배치")

    try:
        base_stream = io.BytesIO()
        empty_room_image.save(base_stream, format="WEBP")
        base_stream.seek(0)
        base_room_file = upload_func(base_stream, mime_type="image/webp")
    except Exception as exc:
        raise PipelineStepError("빈 방 이미지를 Gemini에 업로드하지 못했습니다.") from exc

    furniture_files: List[object] = []
    uploaded_labels: List[str] = []
    for path in furniture_paths or []:
        try:
            _ensure_source_exists(path)
        except PipelineStepError as missing_exc:
            logger.warning("가구 이미지가 존재하지 않아 건너뜀: %s", path)
            logger.debug("세부 정보: %s", missing_exc)
            continue

        mimetype = mimetypes.guess_type(path)[0]
        if mimetype not in SUPPORTED_IMAGE_MIMETYPES:
            logger.warning("지원하지 않는 가구 이미지 형식(%s), PNG로 처리합니다. (%s)", mimetype, path)
            mimetype = "image/png"

        try:
            with open(path, "rb") as furniture_file:
                file_bytes = furniture_file.read()
            furniture_stream = io.BytesIO(file_bytes)
            uploaded = upload_func(furniture_stream, mime_type=mimetype)
            furniture_files.append(uploaded)
            uploaded_labels.append(os.path.splitext(os.path.basename(path))[0])
            logger.debug("가구 이미지 업로드 완료: %s", path)
        except Exception as exc:
            logger.warning("가구 이미지 업로드 실패(%s): %s", path, exc)

    furniture_available = bool(furniture_files)
    if furniture_paths and not furniture_available:
        logger.warning("가구 경로가 제공되었지만 업로드가 모두 실패했습니다. 프롬프트에서 가구 생성을 허용합니다.")

    if furniture_available:
        furniture_instruction = (
            "# 2. 배치할 가구 (필수):\n"
            f"- 반드시 다음 참조 이미지를 활용해 동일한 디자인을 재현하세요: {', '.join(uploaded_labels)}\n"
            "- 형태·색상을 유지한 채 방의 원근과 조명에 맞도록만 보정하세요.\n"
        )
    else:
        furniture_instruction = (
            "# 2. 배치할 가구 (없음):\n"
            "- 제공된 참조 이미지가 없으므로 스타일 프롬프트에 맞는 가구를 직접 생성하세요.\n"
        )

    prompt = f"""
    당신은 AI 인테리어 디자이너입니다.
    '빈 방' 이미지(입력 1)를 베이스로, '가구' 이미지(입력 2...)들을 배치하세요.

    # ⚠️ CRITICAL CONSTRAINTS (절대 지켜야 할 규칙):
    - 빈 방 이미지의 카메라 앵글, 시점, 구도를 **절대 변경하지 마세요**.
    - 방의 구조(벽, 창문, 천장, 바닥)를 **절대 변경하지 마세요**.
    - 원본 사진의 원근감과 시야각을 **정확히 유지**하세요.
    - 단지 빈 공간에 가구를 **추가**하는 것만 하세요 (구조 변경 금지).

    # 1. 적용할 스타일 (필수):
    {style_prompt}

    {furniture_instruction}

    # 3. 이번 디자인의 특별 지침 (가구 선택 및 배치에만 적용):
    {variation_guide}

    # 4. 가구 배치 규칙:
    - 빈 방 이미지의 원근감과 시점을 분석하여, 그에 맞는 자연스러운 위치에 가구를 배치하세요.
    - 가구는 바닥에 안정적으로 놓여있어야 하며, 벽이나 다른 가구와 자연스럽게 조화를 이뤄야 합니다.
    - 가구들이 서로 겹치거나 공중에 떠 있으면 안 됩니다.
    - 각 디자인마다 서로 다른 가구의 색상, 크기, 스타일을 사용하여 다양성을 확보하세요.
    - 가구의 종류와 개수를 다양하게 변화시키세요 (예: 어떤 디자인은 소파 중심, 어떤 디자인은 테이블 중심).

    # 5. 금지 사항:
    - 방의 벽, 창문, 문의 위치나 크기를 변경하지 마세요.
    - 카메라 앵글이나 시점을 바꾸지 마세요.
    - 방의 전체적인 구조나 레이아웃을 재구성하지 마세요.
    - 단순히 가구를 추가하는 것에만 집중하세요.

    # 출력 규칙:
    - 절대 텍스트로 응답하지 마세요.
    - 오직 빈 방 이미지에 가구가 합성된 최종 이미지 파일 하나만 반환하세요.
    - 원본 빈 방 이미지의 구도와 구조를 정확히 유지하세요.
    """.strip()

    request_payload = [prompt, base_room_file] + furniture_files

    try:
        response = generative_model.generate_content(
            request_payload,
            request_options={"timeout": timeout},
        )
    except Exception as exc:
        raise PipelineStepError(
            f"Gemini 이미지 합성 API 호출에 실패했습니다: {exc}"
        ) from exc

    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise PipelineStepError("Gemini 응답에 후보 결과가 없습니다.")

    image_bytes: Optional[bytes] = None
    text_messages = []
    for part in candidates[0].content.parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data:
            data = getattr(inline_data, "data", None)
            if isinstance(data, str):
                image_bytes = _decode_image_bytes(data)
            elif isinstance(data, bytes):
                image_bytes = data
            if image_bytes:
                break
        text = getattr(part, "text", None)
        if text:
            text_messages.append(text)

    if image_bytes is None:
        detail = " / ".join(text_messages) if text_messages else "이미지 출력 없음"
        raise PipelineStepError(f"Gemini가 이미지를 반환하지 않았습니다. 응답: {detail}")

    result = Image.open(io.BytesIO(image_bytes))
    result.load()
    logger.info("3단계 완료: 스타일 적용 및 가구 배치 성공")
    return result


def step4_iterative_refinement(
    final_image: Image.Image,
    refinement_prompt: str,
    openai_client,
    *,
    size: str = "1024x1024",
    filename: str = "step4_input.webp",
) -> Image.Image:
    """Refine the generated image using OpenAI."""
    logger.info("4단계 시작: 부분 수정 (prompt=%s)", refinement_prompt)

    try:
        byte_stream = io.BytesIO()
        final_image.save(byte_stream, format="WEBP")
        byte_stream.seek(0)
        image_bytes = byte_stream.read()
    except Exception as exc:
        raise PipelineStepError("최종 이미지를 수정 입력용으로 변환하지 못했습니다.") from exc

    try:
        response = openai_client.images.edit(
            model="gpt-image-1",
            image=(filename, image_bytes, "image/webp"),
            prompt=refinement_prompt,
            size=size,
        )
    except Exception as exc:
        raise PipelineStepError("OpenAI 이미지 수정 API 호출에 실패했습니다.") from exc

    try:
        encoded = response.data[0].b64_json
    except (AttributeError, IndexError, KeyError) as exc:
        raise PipelineStepError("OpenAI 응답에 수정 이미지 데이터가 없습니다.") from exc

    image_bytes = _decode_image_bytes(encoded)
    result = Image.open(io.BytesIO(image_bytes))
    result.load()
    logger.info("4단계 완료: 부분 수정 성공")
    return result


def run_design_pipeline(
    original_image_path: str,
    *,
    openai_client,
    generative_model,
    style_prompt: str,
    refinement_prompt: Optional[str] = None,
    furniture_paths: Optional[Sequence[str]] = None,
    size: str = "1024x1024",
    gemini_timeout: int = 180,
    variations: int = 1,
    use_catalog_furniture: bool = False,
    catalog_furniture_plan: Optional[List[List[FurnitureItem]]] = None,
    catalog_instruction: str = "",
) -> Dict[str, object]:
    """Run the full AI pipeline and return all variants."""
    empty_room = generate_empty_room(
        original_image_path,
        openai_client,
        size=size,
    )

    variants: List[Dict[str, Image.Image]] = []
    variation_count = max(1, int(variations))
    errors: List[str] = []
    catalog_plan = catalog_furniture_plan or []
    temp_empty_room_path: Optional[str] = None

    if use_catalog_furniture:
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        empty_room.save(temp_file, format="PNG")
        temp_file.flush()
        temp_file.close()
        temp_empty_room_path = temp_file.name

    try:
        for index in range(variation_count):
            try:
                logger.info("변형 %d/%d 생성 시작", index + 1, variation_count)
                generated_image: Optional[Image.Image] = None
                generated_meta: Optional[List[dict]] = None

                catalog_items: Optional[List[FurnitureItem]] = None
                if use_catalog_furniture and catalog_plan:
                    if index < len(catalog_plan):
                        catalog_items = catalog_plan[index] or None
                    else:
                        catalog_items = catalog_plan[-1] or None

                if use_catalog_furniture and catalog_items:
                    generated_meta = [
                        {
                            "goods_id": item.goods_id,
                            "goods_name": item.goods_name,
                        }
                        for item in catalog_items
                    ]
                    try:
                        base_path = Path(temp_empty_room_path or original_image_path)
                        catalog_result = inject_with_gemini(
                            base_path,
                            catalog_items,
                            instruction=catalog_instruction or "",
                            generative_model=generative_model,
                        )
                        generated_image = Image.open(catalog_result.output_path)
                        generated_image.load()
                        generated_meta = [
                            {
                                "goods_id": item.goods_id,
                                "goods_name": item.goods_name,
                            }
                            for item in catalog_result.furniture_items
                        ]
                        logger.info("카탈로그 가구 변형 %d 생성 완료", index + 1)
                    except Exception as exc:
                        logger.warning(
                            "카탈로그 가구 삽입 실패(index=%d): %s. 기본 파이프라인으로 대체합니다.",
                            index + 1,
                            exc,
                        )

                if generated_image is None:
                    with_furniture = step3_add_local_furniture(
                        empty_room,
                        style_prompt,
                        furniture_paths or [],
                        generative_model=generative_model,
                        timeout=gemini_timeout,
                        variation_index=index,
                    )
                    generated_image = with_furniture
                else:
                    with_furniture = generated_image

                final_image = generated_image
                if refinement_prompt:
                    final_image = step4_iterative_refinement(
                        generated_image,
                        refinement_prompt,
                        openai_client,
                        size=size,
                    )

                variants.append(
                    {
                        "index": index + 1,
                        "with_furniture": with_furniture,
                        "final_image": final_image,
                        "catalog_furnitures": generated_meta,
                    }
                )
                logger.info("변형 %d 생성 완료", index + 1)
            except PipelineStepError as exc:
                logger.error("변형 %d 생성 실패: %s", index + 1, exc)
                errors.append(str(exc))
                if not variants:
                    raise
                break
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("변형 %d 생성 중 예기치 못한 오류 발생", index + 1)
                errors.append(f"예기치 못한 오류: {exc}")
                if not variants:
                    raise PipelineStepError("파이프라인 실행 중 예기치 못한 오류가 발생했습니다.") from exc
                break
    finally:
        if temp_empty_room_path and os.path.exists(temp_empty_room_path):
            try:
                os.remove(temp_empty_room_path)
            except OSError:
                logger.warning("임시 빈 방 이미지 삭제 실패: %s", temp_empty_room_path)

    return {
        "empty_room": empty_room,
        "variants": variants,
        "errors": errors,
    }


__all__ = [
    "PipelineStepError",
    "generate_empty_room",
    "step3_add_local_furniture",
    "step4_iterative_refinement",
    "run_design_pipeline",
]