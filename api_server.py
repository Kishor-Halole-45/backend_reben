"""FastAPI service for query-driven image grounding."""

import io
import gc
import os
import re
from functools import lru_cache
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Owlv2ForObjectDetection,
    Owlv2Processor,
    ViltForQuestionAnswering,
    ViltProcessor,
)

MODEL_NAME = "google/owlv2-base-patch16-ensemble"
REBEN_MODEL_NAME = "hackelle/resnet18-all-v0.1.1"
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
VQA_MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_THRESHOLD = 0.18
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/webp"}

app = FastAPI(title="ReBEN Grounding API", version="1.0.0")
frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _release_other_models(active: str) -> None:
    loaders = {
        "grounding": load_grounding_model,
        "classification": load_reben_model,
        "caption": load_caption_model,
        "vqa": load_vqa_model,
    }
    for name, loader in loaders.items():
        if name != active:
            loader.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@lru_cache(maxsize=1)
def load_grounding_model() -> tuple[Owlv2Processor, Owlv2ForObjectDetection]:
    _release_other_models("grounding")
    processor = Owlv2Processor.from_pretrained(MODEL_NAME)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to(_device())
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def load_reben_model():
    _release_other_models("classification")
    from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(REBEN_MODEL_NAME).to(_device())
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_caption_model() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    _release_other_models("caption")
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME).to(_device())
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def load_vqa_model() -> tuple[ViltProcessor, ViltForQuestionAnswering]:
    _release_other_models("vqa")
    processor = ViltProcessor.from_pretrained(VQA_MODEL_NAME)
    model = ViltForQuestionAnswering.from_pretrained(VQA_MODEL_NAME).to(_device())
    model.eval()
    return processor, model


def classify_image(image: Image.Image) -> dict[str, Any]:
    try:
        from configilm.extra.BENv2_utils import NEW_LABELS

        rgb = image.convert("RGB").resize((120, 120))
        values = torch.from_numpy(__import__("numpy").asarray(rgb, dtype="float32").transpose(2, 0, 1)) / 255
        values = values.repeat(4, 1, 1)[:12].unsqueeze(0).to(_device())
        with torch.no_grad():
            probabilities = torch.sigmoid(load_reben_model()(values)).squeeze(0)
        top_indices = torch.argsort(probabilities, descending=True)[:5]
        predictions = [{"label": NEW_LABELS[int(index)], "score": round(float(probabilities[index]), 4)} for index in top_indices]
        return {"mode": "classification", "predictions": predictions, "model": "reBEN", "device": str(_device())}
    except ModuleNotFoundError:
        labels = ["urban area", "buildings", "roads", "water", "forest", "farmland", "bare land", "grassland"]
        grounded = ground_image(image, ", ".join(labels), threshold=0.08)
        scores = {label: 0.0 for label in labels}
        for detection in grounded["detections"]:
            scores[detection["label"]] = max(scores[detection["label"]], detection["score"])
        predictions = [{"label": label, "score": score} for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]]
        return {"mode": "classification", "predictions": predictions, "model": "OWL-ViT land-cover fallback", "device": str(_device())}


def caption_image(image: Image.Image) -> dict[str, Any]:
    processor, model = load_caption_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(_device())
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=5,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            length_penalty=1.0,
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    caption = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", caption, flags=re.IGNORECASE)
    return {"mode": "caption", "answer": caption, "device": str(_device())}


def _is_count_question(query: str) -> bool:
    return bool(re.search(r"\b(how many|number of|count|数量|how much)\b", query.lower()))


def _is_presence_question(query: str) -> bool:
    return bool(re.search(r"\b(is there|are there|does the image have|can you see|do you see|visible)\b", query.lower()))


def _vqa_object_query(query: str) -> str:
    cleaned = re.sub(r"\b(how many|what is the number of|number of|count|are there|is there|in the image|in this image|can you see|do you see)\b", " ", query.lower())
    cleaned = re.sub(r"[?.,!]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip() or "object"


def _orchestration_route(query: str, image_count: int) -> tuple[str, str]:
    """Return a specialist route using deterministic, ordered intent rules."""
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    if image_count >= 2:
        return "change_detection", "Two images were supplied, so change detection has priority over the text query."
    if _is_count_question(normalized) or _is_presence_question(normalized):
        return "vqa", "The query asks for a count or object presence, so VQA will use grounding evidence."
    if re.search(r"\b(classif|classify|land ?cover|vegetation type|terrain class|scene class|what type of land)\b", normalized):
        return "classification", "The query asks for a land-cover or scene category."
    if re.search(r"\b(caption|describe|description|summari[sz]e|what is visible|what can be seen|scene overview)\b", normalized):
        return "caption", "The query asks for a natural-language scene description."
    if re.search(r"\b(find|detect|identify|locate|highlight|mark|outline|list|show me|where is|where are|objects? such as|objects? in|buildings?|roads?|water|trees?|vehicles?|airplanes?|aeroplanes?|aircraft|planes?)\b", normalized):
        return "grounding", "The query names an object or asks where an object is located."
    if normalized and ("?" in normalized or re.match(r"^(what|which|where|when|why|how|is|are|does|can|could|would)\b", normalized)):
        return "vqa", "The query is a visual question for the VQA model."
    return "caption", "No clear object target or question was supplied, so captioning is the safest route."


def answer_image(image: Image.Image, query: str) -> dict[str, Any]:
    if _is_count_question(query) or _is_presence_question(query):
        object_query = _vqa_object_query(query)
        grounded = ground_image(image, object_query)
        count = len(grounded["detections"])
        if _is_count_question(query):
            answer = str(count)
            interpretation = f"Counted {count} detected {object_query}."
        else:
            answer = "yes" if count else "no"
            interpretation = f"Grounding found {count} {object_query} detection(s)."
        return {"mode": "vqa", "answer": answer, "interpretation": interpretation, "grounding": grounded, "device": str(_device())}
    processor, model = load_vqa_model()
    inputs = processor(images=image.convert("RGB"), text=query, return_tensors="pt").to(_device())
    with torch.no_grad():
        output = model(**inputs)
    answer_id = output.logits.argmax(-1).item()
    answer = model.config.id2label.get(answer_id, "Unable to answer")
    confidence = float(output.logits.softmax(-1).max().item())
    return {"mode": "vqa", "answer": answer, "confidence": round(confidence, 4), "model": VQA_MODEL_NAME, "device": str(_device())}


def orchestrate_image(image: Image.Image, query: str, before: Image.Image | None = None, after: Image.Image | None = None) -> dict[str, Any]:
    """Choose a specialist from the available inputs and return routing evidence."""
    image_count = int(before is not None) + int(after is not None) if before is not None and after is not None else 1
    selected, reasoning = _orchestration_route(query, image_count)
    if selected == "change_detection":
        result = detect_changes(before, after)
    elif selected == "vqa":
        result = answer_image(image, query)
    elif selected == "classification":
        result = classify_image(image)
    elif selected == "caption":
        result = caption_image(image)
    else:
        result = ground_image(image, query)
    result.update({"orchestrated": True, "selected_model": selected, "routing_reason": reasoning, "input_image_count": image_count})
    return result


def _query_labels(query: str) -> list[str]:
    labels = []
    for part in re.split(r",|;|\band\b|\bor\b|\n", query.lower()):
        label = re.sub(r"\b(find|detect|identify|locate|highlight|mark|outline|list|show|label|all|objects?|things?|it|them|in|the|image|please)\b", "", part).strip()
        label = re.sub(r"\s+", " ", label)
        label = re.sub(r"\b(aeroplanes?|aircraft|planes?)\b", "airplane", label)
        label = re.sub(r"\b(automobiles?|cars?)\b", "car", label)
        if label and label not in labels:
            labels.append(label)
    return labels[:8] or ["object"]


def ground_image(image: Image.Image, query: str, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    labels = _query_labels(query)
    processor, model = load_grounding_model()
    inputs = processor(text=[labels], images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(_device()) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)], device=_device())
    results = processor.image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=0.05,
        target_sizes=target_sizes,
    )[0]

    scores_by_label: dict[int, list[float]] = {}
    for score, label_index in zip(results["scores"].tolist(), results["labels"].tolist()):
        scores_by_label.setdefault(int(label_index), []).append(float(score))

    adaptive_thresholds = {}
    for label_index, scores in scores_by_label.items():
        ordered = sorted(scores, reverse=True)
        gaps = [(ordered[index] - ordered[index + 1], (ordered[index] + ordered[index + 1]) / 2) for index in range(len(ordered) - 1)]
        largest_gap = max(gaps, default=(0.0, ordered[0] if ordered else DEFAULT_THRESHOLD))
        adaptive_thresholds[label_index] = max(0.08, min(0.38, largest_gap[1] if largest_gap[0] >= 0.04 else max(0.12, ordered[0] * 0.45)))

    detections = []
    for score, label_index, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_index = int(label_index)
        if not 0 <= label_index < len(labels):
            continue
        score_value = float(score)
        if score_value < adaptive_thresholds.get(label_index, DEFAULT_THRESHOLD):
            continue
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = sorted((max(0.0, min(x1, image.width)), max(0.0, min(x2, image.width))))
        y1, y2 = sorted((max(0.0, min(y1, image.height)), max(0.0, min(y2, image.height))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        detections.append(
            {
                "label": labels[label_index],
                "score": round(score_value, 4),
                "box": {"x": x1, "y": y1, "width": round(x2 - x1, 2), "height": round(y2 - y1, 2)},
            }
        )

    return {
        "query": query,
        "labels": labels,
        "adaptive_thresholds": {labels[index]: round(value, 4) for index, value in adaptive_thresholds.items() if index < len(labels)},
        "detections": detections,
        "image": {"width": image.width, "height": image.height},
        "device": str(_device()),
    }


def detect_changes(before: Image.Image, after: Image.Image) -> dict[str, Any]:
    """Find spatially coherent changes between two co-registered images."""
    import numpy as np

    width = min(before.width, after.width)
    height = min(before.height, after.height)
    scale = min(1.0, 512 / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    first = np.asarray(before.convert("RGB").resize(size).filter(ImageFilter.GaussianBlur(1.5)), dtype=np.int16)
    second = np.asarray(after.convert("RGB").resize(size).filter(ImageFilter.GaussianBlur(1.5)), dtype=np.int16)
    difference = np.abs(first - second).mean(axis=2)
    mask = difference >= 28
    visited = np.zeros(mask.shape, dtype=bool)
    regions = []
    for row, column in zip(*np.where(mask & ~visited)):
        if visited[row, column]:
            continue
        stack = [(int(row), int(column))]
        visited[row, column] = True
        points = []
        while stack:
            current_row, current_column = stack.pop()
            points.append((current_row, current_column))
            for next_row, next_column in ((current_row - 1, current_column), (current_row + 1, current_column), (current_row, current_column - 1), (current_row, current_column + 1)):
                if 0 <= next_row < mask.shape[0] and 0 <= next_column < mask.shape[1] and mask[next_row, next_column] and not visited[next_row, next_column]:
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        if len(points) < max(12, mask.size // 4000):
            continue
        rows, columns = zip(*points)
        regions.append({"x": round(min(columns) / scale, 2), "y": round(min(rows) / scale, 2), "width": round((max(columns) - min(columns) + 1) / scale, 2), "height": round((max(rows) - min(rows) + 1) / scale, 2), "score": round(min(1.0, float(difference[rows, columns].mean()) / 255), 4)})
    regions.sort(key=lambda region: region["score"] * region["width"] * region["height"], reverse=True)
    regions = regions[:30]
    changed_area_percent = round(float(mask.mean() * 100), 2)
    if not regions:
        explanation = "No significant visual change was detected between the two images."
    else:
        locations = []
        for region in regions[:5]:
            center_x = region["x"] + region["width"] / 2
            center_y = region["y"] + region["height"] / 2
            horizontal = "left" if center_x < width / 3 else "right" if center_x > width * 2 / 3 else "center"
            vertical = "top" if center_y < height / 3 else "bottom" if center_y > height * 2 / 3 else "middle"
            locations.append(f"{vertical}-{horizontal}")
        location_text = ", ".join(dict.fromkeys(locations))
        explanation = f"Detected {len(regions)} visually changed region(s), covering about {changed_area_percent}% of the shared image area. The main changes are in the {location_text}. This is a visual difference result; it does not identify the semantic cause of each change."
    return {"mode": "change_detection", "changes": regions, "changed_area_percent": changed_area_percent, "explanation": explanation, "image": {"width": width, "height": height}, "model": "pixel-difference change detector", "device": str(_device())}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "grounding"}


@app.post("/api/ground")
async def ground(file: UploadFile = File(...), query: str = Form(...), threshold: float = Form(DEFAULT_THRESHOLD)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Add an object or scene query first.")
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, WEBP, or TIFF image.")
    if not 0.05 <= threshold <= 0.8:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.05 and 0.8.")

    try:
        image = Image.open(io.BytesIO(await file.read()))
        return ground_image(image, query, threshold)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Grounding failed: {error}") from error


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    before_file: UploadFile | None = File(None),
    after_file: UploadFile | None = File(None),
    mode: str = Form("grounding"),
    query: str = Form(""),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    if mode not in {"orchestrate", "grounding", "classification", "caption", "vqa", "change_detection"}:
        raise HTTPException(status_code=400, detail="Choose a supported model mode.")
    if mode in {"grounding", "vqa"} and not query.strip():
        raise HTTPException(status_code=400, detail="Add a request for the selected model.")
    if mode == "change_detection":
        if before_file is None or after_file is None:
            raise HTTPException(status_code=400, detail="Upload both the earlier and later image.")
        if before_file.content_type not in SUPPORTED_IMAGE_TYPES or after_file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Upload PNG, JPG, WEBP, or TIFF images.")
    elif file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, WEBP, or TIFF image.")
    try:
        if mode == "change_detection":
            before = Image.open(io.BytesIO(await before_file.read()))
            after = Image.open(io.BytesIO(await after_file.read()))
            return detect_changes(before, after)
        image = Image.open(io.BytesIO(await file.read()))
        if mode == "orchestrate":
            before = Image.open(io.BytesIO(await before_file.read())) if before_file is not None else None
            after = Image.open(io.BytesIO(await after_file.read())) if after_file is not None else None
            return orchestrate_image(image, query, before, after)
        if mode == "grounding":
            return ground_image(image, query, threshold)
        if mode == "classification":
            return classify_image(image)
        if mode == "caption":
            return caption_image(image)
        return answer_image(image, query)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model request failed: {error}") from error
