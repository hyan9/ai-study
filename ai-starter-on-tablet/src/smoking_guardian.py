"""Smoking detection CCTV monitor with Qwen3 TTS warnings."""
from __future__ import annotations

import argparse
import dataclasses
import csv
import datetime as dt
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline,
)
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

DEFAULT_SMOKING_MODEL = "Enos-123/smoking-detection"
DEFAULT_QWEN_TTS_MODEL = "Qwen/Qwen3-TTS"
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"


@dataclasses.dataclass
class PersonDetection:
    box: Tuple[int, int, int, int]
    score: float


@dataclasses.dataclass
class SmokingResult:
    score: float
    label: str


@dataclasses.dataclass
class AlertEvent:
    timestamp: str
    label: str
    score: float
    caption: str
    audio_path: str
    frame_path: Optional[str]


class PersonDetector:
    def __init__(self, device: str) -> None:
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self.model.eval().to(device)
        self.device = device
        self.transform = weights.transforms()

    @torch.no_grad()
    def detect(self, frame: np.ndarray, min_score: float = 0.7) -> List[PersonDetection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = self.transform(image).to(self.device)
        outputs = self.model([tensor])[0]
        detections: List[PersonDetection] = []
        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            if int(label) != 1:
                continue
            if float(score) < min_score:
                continue
            x1, y1, x2, y2 = [int(x) for x in box.tolist()]
            detections.append(PersonDetection((x1, y1, x2, y2), float(score)))
        return detections


class SmokingDetector:
    def __init__(self, model_id: str, device: str) -> None:
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(device).eval()

    @torch.no_grad()
    def classify(self, frame: np.ndarray) -> SmokingResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        scores = torch.softmax(outputs.logits, dim=-1)[0]
        best = int(torch.argmax(scores))
        label = self.model.config.id2label[best]
        return SmokingResult(score=float(scores[best]), label=label)


class AppearanceCaptioner:
    def __init__(self, model_id: str, device: str) -> None:
        device_index = 0 if device.startswith("cuda") else -1
        self.pipe = pipeline("image-to-text", model=model_id, device=device_index)

    def describe(self, frame: np.ndarray) -> str:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        results = self.pipe(image, max_new_tokens=60)
        caption = results[0]["generated_text"].strip()
        return caption


class Qwen3TTS:
    def __init__(self, model_id: str, device: str, out_dir: Path, voice: str) -> None:
        device_index = 0 if device.startswith("cuda") else -1
        self.pipe = pipeline("text-to-speech", model=model_id, device=device_index)
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice

    def synthesize(self, text: str) -> Path:
        if self.voice:
            try:
                result = self.pipe(text, voice=self.voice)
            except TypeError:
                try:
                    result = self.pipe(text, speaker=self.voice)
                except TypeError:
                    result = self.pipe(text)
        else:
            result = self.pipe(text)
        audio = result["audio"]
        sample_rate = result["sampling_rate"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.out_dir / f"warning_{timestamp}.wav"
        sf.write(out_path, audio, sample_rate)
        return out_path


class AudioPlayer:
    def __init__(self) -> None:
        self._sounddevice = None
        try:
            import sounddevice  # type: ignore

            self._sounddevice = sounddevice
        except Exception:
            self._sounddevice = None

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        if self._sounddevice is None:
            return
        self._sounddevice.play(audio, sample_rate)
        self._sounddevice.wait()


class WarningLogic:
    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = cooldown_s
        self.last_warning_time = 0.0

    def should_warn(self) -> bool:
        now = time.time()
        if now - self.last_warning_time >= self.cooldown_s:
            self.last_warning_time = now
            return True
        return False


class EventLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "label",
                        "score",
                        "caption",
                        "audio_path",
                        "frame_path",
                    ],
                )
                writer.writeheader()

    def write(self, event: AlertEvent) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "label",
                    "score",
                    "caption",
                    "audio_path",
                    "frame_path",
                ],
            )
            writer.writerow(dataclasses.asdict(event))


def crop_frame(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2]


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    new_size = (max_width, int(frame.shape[0] * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def build_warning_text(caption: str, label: str, score: float) -> str:
    threat_line = f"감지 결과: {label} 확률 {score:.0%}."
    return (
        "경고합니다. 이 구역은 금연 구역입니다. "
        f"인상착의는 다음과 같습니다: {caption}. "
        f"{threat_line} 즉시 담배를 끄고 자리를 정리하세요."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoking detection CCTV monitor")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g., 0) or video/RTSP URL",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu override")
    parser.add_argument("--smoking-model", type=str, default=DEFAULT_SMOKING_MODEL)
    parser.add_argument("--tts-model", type=str, default=DEFAULT_QWEN_TTS_MODEL)
    parser.add_argument(
        "--tts-voice",
        type=str,
        default="latest",
        help="Qwen3 TTS voice preset or speaker name",
    )
    parser.add_argument("--caption-model", type=str, default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--min-smoking-score", type=float, default=0.6)
    parser.add_argument("--person-score", type=float, default=0.7)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--log-csv", type=str, default="outputs/events.csv")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save frames when a smoking alert is triggered",
    )
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--play-audio", action="store_true")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.save_dir)
    event_logger = EventLogger(Path(args.log_csv))
    audio_player = AudioPlayer() if args.play_audio else None

    person_detector = PersonDetector(device)
    smoking_detector = SmokingDetector(args.smoking_model, device)
    captioner = AppearanceCaptioner(args.caption_model, device)
    tts = Qwen3TTS(args.tts_model, device, output_dir / "tts", args.tts_voice)
    warning_logic = WarningLogic(args.cooldown)

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError("카메라/스트림을 열 수 없습니다. 소스를 확인하세요.")

    frame_count = 0
    print("[INFO] CCTV 감시 시작. 종료하려면 q 키를 누르세요.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 프레임을 읽을 수 없습니다.")
                break

            frame = resize_frame(frame, args.max_width)
            frame_count += 1
            if frame_count % args.frame_skip != 0:
                cv2.imshow("Smoking Guardian", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            detections = person_detector.detect(frame, min_score=args.person_score)
            for detection in detections:
                person_crop = crop_frame(frame, detection.box)
                smoking_result = smoking_detector.classify(person_crop)
                if smoking_result.score < args.min_smoking_score:
                    continue

                if not warning_logic.should_warn():
                    continue

                caption = captioner.describe(person_crop)
                warning_text = build_warning_text(
                    caption, smoking_result.label, smoking_result.score
                )
                audio_path = tts.synthesize(warning_text)
                if audio_player is not None:
                    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
                    audio_player.play(audio, sample_rate)

                timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_path: Optional[str] = None
                if args.save_frames:
                    frame_path = str(output_dir / f"smoking_{timestamp}.jpg")
                    cv2.imwrite(frame_path, frame)

                event_logger.write(
                    AlertEvent(
                        timestamp=timestamp,
                        label=smoking_result.label,
                        score=smoking_result.score,
                        caption=caption,
                        audio_path=str(audio_path),
                        frame_path=frame_path,
                    )
                )
                print(f"[ALERT] 흡연 감지: {warning_text}")
                print(f"[ALERT] 음성 저장 위치: {audio_path}")

            for detection in detections:
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"person {detection.score:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.imshow("Smoking Guardian", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
