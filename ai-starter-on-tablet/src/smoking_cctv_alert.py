"""Smoking area CCTV analysis pipeline.

This module provides a production-ready skeleton for detecting smoking activity
in CCTV footage, tracking repeated behavior, and dispatching alert signals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("smoking_cctv_alert")


@dataclass(frozen=True)
class Zone:
    """Defines a polygon zone (e.g., smoking area) in normalized coordinates."""

    zone_id: str
    polygon: Sequence[Tuple[float, float]]
    min_smoke_seconds: float


@dataclass(frozen=True)
class AlertTarget:
    """Defines where to send alerts."""

    name: str
    webhook_url: str
    cool_down_seconds: float


@dataclass
class AppConfig:
    """Configuration for the CCTV alert pipeline."""

    sample_rate_fps: float
    min_confidence: float
    zones: List[Zone]
    alert_targets: List[AlertTarget]
    dry_run: bool = False

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "AppConfig":
        zones = [
            Zone(
                zone_id=item["zone_id"],
                polygon=[tuple(point) for point in item["polygon"]],
                min_smoke_seconds=float(item["min_smoke_seconds"]),
            )
            for item in payload.get("zones", [])
        ]
        alert_targets = [
            AlertTarget(
                name=item["name"],
                webhook_url=item["webhook_url"],
                cool_down_seconds=float(item["cool_down_seconds"]),
            )
            for item in payload.get("alert_targets", [])
        ]
        return cls(
            sample_rate_fps=float(payload.get("sample_rate_fps", 2.0)),
            min_confidence=float(payload.get("min_confidence", 0.55)),
            zones=zones,
            alert_targets=alert_targets,
            dry_run=bool(payload.get("dry_run", False)),
        )


@dataclass(frozen=True)
class Detection:
    """Represents a detected smoking event in a frame."""

    track_id: str
    zone_id: str
    confidence: float
    timestamp: float


@dataclass
class SmokeEvent:
    """Tracks a potential smoking event over time."""

    track_id: str
    zone_id: str
    first_seen: float
    last_seen: float
    frame_count: int = 1

    def update(self, timestamp: float) -> None:
        self.last_seen = timestamp
        self.frame_count += 1

    def duration_seconds(self) -> float:
        return self.last_seen - self.first_seen


@dataclass
class Alert:
    """Alert payload after confirming smoking behavior."""

    track_id: str
    zone_id: str
    timestamp: float
    duration_seconds: float


class EventTracker:
    """Aggregates detections and emits alerts when thresholds are crossed."""

    def __init__(self, zones: Sequence[Zone]) -> None:
        self._zones = {zone.zone_id: zone for zone in zones}
        self._events: Dict[str, SmokeEvent] = {}
        self._alerted: Dict[str, float] = {}

    def process(self, detections: Iterable[Detection]) -> List[Alert]:
        alerts: List[Alert] = []
        for detection in detections:
            zone = self._zones.get(detection.zone_id)
            if zone is None:
                LOGGER.debug("Skipping detection for unknown zone %s", detection.zone_id)
                continue
            event = self._events.get(detection.track_id)
            if event is None:
                event = SmokeEvent(
                    track_id=detection.track_id,
                    zone_id=detection.zone_id,
                    first_seen=detection.timestamp,
                    last_seen=detection.timestamp,
                )
                self._events[detection.track_id] = event
            else:
                event.update(detection.timestamp)

            if event.duration_seconds() >= zone.min_smoke_seconds:
                last_alert = self._alerted.get(detection.track_id, 0.0)
                if detection.timestamp - last_alert >= zone.min_smoke_seconds:
                    self._alerted[detection.track_id] = detection.timestamp
                    alerts.append(
                        Alert(
                            track_id=detection.track_id,
                            zone_id=detection.zone_id,
                            timestamp=detection.timestamp,
                            duration_seconds=event.duration_seconds(),
                        )
                    )
        return alerts


class AlertDispatcher:
    """Sends alerts to configured webhook endpoints."""

    def __init__(self, targets: Sequence[AlertTarget], dry_run: bool) -> None:
        self._targets = targets
        self._dry_run = dry_run
        self._last_sent: Dict[str, float] = {}

    def dispatch(self, alert: Alert) -> None:
        payload = {
            "track_id": alert.track_id,
            "zone_id": alert.zone_id,
            "timestamp": dt.datetime.fromtimestamp(alert.timestamp).isoformat(),
            "duration_seconds": round(alert.duration_seconds, 2),
            "message": f"Smoking detected in zone {alert.zone_id}",
        }
        for target in self._targets:
            last_sent = self._last_sent.get(target.name, 0.0)
            if alert.timestamp - last_sent < target.cool_down_seconds:
                continue
            self._last_sent[target.name] = alert.timestamp
            if self._dry_run:
                LOGGER.info("[DRY RUN] Alert to %s: %s", target.name, payload)
                continue
            LOGGER.info("Dispatching alert to %s", target.name)
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                target.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                LOGGER.info("Alert response: %s", response.status)


class SmokeDetector:
    """Placeholder detector for smoking behavior.

    Replace this with a real model (e.g., YOLOv8 + cigarette/smoke classifier).
    """

    def __init__(self, min_confidence: float) -> None:
        self._min_confidence = min_confidence

    def detect(self, frame: object, timestamp: float) -> List[Detection]:
        del frame
        del timestamp
        return []


def _require_cv2() -> None:
    if importlib.util.find_spec("cv2") is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for live video processing. "
            "Install opencv-python to enable this feature."
        )


def iter_frames(video_path: Path, sample_rate_fps: float) -> Iterator[Tuple[object, float]]:
    _require_cv2()
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")
    frame_interval = 1.0 / sample_rate_fps
    last_ts = 0.0
    while True:
        success, frame = capture.read()
        if not success:
            break
        timestamp = time.time()
        if timestamp - last_ts >= frame_interval:
            last_ts = timestamp
            yield frame, timestamp
    capture.release()


def generate_synthetic_detections(zones: Sequence[Zone]) -> List[Detection]:
    now = time.time()
    detections: List[Detection] = []
    for index, zone in enumerate(zones, start=1):
        detections.append(
            Detection(
                track_id=f"synthetic-{index}",
                zone_id=zone.zone_id,
                confidence=0.9,
                timestamp=now,
            )
        )
    return detections


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return AppConfig.from_json(payload)


def run_pipeline(config: AppConfig, video_path: Optional[Path]) -> None:
    tracker = EventTracker(config.zones)
    dispatcher = AlertDispatcher(config.alert_targets, config.dry_run)
    detector = SmokeDetector(config.min_confidence)

    if config.dry_run:
        LOGGER.info("Running in dry-run mode with synthetic detections")
        detections = generate_synthetic_detections(config.zones)
        alerts = tracker.process(detections)
        for alert in alerts:
            dispatcher.dispatch(alert)
        return

    if video_path is None:
        raise ValueError("Video path is required unless --dry-run is set.")

    for frame, timestamp in iter_frames(video_path, config.sample_rate_fps):
        detections = detector.detect(frame, timestamp)
        alerts = tracker.process(detections)
        for alert in alerts:
            dispatcher.dispatch(alert)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoking CCTV alert pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/smoking_alert_config.json"),
        help="Path to configuration JSON.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to video file or stream.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic detections for validation.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        config.dry_run = True
    run_pipeline(config, args.video)


if __name__ == "__main__":
    main()
