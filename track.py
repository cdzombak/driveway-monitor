import dataclasses
import datetime
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Final

# noinspection PyPackageRequirements
import celpy
import cv2
import torch
from ultralytics import YOLO

import lib_dmutil
import lib_mpex
from health import HealthPing
from lib_geom import Point, Box, Vector
from log import LOG_DEFAULT_FMT
from ntfy import ObjectNotification


@dataclass(frozen=True)
class TrackPrediction:
    t: datetime.datetime
    model_id: int
    classification: str
    is_track: bool
    box: Box
    image: cv2.typing.MatLike
    id: str = dataclasses.field(default_factory=lib_dmutil.rand_id)

    def to_cel(self) -> celpy.celtypes.Value:
        """
        Convert this object to a CEL value, leaving out the image.
        :return:
        """
        return celpy.celtypes.MapType(
            {
                "t": celpy.celtypes.TimestampType(self.t),
                "classification": celpy.celtypes.StringType(self.classification),
                "box": self.box.to_cel(),
            }
        )


@dataclass
class Track:
    predictions: List[TrackPrediction]
    best_image: cv2.typing.MatLike
    best_image_coverage: float
    is_model_track: bool
    triggered_notification: bool
    id: str

    @staticmethod
    def from_prediction(p: TrackPrediction) -> "Track":
        return Track(
            predictions=[p],
            best_image=p.image,
            best_image_coverage=(p.box.b.x - p.box.a.x) * (p.box.b.y - p.box.a.y),
            is_model_track=p.is_track,
            triggered_notification=False,
            id=p.id,
        )

    def first_t(self) -> datetime.datetime:
        return self.predictions[0].t

    def last_t(self) -> datetime.datetime:
        return self.predictions[-1].t

    def first_box(self) -> Box:
        return self.predictions[0].box

    def last_box(self) -> Box:
        return self.predictions[-1].box

    def average_box(self) -> Box:
        return Box(
            a=Point(
                x=sum([p.box.a.x for p in self.predictions]) / len(self.predictions),
                y=sum([p.box.a.y for p in self.predictions]) / len(self.predictions),
            ),
            b=Point(
                x=sum([p.box.b.x for p in self.predictions]) / len(self.predictions),
                y=sum([p.box.b.y for p in self.predictions]) / len(self.predictions),
            ),
        )

    def total_box(self) -> Box:
        return Box(
            a=Point(
                x=min([p.box.a.x for p in self.predictions]),
                y=min([p.box.a.y for p in self.predictions]),
            ),
            b=Point(
                x=max([p.box.b.x for p in self.predictions]),
                y=max([p.box.b.y for p in self.predictions]),
            ),
        )

    def classification(self) -> str:
        name_votes = {}
        for p in self.predictions:
            name_votes[p.classification] = name_votes.get(p.classification, 0) + 1
        return max(name_votes, key=name_votes.get)

    def length_t(self) -> datetime.timedelta:
        return self.last_t() - self.first_t()

    def movement_vector(self) -> Vector:
        return self.first_box().center().vector_to(self.last_box().center())

    def to_cel(self) -> celpy.celtypes.Value:
        """
        Convert this object to a CEL value, leaving out image-related fields.
        :return:
        """
        return celpy.celtypes.MapType(
            {
                "predictions": celpy.celtypes.ListType(
                    [p.to_cel() for p in self.predictions]
                ),
                "first_t": celpy.celtypes.TimestampType(self.first_t()),
                "last_t": celpy.celtypes.TimestampType(self.last_t()),
                "first_box": self.first_box().to_cel(),
                "last_box": self.last_box().to_cel(),
                "classification": celpy.celtypes.StringType(self.classification()),
                "length_t": celpy.celtypes.DurationType(self.length_t()),
                "average_box": self.average_box().to_cel(),
                "total_box": self.total_box().to_cel(),
                "movement_vector": self.movement_vector().to_cel(),
            }
        )

    def last_2_box_avg(self) -> Box:
        if len(self.predictions) < 2:
            return self.last_box()
        return self.predictions[-2].box.average_with(self.last_box())

    def add_prediction(self, p: TrackPrediction):
        self.predictions.append(p)
        if p.is_track:
            self.is_model_track = True
        image_coverage = (p.box.b.x - p.box.a.x) * (p.box.b.y - p.box.a.y)
        if image_coverage > self.best_image_coverage:
            self.best_image = p.image
            self.best_image_coverage = image_coverage


class VideoEnded(Exception):
    pass


@dataclass
class ModelConfig:
    log_level: Optional[int] = logging.INFO
    device: Optional[str] = None
    half: Optional[bool] = None
    confidence: float = 0.5
    iou: float = 0.15
    max_det: int = 5
    liveness_tick_s: float = 30.0
    healthcheck_ping_url: Optional[str] = None
    video_read_timeout_ms: int = (
        15000  # Timeout for VideoCapture read operations (milliseconds)
    )
    stream_reconnect_initial_backoff_s: float = 1.0
    stream_reconnect_max_backoff_s: float = 30.0


class PredModel(lib_mpex.ChildProcess):
    def __init__(
        self,
        in_fname: str,
        config: ModelConfig,
        output_queue: multiprocessing.Queue,  # of TrackPrediction
        health_ping_queue: multiprocessing.Queue,  # of HealthPing
    ):
        self._in_fname = in_fname
        self._config = config
        self._output_queue = output_queue
        self._health_ping_queue = health_ping_queue
        self._frames_seen_last_run = 0

    def _run(self):
        is_stream: Final = self._in_fname.casefold().startswith(
            "rtsp:"
        ) or self._in_fname.casefold().startswith("rtsps:")

        logger: Final = logging.getLogger(__name__ + ".Model")
        logging.basicConfig(level=self._config.log_level, format=LOG_DEFAULT_FMT)
        logger.debug(f"healthcheck ping URL: {self._config.healthcheck_ping_url}")

        model_name: Final = "yolov8n.pt"
        logger.info(f"starting model {model_name}")
        model = YOLO(model_name)

        dev = self._config.device
        if dev is None:
            has_mps = torch.backends.mps.is_available()
            has_cuda = torch.cuda.is_available()
            if has_mps:
                dev = "mps"
            elif has_cuda:
                dev = "cuda"
            else:
                dev = "cpu"
        half = self._config.half
        if half is None and dev in {"mps", "cuda"}:
            half = True
        else:
            half = False
        logger.info(f"{model_name} will use device '{dev}' (half={half})")

        reconnect_delay_s = self._config.stream_reconnect_initial_backoff_s
        reconnect_delay_s = max(reconnect_delay_s, 0.1)
        reconnect_delay_max_s = max(
            reconnect_delay_s, self._config.stream_reconnect_max_backoff_s
        )
        while True:
            try:
                self._run_capture_loop(dev, half, logger, model)
            except VideoEnded:
                if not is_stream:
                    break
                logger.warning(
                    "video stream ended or stalled; will try to reopen the source"
                )
            except (IOError, cv2.error) as exc:
                if not is_stream:
                    raise
                logger.warning(f"video stream error '{exc}'; will try to reopen")
            else:
                # non-stream capture completed successfully
                if not is_stream:
                    break

            if not is_stream:
                break

            if self._frames_seen_last_run > 0:
                reconnect_delay_s = self._config.stream_reconnect_initial_backoff_s
            else:
                reconnect_delay_s = min(
                    reconnect_delay_s * 2,
                    reconnect_delay_max_s,
                )

            logger.info(f"attempting to reopen stream in {reconnect_delay_s:.1f}s ...")
            time.sleep(reconnect_delay_s)
            # loop continues until the process is terminated by the parent

        # at this point we're done, with no error; delay exit if requested:
        self._delay_exit_if_requested(logger)

    def _run_capture_loop(self, dev, half, logger, model):
        liveness_tick_t: Final = datetime.timedelta(
            seconds=self._config.liveness_tick_s
        )
        last_liveness_tick_at: Optional[datetime.datetime] = None
        frames_since_last_liveness_tick = 0
        logger.info(f"opening video source {self._in_fname}")
        self._frames_seen_last_run = 0
        cap = cv2.VideoCapture(self._in_fname)
        try:
            # Set read timeout to prevent indefinite blocking during RTSP connection issues
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._config.video_read_timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._config.video_read_timeout_ms)
            logger.debug(
                f"VideoCapture timeouts set to {self._config.video_read_timeout_ms}ms"
            )
            if not cap.isOpened():
                raise IOError(f"failed to open video source {self._in_fname}")
            while cap.isOpened():
                success, frame = cap.read()
                if success and frame is not None:
                    self._frames_seen_last_run += 1
                    utcnow = datetime.datetime.now(datetime.UTC)
                    frames_since_last_liveness_tick += 1
                    results = model.track(
                        frame,
                        conf=self._config.confidence,
                        iou=self._config.iou,
                        max_det=self._config.max_det,
                        persist=True,
                        device=dev,
                        half=half,
                        verbose=False,
                    )
                    for b in results[0].boxes:
                        if b.id is not None:
                            xyxyn = b.xyxyn.numpy()
                            p = TrackPrediction(
                                t=utcnow,
                                model_id=int(b.id.item()),
                                classification=results[0].names[int(b.cls)],
                                is_track=b.is_track,
                                box=Box(
                                    a=Point(x=xyxyn.item(0), y=xyxyn.item(1)),
                                    b=Point(x=xyxyn.item(2), y=xyxyn.item(3)),
                                ),
                                image=frame,
                            )
                            self._output_queue.put_nowait(p)

                    if (
                        last_liveness_tick_at is None
                        or (utcnow - last_liveness_tick_at) > liveness_tick_t
                    ):
                        if last_liveness_tick_at is not None:
                            logger.debug(
                                f"liveness tick at {utcnow}; processed "
                                f"{frames_since_last_liveness_tick} frames since last tick "
                                f"({frames_since_last_liveness_tick / (utcnow - last_liveness_tick_at).total_seconds()} fps)"
                            )
                        last_liveness_tick_at = utcnow
                        frames_since_last_liveness_tick = 0
                        self._health_ping_queue.put_nowait(
                            HealthPing(
                                at_t=utcnow,
                                url=self._config.healthcheck_ping_url,
                            )
                        )
                else:
                    logger.warning(
                        f"read from video source {self._in_fname} failed; reconnect required"
                    )
                    raise VideoEnded
        finally:
            cap.release()
            logger.debug(f"released video source {self._in_fname}")

    @staticmethod
    def _delay_exit_if_requested(logger):
        exit_mins_str = os.getenv("DM_DEV_EXIT_DELAY_MINS", "")
        if not exit_mins_str:
            return
        exit_mins = int(exit_mins_str)
        if exit_mins <= 0:
            return
        logger.info(f"DM_DEV_EXIT_DELAY_MINS={exit_mins_str}; delaying exit")
        time.sleep(exit_mins * 60)


@dataclass
class TrackerConfig:
    log_level: Optional[int] = logging.INFO
    # prune out tracks that have seen no activity in this many seconds.
    # this prevents them from being appended to by new motion:
    inactive_track_prune_s: float = 1.0
    # minimum overlap percentage with the average of the last 2 boxes in the track,
    # assuming best case (classification is the same):
    track_connect_min_overlap: float = 0.2
    # only notify if the track is classified as one of these:
    notify_classification_allowlist: Optional[List[str]] = None
    # don't notify if the track is classified as one of these:
    notify_classification_blocklist: Optional[List[str]] = None
    # only notify if the track's length (in time) is at least this many seconds:
    notify_min_track_length_s: float = 1
    # allows customizing the minimum track length, in seconds, per classification
    # (e.g. a person walking might need to be tracked for longer than a car to
    # warrant a notification):
    notify_min_track_length_s_per_classification: dict[str, float] = dataclasses.field(
        default_factory=lambda: {}
    )
    notify_track_cel: Optional[str] = None


class Tracker(lib_mpex.ChildProcess):
    def __init__(
        self,
        config: TrackerConfig,
        input_queue: multiprocessing.Queue,  # of TrackPrediction
        output_queue: multiprocessing.Queue,  # of lib_ntfy.Notification
    ):
        self._config = config
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._tracks = list()

    def _run(self):
        logger = logging.getLogger(__name__ + ".Tracker")
        logging.basicConfig(level=self._config.log_level, format=LOG_DEFAULT_FMT)
        logger.info("starting tracker")

        cel_env = celpy.Environment()
        notify_cel_program: Optional[celpy.Runner] = None
        if self._config.notify_track_cel:
            notify_cel_ast = cel_env.compile(self._config.notify_track_cel)
            notify_cel_program = cel_env.program(notify_cel_ast)
        for ln in (
            "Evaluator",
            "evaluation",
            "NameContainer",
            "celtypes",
            "Environment",
        ):
            # https://github.com/cloud-custodian/cel-python/issues/46
            logging.getLogger(ln).setLevel(max(self._config.log_level, logging.WARNING))

        while True:
            p: TrackPrediction = self._input_queue.get()
            now = p.t
            logger.debug(
                f"(pred {p.id}) received prediction of {p.classification} at {now}"
            )

            # prune out tracks older than threshold:
            self._tracks = [
                t
                for t in self._tracks
                if now - t.last_t()
                < datetime.timedelta(seconds=self._config.inactive_track_prune_s)
            ]

            # connect to a preexisting track if possible:
            track: Optional[Track] = None
            for candidate in self._tracks:
                overlap_needed = self._config.track_connect_min_overlap
                # overlap requirement increases is classification is different:
                if p.classification != candidate.classification():
                    overlap_needed = overlap_needed * 1.5
                # FUTURE(cdzombak): consider velocity vector similarity
                candidate_ia = candidate.last_2_box_avg().percent_intersection_with(
                    p.box
                )
                if candidate_ia > overlap_needed:
                    if track is not None:
                        if candidate_ia > track.last_box().percent_intersection_with(
                            p.box
                        ):
                            track = candidate
                    else:
                        track = candidate
                    break

            # update matching track or create a new track if necessary:
            if track is not None:
                logger.debug(
                    f"(pred {p.id}) adding prediction of {p.classification} "
                    f"to existing track {track.id}"
                )
                track.add_prediction(p)
            else:
                logger.debug(
                    f"(pred {p.id}) creating new track for prediction of {p.classification}"
                )
                track = Track.from_prediction(p)
                self._tracks.append(track)

            # if the model doesn't think this is a track, skip further processing:
            if not track.is_model_track:
                logger.debug(
                    f"(trck {track.id}) is not a track yet; skip further processing"
                )
                continue

            # if this track has already triggered a notification, skip further processing:
            if track.triggered_notification:
                logger.debug(
                    f"(trck {track.id}) track has already triggered a notification; "
                    f"skip further processing"
                )
                continue

            # determine whether this track now meets notification criteria:
            if (
                self._config.notify_classification_allowlist
                and track.classification()
                not in self._config.notify_classification_allowlist
            ):
                logger.debug(
                    f"(trck {track.id}) track classification {track.classification()} is not "
                    f"allowlisted; skip further processing"
                )
                continue
            if (
                self._config.notify_classification_blocklist
                and track.classification()
                in self._config.notify_classification_blocklist
            ):
                logger.debug(
                    f"(trck {track.id}) track classification {track.classification()} is blocklisted; "
                    f"skip further processing"
                )
                continue
            min_track_len = datetime.timedelta(
                seconds=self._config.notify_min_track_length_s_per_classification.get(
                    track.classification(), self._config.notify_min_track_length_s
                )
            )
            if track.length_t() < min_track_len:
                logger.debug(
                    f"(trck {track.id}) track length {track.length_t()} is less than "
                    f"requirement of {min_track_len}; skip further processing"
                )
                continue

            # evaluate notify CEL rules:
            if notify_cel_program and not notify_cel_program.evaluate(
                {"track": track.to_cel()}
            ):
                logger.debug(
                    f"(trck {track.id}) did not pass notify_cel expression; "
                    f"skip further processing"
                )
                continue

            logger.info(
                f"(trck {track.id}) has met criteria for notification of {track.classification()}; "
                f"triggered notification"
            )

            # at this point the track has met all criteria; send notification:
            jpeg: Optional[bytes] = None
            ok, jpegarr = cv2.imencode(".jpg", track.best_image)
            if ok:
                jpeg = jpegarr.tobytes()
            else:
                logger.warning("failed to encode frame to JPEG")
            self._output_queue.put_nowait(
                ObjectNotification(
                    t=track.first_t(),
                    classification=track.classification(),
                    event="arrived in driveway",
                    jpeg_image=jpeg,
                    id=track.id,
                )
            )
            track.triggered_notification = True
