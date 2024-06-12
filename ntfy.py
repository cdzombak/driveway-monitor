import dataclasses
import datetime
import logging
import multiprocessing
from abc import ABC
from enum import Enum
from typing import Dict, Optional

import requests

import lib_mpex
from log import LOG_DEFAULT_FMT


@dataclasses.dataclass(frozen=True)
class NtfyRecord:
    id: str
    expires_at: datetime.datetime
    jpeg_image: Optional[bytes]


@dataclasses.dataclass
class NtfyConfig:
    external_base_url: str = "http://localhost:5550"
    log_level: Optional[int] = logging.INFO
    topic: str = "driveway-monitor"
    server: str = "https://ntfy.sh"
    token: Optional[str] = None
    debounce_threshold_s: float = 60.0
    default_priority: str = "1"
    priorities: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "car": "4",
            "truck": "4",
            "person": "4",
        }
    )
    req_timeout_s: float = 10.0


class Notification(ABC):
    def message(self) -> str:
        raise NotImplementedError

    def title(self) -> Optional[str]:
        raise NotImplementedError

    def ntfy_tags(self) -> Optional[str]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ObjectNotification(Notification):
    t: datetime.datetime
    classification: str
    event: str
    id: str
    jpeg_image: Optional[bytes]

    def message(self):
        return f"{self.classification} {self.event}.".capitalize()

    def title(self):
        return f"{self.classification} {self.event}".capitalize()

    def ntfy_tags(self) -> str:
        return {
            "car": "blue_car",
            "truck": "truck",
            "person": "walking",
        }.get(self.classification, "camera_flash")


class FeedbackType(Enum):
    MUTED = "muted"
    UNMUTED = "unmuted"


@dataclasses.dataclass(frozen=True)
class FeedbackNotification(Notification):
    type: FeedbackType
    key: str
    mute_seconds: Optional[int] = None

    def message(self):
        return f"Notifications {self.type.value}."

    def title(self):
        return "driveway-monitor"

    def ntfy_tags(self) -> str:
        if self.type == FeedbackType.MUTED:
            return "mute"
        elif self.type == FeedbackType.UNMUTED:
            return "loud_sound"
        else:
            return "speech_balloon"


class UnknownNotificationType(Exception):
    pass


class Notifier(lib_mpex.ChildProcess):
    @staticmethod
    def valid_priorities() -> set:
        return {
            "1",
            "2",
            "3",
            "4",
            "5",
            "max",
            "urgent",
            "high",
            "default",
            "low",
            "min",
        }

    def __init__(
        self,
        config: NtfyConfig,
        input_queue: multiprocessing.Queue,
        web_share_ns,
        records_dict: Dict[str, NtfyRecord],
    ):
        self._config = config
        if self._config.external_base_url.endswith("/"):
            self._config.external_base_url = self._config.external_base_url[:-1]
        self._input_queue = input_queue
        self._last_notification: Dict[str, datetime.datetime] = {}
        self._web_share_ns = web_share_ns
        self._web_share_ns.mute_until = None
        self._records = records_dict

    def _prep_ntfy_headers(self, n: Notification) -> Dict[str, str]:
        headers = {}

        tags = n.ntfy_tags()
        if tags:
            headers["Tags"] = tags
        title = n.title()
        if title:
            headers["Title"] = title

        if isinstance(n, ObjectNotification):
            if n.jpeg_image:
                photo_url = f"{self._config.external_base_url}/photo/{n.id}.jpg"
                headers["Click"] = photo_url
                headers["Attach"] = photo_url
                headers["Actions"] = (
                    f"view, Photo, {photo_url}; "
                    f"{self._ntfy_mute_action_blob(10 * 60, n.id)}; "
                    f"{self._ntfy_mute_action_blob(60 * 60, n.id)}"
                )
            else:
                headers["Actions"] = (
                    f"{self._ntfy_mute_action_blob(10 * 60, n.id)}; "
                    f"{self._ntfy_mute_action_blob(60 * 60, n.id)}; "
                    f"{self._ntfy_mute_action_blob(4 * 60 * 60, n.id)}"
                )
            headers["Priority"] = self._config.priorities.get(
                n.classification,
                self._config.default_priority,
            )
        elif isinstance(n, FeedbackNotification):
            if n.type == FeedbackType.MUTED:
                headers["Priority"] = "2"
                if n.mute_seconds is not None and n.mute_seconds > 60 * 60:
                    headers["Actions"] = f"{self._ntfy_mute_action_blob(0, n.key)}"
                else:
                    headers["Actions"] = (
                        f"{self._ntfy_mute_action_blob(0, n.key)}; "
                        f"{self._ntfy_mute_action_blob(4 * 60 * 60, n.key)}; "
                        f"{self._ntfy_mute_action_blob(12*60*60, n.key)}"
                    )
            elif n.type == FeedbackType.UNMUTED:
                headers["Priority"] = "4"
                headers["Actions"] = (
                    f"{self._ntfy_mute_action_blob(10*60, n.key)}; "
                    f"{self._ntfy_mute_action_blob(60 * 60, n.key)}; "
                    f"{self._ntfy_mute_action_blob(4 * 60 * 60, n.key)}"
                )

        return headers

    def _ntfy_mute_action_blob(self, seconds: int, key: str) -> str:
        t_str = f"{seconds // 60}m"
        if seconds >= 3600:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            if minutes == 0:
                t_str = f"{hours}h"
            else:
                t_str = f"{hours}h {minutes}m"
        act_str = f"Mute {t_str}"
        if seconds == 0:
            act_str = "Unmute"
        return (
            f"http, {act_str}, {self._config.external_base_url}/mute, "
            f'body=\'{{"s": {seconds}, "key": "{key}"}}\', '
            f"headers.content-type=application/json, clear=true"
        )

    def _suppress(self, logger, n: ObjectNotification) -> bool:
        mute_until: Optional[datetime.datetime] = self._web_share_ns.mute_until
        if mute_until and n.t < mute_until:
            logger.info(
                f"notification '{n.title()}' suppressed due to mute until {mute_until}"
            )
            return True

        last_ntfy = self._last_notification.get(n.classification)
        if last_ntfy and n.t - last_ntfy < datetime.timedelta(
            seconds=self._config.debounce_threshold_s
        ):
            logger.info(f"notification '{n.title()}' suppressed due to debounce")
            return True

        self._last_notification[n.classification] = n.t
        return False

    def _run(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=self._config.log_level, format=LOG_DEFAULT_FMT)
        logger.info("starting notifier")

        while True:
            n: Notification = self._input_queue.get()
            logger.debug("received notification: " + n.message())

            if isinstance(n, ObjectNotification):
                if self._suppress(logger, n):
                    continue
                self._records[n.id] = NtfyRecord(
                    id=n.id,
                    jpeg_image=n.jpeg_image,
                    expires_at=n.t + datetime.timedelta(days=1),
                )

            try:
                headers = self._prep_ntfy_headers(n)
            except UnknownNotificationType:
                logger.error(
                    f"received unknown notification type from input queue: {type(n)}"
                )
                continue

            if self._config.token:
                headers["Authorization"] = "Bearer " + self._config.token

            try:
                requests.post(
                    f"{self._config.server}/{self._config.topic}",
                    data=n.message().encode(encoding="utf-8"),
                    headers=headers,
                    timeout=self._config.req_timeout_s,
                )
                logger.info(f"notification '{n.message()}' sent")
            except requests.RequestException as e:
                logger.error(f"error sending notification: {e}")

            self._prune_photos_cache()

    def _prune_photos_cache(self):
        # prune old photos from the shared dict,
        # using input_queue notifications as a periodic ticker:
        utcnow = datetime.datetime.now(datetime.timezone.utc)
        keys_to_prune = [k for k, v in self._records.items() if v.expires_at < utcnow]
        for k in keys_to_prune:
            del self._records[k]


def print_notifier(notifications_queue: multiprocessing.Queue):
    while True:
        n: ObjectNotification = notifications_queue.get()
        print(f"***NOTIFICATION*** at {n.t}: {n.classification.capitalize()} {n.event}")
