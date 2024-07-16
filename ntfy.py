import base64
import dataclasses
import datetime
import json
import logging
import multiprocessing
import os.path
from abc import ABC
from enum import Enum
from typing import Dict, Optional, Final

import requests

import lib_mpex
from log import LOG_DEFAULT_FMT


class ImageAttachMethod(Enum):
    ATTACH = "attach"
    CLICK = "click"

    @staticmethod
    def from_str(method: str) -> "ImageAttachMethod":
        return {
            ImageAttachMethod.ATTACH.value.lower(): ImageAttachMethod.ATTACH,
            ImageAttachMethod.CLICK.value.lower(): ImageAttachMethod.CLICK,
        }[method.lower()]


class NtfyPriority(Enum):
    N_1 = "1"
    MIN = "min"
    N_2 = "2"
    LOW = "low"
    N_3 = "3"
    DEFAULT = "default"
    N_4 = "4"
    HIGH = "high"
    N_5 = "5"
    MAX = "max"
    URGENT = "urgent"

    @staticmethod
    def all_values() -> set:
        return {e.value for e in NtfyPriority}

    @staticmethod
    def from_str(pri: str) -> "NtfyPriority":
        return {
            NtfyPriority.N_1.value.lower(): NtfyPriority.N_1,
            NtfyPriority.MIN.value.lower(): NtfyPriority.MIN,
            NtfyPriority.N_2.value.lower(): NtfyPriority.N_2,
            NtfyPriority.LOW.value.lower(): NtfyPriority.LOW,
            NtfyPriority.N_3.value.lower(): NtfyPriority.N_3,
            NtfyPriority.DEFAULT.value.lower(): NtfyPriority.DEFAULT,
            NtfyPriority.N_4.value.lower(): NtfyPriority.N_4,
            NtfyPriority.HIGH.value.lower(): NtfyPriority.HIGH,
            NtfyPriority.N_5.value.lower(): NtfyPriority.N_5,
            NtfyPriority.MAX.value.lower(): NtfyPriority.MAX,
            NtfyPriority.URGENT.value.lower(): NtfyPriority.URGENT,
        }[pri.lower()]


NOTIF_PRIORITY_UNMUTED: Final = NtfyPriority.DEFAULT.value
NOTIF_PRIORITY_MUTED: Final = NtfyPriority.MIN.value


@dataclasses.dataclass(frozen=True)
class NtfyRecord:
    id: str
    expires_at: datetime.datetime
    jpeg_image: Optional[bytes]


@dataclasses.dataclass
class EnrichmentConfig:
    enable: bool = False
    endpoint: str = ""
    keep_alive: str = "240m"
    model: str = "llava"
    prompt_files: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    timeout_s: float = 5.0


@dataclasses.dataclass
class NtfyConfig:
    enrichment: EnrichmentConfig = dataclasses.field(default_factory=EnrichmentConfig)
    external_base_url: str = "http://localhost:5550"
    log_level: Optional[int] = logging.INFO
    topic: str = "driveway-monitor"
    server: str = "https://ntfy.sh"
    token: Optional[str] = None
    debounce_threshold_s: float = 60.0
    default_priority: NtfyPriority = NtfyPriority.DEFAULT
    priorities: Dict[str, NtfyPriority] = dataclasses.field(default_factory=lambda: {})
    req_timeout_s: float = 10.0
    image_method: Optional[ImageAttachMethod] = None
    images_cc_dir: Optional[str] = None


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
    enriched_class: Optional[str] = None

    def message(self):
        if self.enriched_class:
            return f"Likely: {self.enriched_class}.".capitalize()
        return self.title()

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
                if (
                    self._config.image_method is None
                    or self._config.image_method == ImageAttachMethod.CLICK
                ):
                    headers["Click"] = photo_url
                if (
                    self._config.image_method is None
                    or self._config.image_method == ImageAttachMethod.ATTACH
                ):
                    headers["Attach"] = photo_url
            headers["Actions"] = (
                f"{self._ntfy_mute_action_blob(10 * 60, n.id)}; "
                f"{self._ntfy_mute_action_blob(60 * 60, n.id)}; "
                f"{self._ntfy_mute_action_blob(4 * 60 * 60, n.id)}"
            )
            headers["Priority"] = self._config.priorities.get(
                n.classification,
                self._config.default_priority,
            ).value
        elif isinstance(n, FeedbackNotification):
            if n.type == FeedbackType.MUTED:
                headers["Priority"] = NOTIF_PRIORITY_MUTED
                if n.mute_seconds is not None and n.mute_seconds > 60 * 60:
                    headers["Actions"] = f"{self._ntfy_mute_action_blob(0, n.key)}"
                else:
                    headers["Actions"] = (
                        f"{self._ntfy_mute_action_blob(0, n.key)}; "
                        f"{self._ntfy_mute_action_blob(4 * 60 * 60, n.key)}; "
                        f"{self._ntfy_mute_action_blob(12*60*60, n.key)}"
                    )
            elif n.type == FeedbackType.UNMUTED:
                headers["Priority"] = NOTIF_PRIORITY_UNMUTED
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

    def _enrich(self, logger, n: ObjectNotification) -> ObjectNotification:
        if not self._config.enrichment.enable:
            return n
        if not n.jpeg_image:
            return n

        prompt_file = self._config.enrichment.prompt_files.get(n.classification)
        if not prompt_file:
            return n
        try:
            with open(prompt_file, "r") as f:
                enrichment_prompt = f.read()
        except Exception as e:
            logger.error(f"error reading enrichment prompt file '{prompt_file}': {e}")
            return n
        if not enrichment_prompt:
            return n

        try:
            resp = requests.post(
                self._config.enrichment.endpoint,
                json={
                    "model": self._config.enrichment.model,
                    "stream": False,
                    "images": [
                        base64.b64encode(n.jpeg_image).decode("ascii"),
                    ],
                    "keep_alive": self._config.enrichment.keep_alive,
                    "format": "json",
                    "prompt": enrichment_prompt,
                },
                timeout=self._config.enrichment.timeout_s,
            )
            parsed = resp.json()
        except requests.Timeout:
            logger.error("enrichment request timed out")
            return n
        except requests.RequestException as e:
            logger.error(f"enrichment failed: {e}")
            return n

        model_resp_str = parsed.get("response")
        if not model_resp_str:
            logger.error("enrichment response is missing")
            return n

        try:
            model_resp_parsed = json.loads(model_resp_str)
        except json.JSONDecodeError as e:
            logger.info(f"enrichment model did not produce valid JSON: {e}")
            logger.info(f"response: {model_resp_str}")
            return n

        if "type" not in model_resp_parsed and "error" not in model_resp_parsed:
            logger.info("enrichment model did not produce expected JSON keys")
            return n

        model_desc = model_resp_parsed.get("desc", "unknown")
        if model_desc == "unknown" or model_desc == "":
            model_err = model_resp_parsed.get("error")
            if not model_err:
                model_err = "(no error returned)"
            logger.info(
                f"enrichment model could not produce a useful description: {model_err}"
            )
            return n

        return ObjectNotification(
            t=n.t,
            classification=n.classification,
            event=n.event,
            id=n.id,
            jpeg_image=n.jpeg_image,
            enriched_class=model_desc,
        )

    def _run(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=self._config.log_level, format=LOG_DEFAULT_FMT)
        logger.info("starting notifier")

        while True:
            n: Notification = self._input_queue.get()
            logger.debug("received notification: " + n.message())

            if isinstance(n, ObjectNotification):
                if n.jpeg_image and self._config.images_cc_dir:
                    try:
                        dst_fname = f"{n.id}.jpg"
                        dst_path = os.path.join(self._config.images_cc_dir, dst_fname)
                        with open(dst_path, "wb") as f:
                            f.write(n.jpeg_image)
                    except Exception as e:
                        logger.error(f"error writing image to disk: {e}")
                if self._suppress(logger, n):
                    continue
                self._records[n.id] = NtfyRecord(
                    id=n.id,
                    jpeg_image=n.jpeg_image,
                    expires_at=n.t + datetime.timedelta(days=1),
                )
                n = self._enrich(logger, n)

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
