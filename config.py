import json
import logging
from dataclasses import dataclass
from typing import Optional

from health import HealthPingerConfig
from ntfy import NtfyConfig, ImageAttachMethod, NtfyPriority
from track import ModelConfig, TrackerConfig
from web import WebConfig


@dataclass
class Config:
    model: ModelConfig
    tracker: TrackerConfig
    notifier: NtfyConfig
    health_pinger: HealthPingerConfig
    web: WebConfig


class ConfigValidationError(ValueError):
    pass


def config_from_file(
    config_file: Optional[str],
) -> Config:
    """
    Load a Config object from a JSON file.
    Validates the config file's contents.
    Returns a default config if no file is provided.

    :param config_file:
    :raises ValueError: if the config file is invalid
    :return: validated Config instance
    """
    logger = logging.getLogger(__name__)

    cfg = Config(
        model=ModelConfig(),
        notifier=NtfyConfig(),
        tracker=TrackerConfig(),
        health_pinger=HealthPingerConfig(),
        web=WebConfig(),
    )
    if not config_file:
        logger.info("no config file provided; using default config")
        return cfg

    logger.info(f"loading config from {config_file}")
    with open(config_file) as f:
        cfg_dict = json.load(f)

    # model:
    model_cfg_dict = cfg_dict.get("model", {})
    cfg.model.device = model_cfg_dict.get("device", cfg.model.device)
    if cfg.model.device is not None and not isinstance(cfg.model.device, str):
        raise ConfigValidationError("model.device must be a string")
    cfg.model.half = model_cfg_dict.get("half", cfg.model.half)
    if cfg.model.half is not None and not isinstance(cfg.model.half, bool):
        raise ConfigValidationError("model.half must be a bool")
    cfg.model.confidence = model_cfg_dict.get("confidence", cfg.model.confidence)
    if not isinstance(cfg.model.confidence, (int, float)):
        raise ConfigValidationError("model.confidence must be a number")
    cfg.model.iou = model_cfg_dict.get("iou", cfg.model.iou)
    if not isinstance(cfg.model.iou, (int, float)):
        raise ConfigValidationError("model.iou must be a number")
    cfg.model.max_det = model_cfg_dict.get("max_det", cfg.model.max_det)
    if not isinstance(cfg.model.max_det, int):
        raise ConfigValidationError("model.max_det must be an int")
    cfg.model.liveness_tick_s = model_cfg_dict.get(
        "liveness_tick_s", cfg.model.liveness_tick_s
    )
    if not isinstance(cfg.model.liveness_tick_s, (int, float)):
        raise ConfigValidationError("model.liveness_tick_s must be a number")
    if cfg.model.liveness_tick_s < 5:
        raise ConfigValidationError("model.liveness_tick_s must be >= 5")
    cfg.model.healthcheck_ping_url = model_cfg_dict.get(
        "healthcheck_ping_url", cfg.model.healthcheck_ping_url
    )
    if cfg.model.healthcheck_ping_url is not None and not isinstance(
        cfg.model.healthcheck_ping_url, str
    ):
        raise ConfigValidationError("model.healthcheck_ping_url must be a string")

    # notifier:
    ntfy_cfg_dict = cfg_dict.get("notifier", {})
    cfg.notifier.topic = ntfy_cfg_dict.get("topic", cfg.notifier.topic)
    if not isinstance(cfg.notifier.topic, str):
        raise ConfigValidationError("notifier.topic must be a string")
    cfg.notifier.server = ntfy_cfg_dict.get("server", cfg.notifier.server)
    if not isinstance(cfg.notifier.server, str):
        raise ConfigValidationError("notifier.server must be a string")
    cfg.notifier.token = ntfy_cfg_dict.get("token", cfg.notifier.token)
    if cfg.notifier.token is not None and not isinstance(cfg.notifier.token, str):
        raise ConfigValidationError("notifier.token must be a string")
    cfg.notifier.debounce_threshold_s = ntfy_cfg_dict.get(
        "debounce_threshold_s", cfg.notifier.debounce_threshold_s
    )
    if not isinstance(cfg.notifier.debounce_threshold_s, (int, float)):
        raise ConfigValidationError("notifier.debounce_threshold_s must be a number")
    default_priority_str = ntfy_cfg_dict.get(
        "default_priority", cfg.notifier.default_priority
    )
    if default_priority_str:
        cfg.notifier.default_priority = NtfyPriority.from_str(default_priority_str)
    cfg.notifier.priorities = ntfy_cfg_dict.get("priorities", cfg.notifier.priorities)
    for k, v in cfg.notifier.priorities.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ConfigValidationError(
                "notifier.priorities must be a dict of str -> str"
            )
        cfg.notifier.priorities[k] = NtfyPriority.from_str(v)
    cfg.notifier.req_timeout_s = ntfy_cfg_dict.get(
        "req_timeout_s", cfg.notifier.req_timeout_s
    )
    if not isinstance(cfg.notifier.req_timeout_s, (int, float)):
        raise ConfigValidationError("notifier.req_timeout_s must be a number")
    image_method_str = ntfy_cfg_dict.get(
        "image_method", cfg.notifier.image_method
    )
    if image_method_str:
        cfg.notifier.image_method = ImageAttachMethod.from_str(image_method_str)
    cfg.notifier.images_cc_dir = ntfy_cfg_dict.get(
        "images_cc_dir", cfg.notifier.images_cc_dir
    )
    if cfg.notifier.images_cc_dir is not None and not isinstance(
        cfg.notifier.images_cc_dir, str
    ):
        raise ConfigValidationError("notifier.images_cc_dir must be a string")

    # tracker:
    tracker_cfg_dict = cfg_dict.get("tracker", {})
    cfg.tracker.inactive_track_prune_s = tracker_cfg_dict.get(
        "inactive_track_prune_s", cfg.tracker.inactive_track_prune_s
    )
    if not isinstance(cfg.tracker.inactive_track_prune_s, (int, float)):
        raise ConfigValidationError("tracker.inactive_track_prune_s must be a number")
    cfg.tracker.track_connect_min_overlap = tracker_cfg_dict.get(
        "track_connect_min_overlap", cfg.tracker.track_connect_min_overlap
    )
    if not isinstance(cfg.tracker.track_connect_min_overlap, (int, float)):
        raise ConfigValidationError(
            "tracker.track_connect_min_overlap must be a number"
        )
    if (
        cfg.tracker.track_connect_min_overlap < 0
        or cfg.tracker.track_connect_min_overlap > 1
    ):
        raise ConfigValidationError(
            "tracker.track_connect_min_overlap must be in [0, 1]"
        )

    # notification criteria:
    notify_crit_dict = cfg_dict.get("notification_criteria", {})
    cfg.tracker.notify_classification_allowlist = notify_crit_dict.get(
        "classification_allowlist", cfg.tracker.notify_classification_allowlist
    )
    if cfg.tracker.notify_classification_allowlist is not None:
        if not isinstance(cfg.tracker.notify_classification_allowlist, list):
            raise ConfigValidationError(
                "notification_criteria.classification_allowlist must be a list"
            )
        for item in cfg.tracker.notify_classification_allowlist:
            if not isinstance(item, str):
                raise ConfigValidationError(
                    "notification_criteria.classification_allowlist items must be strings"
                )
    cfg.tracker.notify_classification_blocklist = notify_crit_dict.get(
        "classification_blocklist", cfg.tracker.notify_classification_blocklist
    )
    if cfg.tracker.notify_classification_blocklist is not None:
        if not isinstance(cfg.tracker.notify_classification_blocklist, list):
            raise ConfigValidationError(
                "notification_criteria.classification_blocklist must be a list"
            )
        for item in cfg.tracker.notify_classification_blocklist:
            if not isinstance(item, str):
                raise ConfigValidationError(
                    "notification_criteria.classification_blocklist items must be strings"
                )
    cfg.tracker.notify_min_track_length_s = notify_crit_dict.get(
        "min_track_length_s", cfg.tracker.notify_min_track_length_s
    )
    if not isinstance(cfg.tracker.notify_min_track_length_s, (int, float)):
        raise ConfigValidationError(
            "notification_criteria.min_track_length_s must be a number"
        )
    cfg.tracker.notify_min_track_length_s_per_classification = notify_crit_dict.get(
        "min_track_length_s_per_classification",
        cfg.tracker.notify_min_track_length_s_per_classification,
    )
    if not isinstance(cfg.tracker.notify_min_track_length_s_per_classification, dict):
        raise ConfigValidationError(
            "notification_criteria.min_track_length_s_per_classification must be a dict"
        )
    for k, v in cfg.tracker.notify_min_track_length_s_per_classification.items():
        if not isinstance(k, str) or not isinstance(v, (int, float)):
            raise ConfigValidationError(
                "notification_criteria.min_track_length_s_per_classification must be a dict of str -> number"
            )
    cfg.tracker.notify_track_cel = notify_crit_dict.get(
        "track_cel", cfg.tracker.notify_track_cel
    )
    if cfg.tracker.notify_track_cel is not None and not isinstance(
        cfg.tracker.notify_track_cel, str
    ):
        raise ConfigValidationError(
            "notification_criteria.track_cel must be a string representing a valid CEL expression (see https://cel.dev )"
        )

    # web:
    web_dict = cfg_dict.get("web", {})
    cfg.notifier.external_base_url = web_dict.get(
        "external_base_url", cfg.notifier.external_base_url
    )
    if not isinstance(cfg.notifier.external_base_url, str):
        raise ConfigValidationError("web.external_base_url must be a string")
    # noinspection HttpUrlsUsage
    if not (
        cfg.notifier.external_base_url.casefold().startswith("http://")
        or cfg.notifier.external_base_url.casefold().startswith("https://")
    ):
        # noinspection HttpUrlsUsage
        raise ConfigValidationError(
            "web.external_base_url must start with http:// or https://"
        )
    cfg.web.port = web_dict.get("port", cfg.web.port)
    if not isinstance(cfg.web.port, int):
        raise ConfigValidationError("web.port must be an int")
    cfg.web.bind_to = web_dict.get("bind_to", cfg.web.bind_to)
    if cfg.web.bind_to is not None and not isinstance(cfg.web.bind_to, str):
        raise ConfigValidationError("web.bind_to must be a string")
    # noinspection HttpUrlsUsage
    if cfg.notifier.external_base_url.casefold().startswith("http://"):
        logger.warning(
            "ntfy actions (e.g. mute options) on a non-HTTPS server "
            "will not work with web interface or iOS app! consider "
            "using HTTPS for web.external_base_url."
        )

    # health:
    cfg.health_pinger.req_timeout_s = int(cfg.model.liveness_tick_s - 1.0)

    # enrichment:
    enrichment_dict = cfg_dict.get("enrichment", {})
    cfg.notifier.enrichment.enable = enrichment_dict.get(
        "enable", cfg.notifier.enrichment.enable
    )
    if not isinstance(cfg.notifier.enrichment.enable, bool):
        raise ConfigValidationError("enrichment.enable must be a bool")
    if cfg.notifier.enrichment.enable:
        cfg.notifier.enrichment.prompt_files = enrichment_dict.get(
            "prompt_files", cfg.notifier.enrichment.prompt_files
        )
        if not isinstance(cfg.notifier.enrichment.prompt_files, dict):
            raise ConfigValidationError("enrichment.prompt_files must be a dict")
        for k, v in cfg.notifier.enrichment.prompt_files.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ConfigValidationError(
                    "enrichment.prompt_files must be a dict of str -> str"
                )
            try:
                with open(v) as f:
                    f.read()
            except Exception as e:
                raise ConfigValidationError(
                    f"enrichment.prompt_files: error reading file '{v}': {e}"
                )
        cfg.notifier.enrichment.endpoint = enrichment_dict.get(
            "endpoint", cfg.notifier.enrichment.endpoint
        )
        if not cfg.notifier.enrichment.endpoint or not isinstance(
            cfg.notifier.enrichment.endpoint, str
        ):
            raise ConfigValidationError("enrichment.endpoint must be a string")
        if not (
            cfg.notifier.enrichment.endpoint.casefold().startswith("http://")
            or cfg.notifier.enrichment.endpoint.casefold().startswith("https://")
        ):
            # noinspection HttpUrlsUsage
            raise ConfigValidationError(
                "enrichment.endpoint must start with http:// or https://"
            )
        cfg.notifier.enrichment.model = enrichment_dict.get(
            "model", cfg.notifier.enrichment.model
        )
        if not isinstance(cfg.notifier.enrichment.model, str):
            raise ConfigValidationError("enrichment.model must be a string")
        cfg.notifier.enrichment.timeout_s = enrichment_dict.get(
            "timeout_s", cfg.notifier.enrichment.timeout_s
        )
        if not isinstance(cfg.notifier.enrichment.timeout_s, (int, float)):
            raise ConfigValidationError("enrichment.timeout_s must be a number")
        cfg.notifier.enrichment.keep_alive = enrichment_dict.get(
            "keep_alive", cfg.notifier.enrichment.keep_alive
        )
        if not isinstance(cfg.notifier.enrichment.keep_alive, str):
            raise ConfigValidationError("enrichment.keep_alive must be a str")

    logger.info("config loaded & validated")
    return cfg
