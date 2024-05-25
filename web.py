import datetime
import json
import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import Optional, Dict

import waitress
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS

import lib_mpex
from ntfy import NtfyRecord, FeedbackNotification, FeedbackType


@dataclass
class WebConfig:
    log_level: Optional[int] = logging.INFO
    port: int = 5550
    bind_to: str = "*"


class WebServer(lib_mpex.ChildProcess):
    def __init__(
        self,
        config: WebConfig,
        ntfy_share_ns,
        ntfy_records: Dict[str, NtfyRecord],
        ntfy_queue: multiprocessing.Queue,
    ):
        self._config = config
        self._ntfy_share_ns = ntfy_share_ns
        self._ntfy_records = ntfy_records
        self._ntfy_queue = ntfy_queue

    def _run(self):
        logging.getLogger("waitress").setLevel(self._config.log_level + 10)
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=self._config.log_level)
        logger.info("configuring web server")

        app = Flask(__name__)
        CORS(app)

        @app.route("/health", methods=["GET"])
        def health():
            logger.debug(f"{request.remote_addr} {request.method} {request.path}")
            return jsonify({"status": "ok"})

        @app.route("/mute", methods=["POST"])
        def mute():
            data = request.get_json()
            logger.info(
                f"{request.remote_addr} {request.method} {request.path} {json.dumps(data)}"
            )

            key = data.get("key", None)
            if not key:
                return jsonify({"error": "missing 'key' field"}), 403
            if key not in self._ntfy_records:
                return jsonify({"error": "bad key"}), 403

            secs = data.get("s", None)
            if secs is None:
                return jsonify({"error": "missing 's' field"}), 400
            try:
                secs = int(secs)
            except ValueError:
                return jsonify({"error": "invalid 's' field"}), 400

            utcnow = datetime.datetime.now(datetime.UTC)
            if secs < 1:
                mute_until = utcnow
                resp_ntfy = FeedbackNotification(
                    type=FeedbackType.UNMUTED,
                    key=key,
                )
            else:
                mute_until = utcnow + datetime.timedelta(
                    seconds=secs,
                )
                resp_ntfy = FeedbackNotification(
                    type=FeedbackType.MUTED,
                    key=key,
                    mute_seconds=secs,
                )

            self._ntfy_share_ns.mute_until = mute_until
            logger.info(f"muted until {mute_until}")
            self._ntfy_queue.put_nowait(resp_ntfy)
            return jsonify({"status": "ok"})

        @app.route("/photo/<fname>", methods=["GET"])
        def photo(fname):
            logger.info(f"{request.remote_addr} {request.method} {request.path}")

            if not fname.endswith(".jpg"):
                return jsonify({"error": "invalid filename"}), 400

            key = fname[:-4]
            photo_rec: Optional[NtfyRecord] = self._ntfy_records.get(key, None)
            if photo_rec is None:
                return jsonify({"error": "record not found"}), 404
            if photo_rec.jpeg_image is None:
                return jsonify({"error": "record has no photo"}), 404

            response = make_response(photo_rec.jpeg_image)
            response.headers.set("Content-Type", "image/jpeg")
            return response

        logger.info("starting web server")
        listen = f"{self._config.bind_to}:{self._config.port}"
        dm_docker_also_bind = os.getenv("DM_DOCKER_ALSO_BIND", None)
        if dm_docker_also_bind:
            listen = listen + " " + dm_docker_also_bind
        waitress.serve(app, listen=listen)
