import logging
import multiprocessing
from dataclasses import dataclass

import requests

import lib_mpex
from log import LOG_DEFAULT_FMT


@dataclass
class HealthPing:
    url: str


@dataclass
class HealthPingerConfig:
    log_level: int = logging.INFO
    req_timeout_s: int = 30


class HealthPinger(lib_mpex.ChildProcess):
    def __init__(
        self,
        config: HealthPingerConfig,
        input_queue: multiprocessing.Queue,  # of HealthPing
    ):
        self._config = config
        self._input_queue = input_queue

    def _run(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=self._config.log_level, format=LOG_DEFAULT_FMT)
        logger.info("starting health pinger")

        while True:
            p: HealthPing = self._input_queue.get()
            logger.debug(f"processing ping request for {p.url}")
            try:
                requests.get(p.url, timeout=self._config.req_timeout_s)
            except requests.RequestException as e:
                logger.error(f"error pinging {p.url}: {e}")
