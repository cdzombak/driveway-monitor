import argparse
import logging
import multiprocessing
import queue
import signal
import sys
import traceback
from typing import Final

import lib_mpex
from config import config_from_file
from health import HealthPinger
from log import LOG_DEFAULT_FMT
from ntfy import Notifier, print_notifier
from track import PredModel, Tracker
from web import WebServer

CHILD_CHECK_INTERVAL_S: Final = 5.0


def main():
    parser = argparse.ArgumentParser(prog="driveway-monitor")
    parser.add_argument("--config", type=str, help="Path to the JSON config file.")
    parser.add_argument(
        "--debug", action="store_true", help="Print debug-level logs (to stderr)."
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print notifications to stdout; disable ntfy.",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to the video file or RTSP stream to process.",
        required=True,
    )
    args = parser.parse_args()

    logger = logging.getLogger("main")
    ll = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=ll, format=LOG_DEFAULT_FMT)

    if sys.version_info < (3, 12):
        logger.error("Python 3.12 or newer is required.")
        sys.exit(1)

    config = config_from_file(args.config)
    config.model.log_level = ll
    config.notifier.log_level = ll
    config.tracker.log_level = ll
    config.health_pinger.log_level = ll
    config.web.log_level = ll

    tracks_queue = multiprocessing.Queue()
    notifications_queue = multiprocessing.Queue()
    exit_queue = multiprocessing.Queue()
    health_ping_queue = multiprocessing.Queue()
    ntfy_web_share_manager = multiprocessing.Manager()
    ntfy_web_records_dict = ntfy_web_share_manager.dict()
    ntfy_web_share_ns = ntfy_web_share_manager.Namespace()
    health_share_manager = multiprocessing.Manager()
    health_share_ns = health_share_manager.Namespace()

    model = PredModel(args.video, config.model, tracks_queue, health_ping_queue)
    model_proc = multiprocessing.Process(target=model.run, args=(exit_queue,))
    tracker = Tracker(config.tracker, tracks_queue, notifications_queue)
    tracker_proc = multiprocessing.Process(target=tracker.run, args=(exit_queue,))
    if args.print:
        notifier_proc = multiprocessing.Process(
            target=print_notifier, args=(notifications_queue,)
        )
    else:
        notifier = Notifier(
            config.notifier,
            notifications_queue,
            ntfy_web_share_ns,
            ntfy_web_records_dict,
        )
        notifier_proc = multiprocessing.Process(target=notifier.run, args=(exit_queue,))
    health_pinger = HealthPinger(
        config.health_pinger, health_ping_queue, health_share_ns
    )
    health_pinger_proc = multiprocessing.Process(
        target=health_pinger.run, args=(exit_queue,)
    )
    ws = WebServer(
        config.web,
        ntfy_web_share_ns,
        ntfy_web_records_dict,
        notifications_queue,
        health_share_ns,
    )
    ws_proc = multiprocessing.Process(target=ws.run, args=(exit_queue,))

    procs = [model_proc, tracker_proc, notifier_proc, health_pinger_proc, ws_proc]

    logger.info("starting child processes ...")
    for p in procs:
        p.start()

    def handle_sigterm(signum, frame):
        logger.info("received SIGTERM; exiting ...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    # sys.exit anywhere below (including from handle_sigterm) unwinds through
    # the finally block, which stops the children on every exit path:
    try:
        supervise(logger, procs, exit_queue)
    except KeyboardInterrupt:
        logger.info("interrupted; exiting ...")
        sys.exit(130)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()


def supervise(
    logger: logging.Logger,
    procs: list[multiprocessing.Process],
    exit_queue: multiprocessing.Queue,
):
    while True:
        try:
            e: lib_mpex.ChildExit = exit_queue.get(timeout=CHILD_CHECK_INTERVAL_S)
        except queue.Empty:
            # a child killed hard (SIGKILL, segfault) never reports its exit,
            # so periodically check liveness instead of blocking forever:
            dead = [p for p in procs if not p.is_alive()]
            if not dead:
                continue
            for p in dead:
                logger.error(
                    f"child (pid {p.pid}) died unexpectedly (exit code {p.exitcode})"
                )
            sys.exit(1)
        else:
            if e.is_exc():
                logger.error(f"{e.exc_info[0]} {e.exc_info[1]}")
                logger.error(f"Error in {e.class_name} (pid {e.pid}): {e.error}")
                traceback.print_exception(*e.exc_info)
                sys.exit(1)
            else:
                logger.info(f"{e.class_name} (pid {e.pid}) exited: {e.error}")
                sys.exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
