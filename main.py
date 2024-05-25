import argparse
import logging
import multiprocessing
import sys
import traceback

import lib_mpex
from config import config_from_file
from health import HealthPinger
from ntfy import Notifier, print_notifier
from track import PredModel, Tracker
from web import WebServer


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
    logging.basicConfig(level=ll)

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
    health_pinger = HealthPinger(config.health_pinger, health_ping_queue)
    health_pinger_proc = multiprocessing.Process(
        target=health_pinger.run, args=(exit_queue,)
    )
    ws = WebServer(
        config.web, ntfy_web_share_ns, ntfy_web_records_dict, notifications_queue
    )
    ws_proc = multiprocessing.Process(target=ws.run, args=(exit_queue,))

    def my_exit(error: bool):
        logger.debug(f"exiting ({'success' if not error else 'with error'}) ...")
        model_proc.terminate()
        tracker_proc.terminate()
        notifier_proc.terminate()
        health_pinger_proc.terminate()
        ws_proc.terminate()
        sys.exit(1 if error else 0)

    logger.info("starting child processes ...")
    model_proc.start()
    tracker_proc.start()
    notifier_proc.start()
    health_pinger_proc.start()
    ws_proc.start()

    while (
        model_proc.is_alive()
        or tracker_proc.is_alive()
        or notifier_proc.is_alive()
        or health_pinger_proc.is_alive()
        or ws_proc.is_alive()
    ):
        e: lib_mpex.ChildExit = exit_queue.get()
        if e.is_exc():
            logger.error(f"{e.exc_info[0]} {e.exc_info[1]}")
            logger.error(f"Error in in {e.class_name} (pid {e.pid}): {e.error}")
            traceback.print_exception(*e.exc_info)
            my_exit(True)
        else:
            logger.info(f"{e.class_name} (pid {e.pid}) exited: {e.error}")
            my_exit(False)

    model_proc.join()
    tracker_proc.join()
    notifier_proc.join()
    health_pinger_proc.join()
    ws_proc.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
