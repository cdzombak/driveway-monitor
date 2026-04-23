import sys
import tempfile
import types
from pathlib import Path
from unittest import TestCase


waitress_stub = types.ModuleType("waitress")
waitress_stub.serve = lambda *args, **kwargs: None
sys.modules.setdefault("waitress", waitress_stub)

flask_stub = types.ModuleType("flask")
flask_stub.Flask = object
flask_stub.jsonify = lambda *args, **kwargs: None
flask_stub.request = object()
flask_stub.make_response = lambda *args, **kwargs: None
sys.modules.setdefault("flask", flask_stub)

flask_cors_stub = types.ModuleType("flask_cors")
flask_cors_stub.CORS = lambda *args, **kwargs: None
sys.modules.setdefault("flask_cors", flask_cors_stub)

from config import config_from_file
from ntfy import NtfyPriority


class TestConfig(TestCase):
    def test_missing_default_priority_keeps_enum_default(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            f.write(
                """
                {
                  "notifier": {
                    "topic": "driveway-monitor"
                  }
                }
                """
            )
            f.flush()

            cfg = config_from_file(f.name)

        self.assertEqual(NtfyPriority.DEFAULT, cfg.notifier.default_priority)

    def test_example_config_loads_without_default_priority(self):
        cfg = config_from_file(
            str(Path(__file__).resolve().parent.parent / "config.example.json")
        )

        self.assertEqual(NtfyPriority.DEFAULT, cfg.notifier.default_priority)
