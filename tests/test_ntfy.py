import datetime
from unittest import TestCase
from unittest.mock import Mock

from ntfy import (
    ImageAttachMethod,
    NtfyPriority,
    EnrichmentType,
    ObjectNotification,
    FeedbackNotification,
    FeedbackType,
    Notifier,
    NtfyConfig,
)


class TestImageAttachMethod(TestCase):
    def test_from_str_attach(self):
        result = ImageAttachMethod.from_str("attach")
        self.assertEqual(ImageAttachMethod.ATTACH, result)

    def test_from_str_click(self):
        result = ImageAttachMethod.from_str("click")
        self.assertEqual(ImageAttachMethod.CLICK, result)

    def test_from_str_case_insensitive(self):
        result = ImageAttachMethod.from_str("ATTACH")
        self.assertEqual(ImageAttachMethod.ATTACH, result)
        
        result = ImageAttachMethod.from_str("Click")
        self.assertEqual(ImageAttachMethod.CLICK, result)


class TestNtfyPriority(TestCase):
    def test_all_values(self):
        values = NtfyPriority.all_values()
        self.assertIsInstance(values, set)
        self.assertIn("1", values)
        self.assertIn("min", values)
        self.assertIn("default", values)
        self.assertIn("max", values)
        self.assertIn("urgent", values)

    def test_from_str_numeric(self):
        self.assertEqual(NtfyPriority.N_1, NtfyPriority.from_str("1"))
        self.assertEqual(NtfyPriority.N_3, NtfyPriority.from_str("3"))
        self.assertEqual(NtfyPriority.N_5, NtfyPriority.from_str("5"))

    def test_from_str_named(self):
        self.assertEqual(NtfyPriority.MIN, NtfyPriority.from_str("min"))
        self.assertEqual(NtfyPriority.LOW, NtfyPriority.from_str("low"))
        self.assertEqual(NtfyPriority.DEFAULT, NtfyPriority.from_str("default"))
        self.assertEqual(NtfyPriority.HIGH, NtfyPriority.from_str("high"))
        self.assertEqual(NtfyPriority.MAX, NtfyPriority.from_str("max"))
        self.assertEqual(NtfyPriority.URGENT, NtfyPriority.from_str("urgent"))

    def test_from_str_case_insensitive(self):
        self.assertEqual(NtfyPriority.DEFAULT, NtfyPriority.from_str("DEFAULT"))
        self.assertEqual(NtfyPriority.URGENT, NtfyPriority.from_str("Urgent"))


class TestEnrichmentType(TestCase):
    def test_from_str_ollama(self):
        result = EnrichmentType.from_str("ollama")
        self.assertEqual(EnrichmentType.OLLAMA, result)

    def test_from_str_openai(self):
        result = EnrichmentType.from_str("openai")
        self.assertEqual(EnrichmentType.OPENAI, result)

    def test_from_str_case_insensitive(self):
        result = EnrichmentType.from_str("OLLAMA")
        self.assertEqual(EnrichmentType.OLLAMA, result)
        
        result = EnrichmentType.from_str("OpenAI")
        self.assertEqual(EnrichmentType.OPENAI, result)


class TestObjectNotification(TestCase):
    def test_message_without_enrichment(self):
        notif = ObjectNotification(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            classification="car",
            event="arrived in driveway",
            id="test123",
            jpeg_image=None,
        )
        
        self.assertEqual("Car arrived in driveway", notif.message())

    def test_message_with_enrichment(self):
        notif = ObjectNotification(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            classification="car",
            event="arrived in driveway",
            id="test123",
            jpeg_image=None,
            enriched_class="red sedan",
        )
        
        self.assertEqual("Likely: red sedan.", notif.message())

    def test_title(self):
        notif = ObjectNotification(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            classification="person",
            event="detected",
            id="test456",
            jpeg_image=None,
        )
        
        self.assertEqual("Person detected", notif.title())

    def test_ntfy_tags_car(self):
        notif = ObjectNotification(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            classification="car",
            event="arrived",
            id="test",
            jpeg_image=None,
        )
        
        self.assertEqual("blue_car", notif.ntfy_tags())

    def test_ntfy_tags_truck(self):
        notif = ObjectNotification(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            classification="truck",
            event="arrived",
            id="test",
            jpeg_image=None,
        )
        
        self.assertEqual("truck", notif.ntfy_tags())

    def test_ntfy_tags_person(self):
        notif = ObjectNotification(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            classification="person",
            event="detected",
            id="test",
            jpeg_image=None,
        )
        
        self.assertEqual("walking", notif.ntfy_tags())

    def test_ntfy_tags_unknown(self):
        notif = ObjectNotification(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            classification="bicycle",
            event="detected",
            id="test",
            jpeg_image=None,
        )
        
        self.assertEqual("camera_flash", notif.ntfy_tags())


class TestFeedbackNotification(TestCase):
    def test_message_muted(self):
        notif = FeedbackNotification(
            type=FeedbackType.MUTED,
            key="test123",
        )
        
        self.assertEqual("Notifications muted.", notif.message())

    def test_message_unmuted(self):
        notif = FeedbackNotification(
            type=FeedbackType.UNMUTED,
            key="test123",
        )
        
        self.assertEqual("Notifications unmuted.", notif.message())

    def test_title(self):
        notif = FeedbackNotification(
            type=FeedbackType.MUTED,
            key="test123",
        )
        
        self.assertEqual("driveway-monitor", notif.title())

    def test_ntfy_tags_muted(self):
        notif = FeedbackNotification(
            type=FeedbackType.MUTED,
            key="test123",
        )
        
        self.assertEqual("mute", notif.ntfy_tags())

    def test_ntfy_tags_unmuted(self):
        notif = FeedbackNotification(
            type=FeedbackType.UNMUTED,
            key="test123",
        )
        
        self.assertEqual("loud_sound", notif.ntfy_tags())


class TestNotifierUtilityMethods(TestCase):
    def setUp(self):
        self.config = NtfyConfig(external_base_url="http://example.com:5550")
        mock_web_share_ns = Mock()
        mock_web_share_ns.mute_until = None
        self.notifier = Notifier(
            config=self.config,
            input_queue=None,
            web_share_ns=mock_web_share_ns,
            records_dict={},
        )

    def test_strip_markdown_fences_with_json_fence(self):
        input_str = "```json\n{\"key\": \"value\"}\n```"
        result = self.notifier._strip_markdown_fences(input_str)
        self.assertEqual('{"key": "value"}', result)

    def test_strip_markdown_fences_with_generic_fence(self):
        input_str = "```\n{\"key\": \"value\"}\n```"
        result = self.notifier._strip_markdown_fences(input_str)
        self.assertEqual('{"key": "value"}', result)

    def test_strip_markdown_fences_without_fence(self):
        input_str = '{"key": "value"}'
        result = self.notifier._strip_markdown_fences(input_str)
        self.assertEqual('{"key": "value"}', result)

    def test_strip_markdown_fences_with_whitespace(self):
        input_str = "  ```json\n{\"key\": \"value\"}\n```  "
        result = self.notifier._strip_markdown_fences(input_str)
        self.assertEqual('{"key": "value"}', result)

    def test_ntfy_mute_action_blob_10_minutes(self):
        result = self.notifier._ntfy_mute_action_blob(600, "key123")
        
        self.assertIn("Mute 10m", result)
        self.assertIn("http://example.com:5550/mute", result)
        self.assertIn('"s": 600', result)
        self.assertIn('"key": "key123"', result)

    def test_ntfy_mute_action_blob_1_hour(self):
        result = self.notifier._ntfy_mute_action_blob(3600, "key123")
        
        self.assertIn("Mute 1h", result)
        self.assertIn("http://example.com:5550/mute", result)
        self.assertIn('"s": 3600', result)

    def test_ntfy_mute_action_blob_4_hours(self):
        result = self.notifier._ntfy_mute_action_blob(14400, "key123")
        
        self.assertIn("Mute 4h", result)
        self.assertIn('"s": 14400', result)

    def test_ntfy_mute_action_blob_1_hour_30_minutes(self):
        result = self.notifier._ntfy_mute_action_blob(5400, "key123")
        
        self.assertIn("Mute 1h 30m", result)
        self.assertIn('"s": 5400', result)

    def test_ntfy_mute_action_blob_unmute(self):
        result = self.notifier._ntfy_mute_action_blob(0, "key123")
        
        self.assertIn("Unmute", result)
        self.assertNotIn("Mute", result.replace("Unmute", ""))
        self.assertIn('"s": 0', result)
