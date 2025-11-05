import datetime
from unittest import TestCase

import numpy as np

from lib_geom import Point, Box, Vector
from track import TrackPrediction, Track


class TestTrackPrediction(TestCase):
    def test_from_prediction_creates_track(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        pred = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=True,
            box=Box(a=Point(0.1, 0.1), b=Point(0.5, 0.5)),
            image=image,
        )

        track = Track.from_prediction(pred)

        self.assertEqual(1, len(track.predictions))
        self.assertEqual(pred, track.predictions[0])
        self.assertTrue(track.is_model_track)
        self.assertFalse(track.triggered_notification)
        self.assertEqual(pred.id, track.id)


class TestTrack(TestCase):
    def setUp(self):
        self.image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.image3 = np.zeros((100, 100, 3), dtype=np.uint8)

        self.pred1 = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=True,
            box=Box(a=Point(0.1, 0.1), b=Point(0.3, 0.3)),
            image=self.image1,
        )

        self.pred2 = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 1, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=True,
            box=Box(a=Point(0.2, 0.2), b=Point(0.4, 0.4)),
            image=self.image2,
        )

        self.pred3 = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 2, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=True,
            box=Box(a=Point(0.3, 0.3), b=Point(0.5, 0.5)),
            image=self.image3,
        )

    def test_first_t(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        self.assertEqual(
            datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            track.first_t(),
        )

    def test_last_t(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        self.assertEqual(
            datetime.datetime(2024, 1, 1, 12, 0, 2, tzinfo=datetime.UTC), track.last_t()
        )

    def test_first_box(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        self.assertEqual(Box(a=Point(0.1, 0.1), b=Point(0.3, 0.3)), track.first_box())

    def test_last_box(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        self.assertEqual(Box(a=Point(0.3, 0.3), b=Point(0.5, 0.5)), track.last_box())

    def test_average_box(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        avg_box = track.average_box()

        self.assertAlmostEqual(0.2, avg_box.a.x, places=5)
        self.assertAlmostEqual(0.2, avg_box.a.y, places=5)
        self.assertAlmostEqual(0.4, avg_box.b.x, places=5)
        self.assertAlmostEqual(0.4, avg_box.b.y, places=5)

    def test_total_box(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        total_box = track.total_box()

        self.assertEqual(0.1, total_box.a.x)
        self.assertEqual(0.1, total_box.a.y)
        self.assertEqual(0.5, total_box.b.x)
        self.assertEqual(0.5, total_box.b.y)

    def test_classification_single_type(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        self.assertEqual("car", track.classification())

    def test_classification_multiple_types_majority_wins(self):
        pred_truck = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 3, tzinfo=datetime.UTC),
            model_id=1,
            classification="truck",
            is_track=True,
            box=Box(a=Point(0.4, 0.4), b=Point(0.6, 0.6)),
            image=self.image1,
        )

        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)
        track.add_prediction(pred_truck)

        self.assertEqual("car", track.classification())

    def test_length_t(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        self.assertEqual(datetime.timedelta(seconds=2), track.length_t())

    def test_length_t_single_prediction(self):
        track = Track.from_prediction(self.pred1)

        self.assertEqual(datetime.timedelta(seconds=0), track.length_t())

    def test_movement_vector(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        vector = track.movement_vector()

        self.assertIsInstance(vector, Vector)
        self.assertGreater(vector.length, 0)

    def test_last_2_box_avg_with_multiple_predictions(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)
        track.add_prediction(self.pred3)

        avg_box = track.last_2_box_avg()

        expected = self.pred2.box.average_with(self.pred3.box)
        self.assertEqual(expected.a.x, avg_box.a.x)
        self.assertEqual(expected.a.y, avg_box.a.y)
        self.assertEqual(expected.b.x, avg_box.b.x)
        self.assertEqual(expected.b.y, avg_box.b.y)

    def test_last_2_box_avg_with_single_prediction(self):
        track = Track.from_prediction(self.pred1)

        avg_box = track.last_2_box_avg()

        self.assertEqual(self.pred1.box, avg_box)

    def test_add_prediction_appends_to_list(self):
        track = Track.from_prediction(self.pred1)

        self.assertEqual(1, len(track.predictions))

        track.add_prediction(self.pred2)

        self.assertEqual(2, len(track.predictions))
        self.assertEqual(self.pred2, track.predictions[1])

    def test_add_prediction_updates_is_model_track(self):
        pred_not_track = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=False,
            box=Box(a=Point(0.1, 0.1), b=Point(0.3, 0.3)),
            image=self.image1,
        )

        track = Track.from_prediction(pred_not_track)
        self.assertFalse(track.is_model_track)

        track.add_prediction(self.pred1)
        self.assertTrue(track.is_model_track)

    def test_add_prediction_updates_best_image_when_coverage_increases(self):
        small_box_pred = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=True,
            box=Box(a=Point(0.1, 0.1), b=Point(0.2, 0.2)),
            image=self.image1,
        )

        large_box_pred = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 1, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=True,
            box=Box(a=Point(0.1, 0.1), b=Point(0.9, 0.9)),
            image=self.image2,
        )

        track = Track.from_prediction(small_box_pred)
        self.assertTrue(np.array_equal(self.image1, track.best_image))

        track.add_prediction(large_box_pred)
        self.assertTrue(np.array_equal(self.image2, track.best_image))

    def test_add_prediction_keeps_best_image_when_coverage_decreases(self):
        large_box_pred = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=True,
            box=Box(a=Point(0.1, 0.1), b=Point(0.9, 0.9)),
            image=self.image1,
        )

        small_box_pred = TrackPrediction(
            t=datetime.datetime(2024, 1, 1, 12, 0, 1, tzinfo=datetime.UTC),
            model_id=1,
            classification="car",
            is_track=True,
            box=Box(a=Point(0.1, 0.1), b=Point(0.2, 0.2)),
            image=self.image2,
        )

        track = Track.from_prediction(large_box_pred)
        self.assertTrue(np.array_equal(self.image1, track.best_image))

        track.add_prediction(small_box_pred)
        self.assertTrue(np.array_equal(self.image1, track.best_image))

    def test_to_cel_returns_cel_value(self):
        track = Track.from_prediction(self.pred1)
        track.add_prediction(self.pred2)

        cel_value = track.to_cel()

        self.assertIsNotNone(cel_value)
