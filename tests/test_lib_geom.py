from unittest import TestCase

from lib_geom import Point, Vector, Box


class TestVector(TestCase):
    def test_from_points(self):
        v = Vector.from_points(Point(0, 0), Point(0, 0))  # no movement
        self.assertEqual(Vector(length=0, direction=0), v)

        v = Vector.from_points(Point(0, 1), Point(0, 0))  # straight up
        self.assertEqual(Vector(length=1, direction=90), v)

        v = Vector.from_points(Point(0, 0), Point(0, 1))  # straight down
        self.assertEqual(Vector(length=1, direction=-90), v)

        v = Vector.from_points(Point(1, 0), Point(0, 0))  # straight left
        self.assertEqual(Vector(length=1, direction=-180), v)

        v = Vector.from_points(Point(0, 0), Point(1, 0))  # straight right
        self.assertEqual(Vector(length=1, direction=0), v)

        v = Vector.from_points(Point(1, 2), Point(0, 1))  # up and to the left
        self.assertEqual(135, v.direction)
        self.assertTrue(float_eq(1.414, v.length), f"(got {v.length}")

        v = Vector.from_points(Point(0, 2), Point(1, 1))  # up and to the right
        self.assertEqual(45, v.direction)
        self.assertTrue(float_eq(1.414, v.length), f"(got {v.length}")

        v = Vector.from_points(Point(2, 1), Point(1, 2))  # down and to the left
        self.assertEqual(-135, v.direction)
        self.assertTrue(float_eq(1.414, v.length), f"(got {v.length}")

        v = Vector.from_points(Point(1, 1), Point(2, 2))  # down and to the right
        self.assertEqual(-45, v.direction)
        self.assertTrue(float_eq(1.414, v.length), f"(got {v.length}")


def float_eq(a: float, b: float, tolerance: float = 0.01) -> bool:
    return abs(a - b) < tolerance


class TestPoint(TestCase):
    def test_distance_to(self):
        a = Point(2, 2)
        b = Point(0, 0)
        dist = a.distance_to(b)
        self.assertTrue(float_eq(2.828, dist), f"(got {dist}")


class TestBox(TestCase):
    def test_w(self):
        sut = Box(
            a=Point(0, 0),
            b=Point(1, 3.5),
        )
        self.assertEqual(1, sut.w())

    def test_h(self):
        sut = Box(
            a=Point(0, 0),
            b=Point(3.5, 1),
        )
        self.assertEqual(1, sut.h())

    def test_center(self):
        sut = Box(
            a=Point(0, 0),
            b=Point(2, 2),
        )
        self.assertEqual(Point(1, 1), sut.center())

    def test_area(self):
        sut = Box(
            a=Point(0, 0),
            b=Point(2, 2),
        )
        self.assertEqual(4, sut.area())

    def test_percent_intersection_with(self):
        a = Box(
            a=Point(0, 0),
            b=Point(2, 3),
        )
        b = Box(
            a=Point(1, 0),
            b=Point(3, 2),
        )
        self.assertEqual(0.25, a.percent_intersection_with(b))
        self.assertEqual(0.25, b.percent_intersection_with(a))

    def test_average_with(self):
        a = Box(
            a=Point(0, 0),
            b=Point(2, 2),
        )
        b = Box(
            a=Point(1, 1),
            b=Point(3, 3),
        )
        avg = a.average_with(b)
        self.assertEqual(Point(0.5, 0.5), avg.a)
        self.assertEqual(Point(2.5, 2.5), avg.b)
        avg = b.average_with(a)
        self.assertEqual(Point(0.5, 0.5), avg.a)
        self.assertEqual(Point(2.5, 2.5), avg.b)
