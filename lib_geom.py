import math
from dataclasses import dataclass

# noinspection PyPackageRequirements
import celpy


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def to_cel(self) -> celpy.celtypes.Value:
        """
        Convert this object to a CEL value.
        :return:
        """
        return celpy.celtypes.MapType(
            {
                "x": celpy.celtypes.DoubleType(self.x),
                "y": celpy.celtypes.DoubleType(self.y),
            }
        )

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def vector_to(self, other: "Point") -> "Vector":
        return Vector.from_points(self, other)


@dataclass(frozen=True)
class Box:
    """
    Box represents a rectangle in an image.

    a is the upper-left point.
    b is the lower-right point.

    Origin is top-left of the image.
    """

    a: Point
    b: Point

    def w(self) -> float:
        return self.b.x - self.a.x

    def h(self) -> float:
        return self.b.y - self.a.y

    def center(self) -> Point:
        return Point(
            x=(self.a.x + self.b.x) / 2,
            y=(self.a.y + self.b.y) / 2,
        )

    def area(self) -> float:
        return self.w() * self.h()

    def to_cel(self) -> celpy.celtypes.Value:
        """
        Convert this object to a CEL value.
        :return:
        """
        return celpy.celtypes.MapType(
            {
                "a": self.a.to_cel(),
                "b": self.b.to_cel(),
                "w": celpy.celtypes.DoubleType(self.w()),
                "h": celpy.celtypes.DoubleType(self.h()),
                "center": self.center().to_cel(),
                "area": celpy.celtypes.DoubleType(self.area()),
            }
        )

    def percent_intersection_with(self, other: "Box") -> float:
        self_a = (self.b.x - self.a.x) * (self.b.y - self.a.y)
        other_a = (other.b.x - other.a.x) * (other.b.y - other.a.y)
        i_a = max(0.0, min(self.b.x, other.b.x) - max(self.a.x, other.a.x)) * max(
            0.0, min(self.b.y, other.b.y) - max(self.a.y, other.a.y)
        )
        return i_a / (self_a + other_a - i_a)

    def average_with(self, other: "Box") -> "Box":
        return Box(
            a=Point(
                x=(self.a.x + other.a.x) / 2,
                y=(self.a.y + other.a.y) / 2,
            ),
            b=Point(
                x=(self.b.x + other.b.x) / 2,
                y=(self.b.y + other.b.y) / 2,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class Vector:
    # direction is expressed in degrees, with:
    #        ┌───┐    ┌───┐    ┌───┐
    #        │ b │    │ b │    │ b │
    #        └─▲─┘    └─▲─┘    └─▲─┘
    #           ╲       │       ╱
    #          135º   90º    45º
    #             ╲     │     ╱
    #              ╲    │    ╱
    #               ╳───┴───╳
    # ┌───┐         │       │         ┌───┐
    # │ b ◀─-180º───┤   a   │───0º────▶ b │
    # └───┘         │       │         └───┘
    #               ╳───┬───╳
    #              ╱    │    ╲
    #             ╱   -90º    ╲
    #         -135º     │    -45º
    #           ╱       │       ╲
    #        ┌─▼─┐    ┌─▼─┐    ┌─▼─┐
    #        │ b │    │ b │    │ b │
    #        └───┘    └───┘    └───┘
    direction: float
    length: float

    def to_cel(self) -> celpy.celtypes.Value:
        """
        Convert this object to a CEL value.
        :return:
        """
        return celpy.celtypes.MapType(
            {
                "direction": celpy.celtypes.DoubleType(self.direction),
                "length": celpy.celtypes.DoubleType(self.length),
            }
        )

    @staticmethod
    def from_points(a: Point, b: Point) -> "Vector":
        dx = b.x - a.x
        dy = b.y - a.y
        return Vector(
            length=(dx**2 + dy**2) ** 0.5,
            direction=-1 * math.degrees(math.atan2(dy, dx)),
        )
