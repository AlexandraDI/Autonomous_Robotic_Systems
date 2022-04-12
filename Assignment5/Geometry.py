"""
Set of funtion to compute the intersection between different geometrical
entities.

Implemented by Gianluca Vico
"""
from typing import Tuple, Optional, List, Union
import numpy as np

EPS = 0


def sphere_sphere_intersection(
    c1: np.array, r1: float, c2: np.array, r2: float
) -> List[np.array]:
    """
    Intersection of 2 non-overlapping circles

    Args:
        c1: centre of the first circle (x, y)
        r1: radius of the first circle
        c2: centre of the second circle (x, y)
        r2: radius of the second circle
    Returns:
        List of intersection points
    """
    l = c2 - c1  # segment between 2 centres
    d = np.linalg.norm(l)

    if d > (r1 + r2) or (c1 == c2).all():
        # Ignore the case when the circles are the same
        return []

    x = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    y = np.sqrt(r1 ** 2 - x ** 2)
    proj = c1 + x * normalize(l)  # projection of the intersection on l

    if np.allclose(y, 0):  # 1 intersection
        return [proj]
    else:
        # 2 intersections
        ortho = y * orthogonal(l)
        return [proj + ortho, proj - ortho]


def sphere_segment_intersection(
    centre: np.array, radius: float, line: Tuple[np.array, np.array]
) -> List:
    """
    Intersection between a sphere and a segment

    Args:
        centre: centre of the sphere
        radius: radius of the sphere
        line: end points of the line

    Returns:
        List of intersecting points
    """
    # Implementation from
    # http://paulbourke.net/geometry/circlesphere/index.html#linesphere

    l = (line[1] - line[0]).reshape(-1)

    a = np.dot(l, l)  # squared length
    b = 2 * np.dot(l, line[0] - centre)
    c = np.dot(centre.reshape(-1), centre.reshape(-1))
    c += np.dot(line[0].reshape(-1), line[0].reshape(-1))
    c -= 2 * np.dot(centre, line[0])
    c -= radius ** 2

    intersection = b * b - 4 * a * c
    if np.abs(a) < EPS or intersection < 0:  # no intersections
        return []

    if intersection == 0 or intersection < EPS:  # one solution
        return [line[0] + l * (-b / (2 * a))]

    # Parameter for the intersection
    t = [(-b + np.sqrt(intersection)) / (2 * a), (-b - np.sqrt(intersection)) / (2 * a)]

    # If t is not in [0, 1] the intersection is outside the segment
    t = [i for i in t if i >= 0 and i <= 1]
    return [line[0] + l * i for i in t]


def ray_segment_intersection(
    origin: np.array, direction: np.array, line: Tuple[np.array, np.array]
) -> Optional[np.array]:
    """
    Compute the intersection between a line and a ray

    Args:
        origin: origin of the ray as np.array
        direction: direction of the ray as normalized np.array
        line: tuple containing the end points of the segment

    Returns:
        Intersection point if any

    """
    point, t1, t2 = line_line_intersection(line, (origin, origin + direction), True)

    if point is None:
        return None

    if t2 >= 0 and t1 >= 0 and t1 <= 1:
        return line[0] + t1 * (line[1] - line[0])
    else:
        return None


def ray_line_intersection(
    origin: np.array, direction: np.array, line: Tuple[np.array, np.array]
) -> Optional[np.array]:
    """
    Compute the intersection between a line and a ray

    Args:
        origin: origin of the ray as np.array
        direction: direction of the ray as normalized np.array
        line: tuple containing the end points of the segment

    Returns:
        Intersection point if any

    """
    point, t1, t2 = line_line_intersection(line, (origin, origin + direction), True)

    if point is None:
        return None

    if t2 >= 0:
        return line[0] + t1 * (line[1] - line[0])
    else:
        return None


def segment_segment_intersection(
    line1: Tuple[np.array, np.array], line2: Tuple[np.array, np.array]
) -> Optional[np.array]:
    """
    Intersection between two segments.
    The segments are defined by their end-points

    Args:
        line1: first segment
        line2: second segment

    Returns:
        Intersection point if any.

    """
    point, t1, t2 = line_line_intersection(line1, line2, True)

    if point is None:
        # print("Parallel lines:", line1, line2)
        return None

    if t1 >= 0 and t2 >= 0 and t1 <= 1 and t2 <= 1:
        return line1[0] + t1 * (line1[1] - line1[0])
    else:
        return None


def line_line_intersection(
    line1: Tuple[np.array, np.array],
    line2: Tuple[np.array, np.array],
    return_parameters: bool = False,
) -> Union[Tuple[Optional[np.array], Tuple[float, float]], Optional[np.array]]:
    """
    Compute the intersection of two lines

    Args:
        line1: tuple containing the two points on the line
        line2: tuple containing the two points on the line
        return_parameters: if true this function returns the parameters to find
            the point on the lines given the direction and the first point of
            the line.

    Returns:
        Intersection point. None if there is no intersection.
        If return_parameters is True, it alse returns the parameters to find the
        point on the lines.

    """
    point = None
    t1 = 0
    t2 = 0

    # Formulation from https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    l1 = line1[1] - line1[0]  # Segment vector
    l2 = line2[1] - line2[0]
    s = line2[0] - line1[0]
    # (x1 - x2) (y3 - y4) - (y1 - y2) (x3 - x4)
    # 1 and 2 are on line1
    # 3 and 4 are on line2
    i = l1[0] * l2[1] - l1[1] * l2[0]

    if i != 0:
        # (x1 - x3) (y3 - y4) - (y1 - y3) (x3 - x4)
        t1 = s[0] * l2[1] - s[1] * l2[0]

        # (x1 - x3) (y1 - y2) - (y1 - y3) (x1 - x2)
        t2 = s[0] * l1[1] - s[1] * l1[0]
        t1 /= i
        t2 /= i
        point = line1[0] + t1 * l1

    if return_parameters:
        return point, t1, t2
    else:
        return point


def capsule_segment_intersection(
    centre1: np.array, centre2: np.array, radius: float, line: Tuple[np.array, np.array]
) -> List[np.array]:
    """
    Compute the intersection between a capsule and a line

    Args:
        centre1: centre of the circul side of the capsule
        centre2: centre of the other circul side of the capsule
        radius: radius of the capsule
        line: line intersecting the capsule

    Returns:
        list of intersection points

    Note:
        This method might retun points inside the capsule and not on the surface
    """
    # TODO remove internal collision
    direction = centre2 - centre1
    points = []

    # This might return points inside the capsule and not on the surface
    points.extend(sphere_segment_intersection(centre1, radius, line))
    points.extend(sphere_segment_intersection(centre2, radius, line))

    # Compute the two side lines
    if (centre2 != centre1).all():
        ortho = orthogonal(direction)

        side = (centre1 + ortho, centre2 + ortho)
        intersection = segment_segment_intersection(side, line)
        if intersection is not None:
            points.append(intersection)

        side = (centre1 - ortho, centre2 - ortho)
        intersection = segment_segment_intersection(side, line)
        if intersection is not None:
            points.append(intersection)

    return points


def dist(p1: np.array, p2: np.array) -> float:
    """
    Args:
        p1: first point
        p2: second point

    Returns:
        Distance between 2 points

    """
    return np.linalg.norm(p2 - p1)


def normalize(v: np.array) -> np.array:
    """
    Args:
        v: input array

    Returns:
        Normalized vector. Same direction of the indput but with magnitude 1

    """
    norm = np.linalg.norm(v)
    if norm == 0:
        # print("Zero vector")
        return v
    return v / norm


def orthogonal(v: np.array) -> np.array:
    """
    Args:
        v: input array

    Returns:
        Perpendicular unit vector
    """
    return normalize(np.flip(v) * np.array([1, -1]))


def ray_point_angle(o: np.array, p: np.array, r_angle: float) -> float:
    """
    Compute the anle between a ray and a point

    Args:
        o: origin of the ray
        p: point
        r_angle: angle of the ray in radiants

    Returns:
        The angle in radiants between a ray and a point

    """
    d = p - o  # segment between the ray and the vector
    return np.arctan2(d[1], d[0]) - r_angle
