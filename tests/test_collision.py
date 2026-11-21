"""Tests for robot_arm.collision."""

import math

import pytest

from robot_arm.collision import (
    CircleObstacle,
    SphereObstacle,
    check_trajectory_clearance,
    link_clearance_2d,
    link_clearance_3d,
    segment_point_distance,
    segment_point_distance_2d,
)


def test_segment_point_distance_perpendicular():
    assert segment_point_distance((0.5, 0.3, 0.0), (0.0, 0.0, 0.0),
                                  (1.0, 0.0, 0.0)) == pytest.approx(0.3)


def test_segment_point_distance_beyond_end():
    assert segment_point_distance((2.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                  (1.0, 0.0, 0.0)) == pytest.approx(1.0)
    assert segment_point_distance((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                  (1.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_segment_point_distance_on_segment():
    assert segment_point_distance((0.5, 0.0, 0.0), (0.0, 0.0, 0.0),
                                  (1.0, 0.0, 0.0)) == pytest.approx(0.0)


def test_segment_point_distance_degenerate():
    assert segment_point_distance((1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                                  (0.0, 0.0, 0.0)) == pytest.approx(math.sqrt(3.0))


def test_segment_point_distance_2d():
    assert segment_point_distance_2d((0.5, 0.3), (0.0, 0.0),
                                     (1.0, 0.0)) == pytest.approx(0.3)
    assert segment_point_distance_2d((2.0, 0.0), (0.0, 0.0),
                                     (1.0, 0.0)) == pytest.approx(1.0)
    assert segment_point_distance_2d((0.5, 0.0), (0.0, 0.0),
                                     (1.0, 0.0)) == pytest.approx(0.0)
    assert segment_point_distance_2d((1.0, 1.0), (0.0, 0.0),
                                     (0.0, 0.0)) == pytest.approx(math.sqrt(2.0))


def test_link_clearance_3d():
    margin = link_clearance_3d(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [0.02],
        [SphereObstacle(center=(0.5, 0.3, 0.0), radius=0.1)])
    assert margin == pytest.approx(0.18)


def test_link_clearance_3d_empty_obstacles():
    assert link_clearance_3d([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [0.02],
                             []) == math.inf


def test_link_clearance_3d_min_over_links_and_obstacles():
    margin = link_clearance_3d(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        [0.02, 0.02],
        [SphereObstacle(center=(0.5, 0.3, 0.0), radius=0.1),
         SphereObstacle(center=(1.0, 0.5, 0.3), radius=0.1)])
    # Second obstacle is 0.3 from the second link: 0.3 - 0.1 - 0.02 = 0.18,
    # tying the first link's margin to the first obstacle.
    assert margin == pytest.approx(0.18)


def test_link_clearance_2d():
    margin = link_clearance_2d(
        [(0.0, 0.0), (1.0, 0.0)], [0.02],
        [CircleObstacle(center=(0.5, 0.3), radius=0.1)])
    assert margin == pytest.approx(0.18)
    assert link_clearance_2d([(0.0, 0.0), (1.0, 0.0)], [0.02], []) == math.inf


def test_check_trajectory_clearance():
    seq = [
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    ]
    obstacles = [SphereObstacle(center=(0.5, 0.3, 0.0), radius=0.1)]
    # Same geometry everywhere: all margins 0.18, worst index is the first.
    margin, worst = check_trajectory_clearance(seq, [0.02], obstacles)
    assert margin == pytest.approx(0.18)
    assert worst == 0

    seq2 = [
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        [(0.0, 0.5, 0.0), (1.0, 0.5, 0.0)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    ]
    # Middle sample passes through the obstacle centre line: the segment
    # is 0.2 from the centre, margin 0.2 - 0.1 - 0.02 = 0.08.
    margin, worst = check_trajectory_clearance(seq2, [0.02], obstacles)
    assert margin == pytest.approx(0.08)
    assert worst == 1


def test_check_trajectory_clearance_empty():
    margin, worst = check_trajectory_clearance([], [0.02],
                                               [SphereObstacle(center=(0.0, 0.0, 0.0), radius=0.1)])
    assert margin == math.inf
    assert worst == -1


def test_check_trajectory_clearance_planar():
    seq = [[(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.4, 0.0)[:2], (1.0, 0.4)]]
    margin, worst = check_trajectory_clearance(
        seq, [0.02], [CircleObstacle(center=(0.5, 0.3), radius=0.1)],
        planar=True)
    # Second sample runs 0.1 from the centre: margin 0.1 - 0.1 - 0.02.
    assert margin == pytest.approx(-0.02)
    assert worst == 1
