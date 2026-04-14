"""This module calculates the camera position and focus detection."""

import numpy as np
from open3d.cpu.pybind.geometry import KDTreeFlann


def _camera_pose(camera):
    """Return the camera world position and forward viewing direction.

    Extracts the camera pose from the view matrix and converts it
    into a 3D position (origin) and normalized forward vector.
    """
    view = np.asarray(camera.get_view_matrix())
    t = np.linalg.inv(view)

    cam_pos = t[:3, 3]
    forward = t[:3, 2]
    forward /= np.linalg.norm(forward) + 1e-9

    return cam_pos, forward


def _dist_to_ray(p, o, d):
    """Compute the shortest distance between a point 'p' and a ray defined by origin 'o' and direction 'd'."""
    return np.linalg.norm(np.cross((p - o), d)) / np.linalg.norm(d)


class FocusDetector:
    """Detects which point in the point cloud is currently being looked at.

    Uses a KD-tree search around a probe point projected along the
    camera forward direction. Returns the nearest valid point and
    its organ label.
    """

    def __init__(self, point_cloud, knn=None, distance_threshold=None):
        """Initialize the focus detector.

        point_cloud       – Open3D point cloud.
        knn               – Number of neighbors checked around the probe.
        focus_threshold   – Max allowed distance from ray to be considered 'focused'.
        """
        self._point_cloud = point_cloud
        self._points = point_cloud.point.positions.numpy()
        self.labels = point_cloud.point.scalar_inst_label.numpy().flatten()
        self.pcd_kdtree = KDTreeFlann(point_cloud.to_legacy())

        self._knn = knn
        self._distance_threshold = distance_threshold

    def compute(self, camera):
        """Compute whether the user is focusing on a plant organ.

        Returns:
            (focused: bool,
            position: np.ndarray | None,
            organ_label: int | None)
        """
        if len(self._points) == 0:
            return False, None, None

        cam_pos, forward = _camera_pose(
            camera
        )  # returns where camera is, and where camera is looking
        # creates a point in front of camera, to search around this point for nearby geometry
        probe = cam_pos + forward
        # find the closest point to the probe
        _, idx, _ = self.pcd_kdtree.search_knn_vector_3d(probe, self._knn)
        best_idx = None
        best_dist = 1e18

        # for each nearby points compute distance from that point to the camera direction,
        # and pick the smallest distance i.e., closest

        for i in idx:
            p = self._points[i]
            d = _dist_to_ray(p, cam_pos, forward)
            if d < best_dist:
                best_dist = d
                best_idx = i  # point closest to camera direction

        if best_idx is None or best_dist > self._distance_threshold:
            return False, None, None

        label = int(self.labels[best_idx])

        if label is None or label == 0:
            return False, None, None

        focus_pos = self._points[best_idx]

        return True, focus_pos, label
