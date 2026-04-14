"""This module contains a class for handling one single point cloud."""

import random
from datetime import date
from enum import Enum

import numpy as np
import open3d as o3d


class PlantPointCloudLabelType(Enum):
    """Enum class for the two different label types."""

    SEMANTIC_LABEL = "SEMANTIC_LABEL"
    INSTANCE_LABEL = "INSTANCE_LABEL"


def _create_random_color() -> np.ndarray:
    """Create a random color."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.asarray([float(r), float(g), float(b)], dtype=np.float32) / 255.0


class PlantPointCloud:
    """This represents a single point cloud."""

    def __init__(self, point_data, caption_date: date, fix_colors=False):
        """Create a new PointCloud object.

        This represents a single set of points for a 3d image
        :param point_data: the point cloud data
        :param caption_date: the date of creation
        """
        self.point_data: o3d.t.geometry.PointCloud = point_data
        self.caption_date: date = caption_date
        self.morphology = None
        if self.point_data.point.__contains__("colors"):
            if fix_colors:
                self._fix_color_format()
            self._original_color = self.point_data.point.colors.clone()
            self._segmented_color = self.point_data.point.colors.clone()
            """This must be calculated in background as this is an expensive operation"""

    def _fix_color_format(self):
        """Fiy the colors in [point_data] format to float32.

        The colors comes formatted in Uint8. To avoid warnings by open3D, we need to format the colors as float32.
        """
        self.point_data.point.colors = (
            self.point_data.point.colors.numpy().astype(np.float32) / 255.0
        )

    def get_points(self) -> o3d.utility.Vector3dVector:  # Vector3dVector:
        """Return the points of the point cloud.

        :return: the points of the point cloud.
        """
        return self.point_data.to_legacy().points

    def get_colors(self):
        """Return the colors of the point cloud.

        :return: the colors of the point cloud.
        """
        try:
            return self.point_data.point.colors
        except Exception as e:
            print(
                f"Failed to load colors. The data might not provide color information :(. Error: {e}"
            )
            return None

    def get_colors_as_legacy(self):
        """Return the colors in legacy format."""
        return self.point_data.to_legacy().colors

    def _get_label_map(
        self, label_type: PlantPointCloudLabelType
    ) -> o3d.core.Tensor | None:
        """Return the label map for the given label type."""
        try:
            match label_type:
                case PlantPointCloudLabelType.SEMANTIC_LABEL:
                    return self.point_data.point.scalar_sem_label
                case PlantPointCloudLabelType.INSTANCE_LABEL:
                    return self.point_data.point.scalar_inst_label
        except Exception as e:
            print(
                f"Failed to load label map. The data might not provide labels. Can not color segments :(. Error: {e}"
            )
            return None

    def paint_one_segment(
        self,
        selected_label: int,
        label_type: PlantPointCloudLabelType,
        color: np.ndarray,
    ):
        """Paint one segment of the plant in the given color."""
        label_map = self._get_label_map(label_type)
        if label_map is None:
            return

        colors = self.get_colors()
        if colors is None:
            return

        labels_np = label_map.numpy().flatten()

        # Pre-compute all unique labels
        unique_labels = np.unique(labels_np)

        # Create color lookup table
        color_lut = colors.clone()

        for label in unique_labels:
            if label == selected_label:
                color_lut[int(label)] = color

        # Vectorized lookup
        colors = color_lut[labels_np]

        # apply the new color
        self.point_data.point.colors = colors

    def paint_each_segment(self, label_type: PlantPointCloudLabelType, color_map=None):
        """Color each segment of the plant in another color.

        Returns the color map for reusing on the next plant.
        """
        if color_map is None:
            color_map = {}

        label_map = self._get_label_map(label_type)
        if label_map is None:
            return None

        if self._segmented_color is None:
            return None

        labels_np = label_map.numpy().flatten()

        # Pre-compute all unique labels
        unique_labels = np.unique(labels_np)

        # Create color lookup table
        max_label = int(np.max(labels_np)) + 1
        color_lut = np.zeros((max_label, 3), dtype=np.float32)

        for label in unique_labels:
            if label not in color_map:
                color_map[label] = _create_random_color()
            color_lut[int(label)] = color_map[label]

        # Vectorized lookup
        self._segmented_color = color_lut[labels_np]
        return color_map

    def reset_color(self):
        """Reset the colors of the plant to the original ones."""
        self.point_data.point.colors = self._original_color

    def apply_segmented_colors(self):
        """Apply the segmentation color as active color."""
        self.point_data.point.colors = self._segmented_color
