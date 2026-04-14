"""This module contains a class for handling a set of point clouds."""

import os
from datetime import date
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np
import open3d

from .plant_morphology_data import MorphologyData
from .plant_point_cloud import PlantPointCloud, PlantPointCloudLabelType


def _get_content_of_folder(path_to_folder) -> list[str]:
    """Return all files from the given folder ascending.

    :param path_to_folder: the folder path
    :return: a list of filenames
    :raises FileNotFoundError: if the folder doesn't exist
    """
    # Build list with all filenames from the given directory.
    # Only include files and ignore everything else.
    # listdir returns files name in the given directory in arbitrary order
    files_in_folder = [
        f for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))
    ]
    # sort the file names ascending
    files_in_folder.sort()
    return files_in_folder


def _extract_caption_date_from_filename(filename: str) -> date:
    """Extract the caption date from the filename.

    This requires the files to have the format like `YYYY-MM-DD_[...]`

    :param filename: the filename of the point cloud data
    :return: the date of caption from the filename
    """
    # split with the first underscore
    date_from_file_name = filename.split("_")
    try:
        return date.fromisoformat(date_from_file_name[0])
    except ValueError:
        # use this as fallback if the file do not have the right format
        return date.fromisoformat("2000-01-01")


def _load_point_cloud_set(path) -> list[PlantPointCloud]:
    """Load the point cloud data from the given path.

    Load the point cloud data set from the path and create PlantPointCloud objects from that.

    This method is not robust against broken ply files. There will be a warning in the terminal from open3d
    but it seems not to raise any exception, so we can not catch these errors. We might want to perform
    input validation before we call this method.
    :param path: the path to the folder with the point cloud data
    :return: a list of PointCloud objects.
    """
    files = _get_content_of_folder(path)
    result = []
    for file in files:
        # @TODO only load .ply files
        if not file.endswith(".ply"):
            continue
        caption_date = _extract_caption_date_from_filename(file)
        path_to_file = Path(join(path, file))
        result.append(
            PlantPointCloud(
                point_data=open3d.t.io.read_point_cloud(path_to_file),
                caption_date=caption_date,
                fix_colors=True,
            )
        )
    return result


def _extract_plant_name(path) -> str:
    """Extract the plant name from the path.

    We expect the plant name as the first part of the folder name
    :param path: the path of folder with the point cloud data
    :return: the name of the plant from the folder name.
    """
    folder_name: str = os.path.basename(path)
    return folder_name


class PlantPointCloudSet:
    """Represents a set of point clouds.

    A set is a series of the same plant over time. While creating an instance of
    this class you have to point to a flower with multiple snapshots of the
    same plant over time.
    """

    def __init__(self, path_to_folder):
        """Create a set of Point clouds.

        This can be used to handle multiple point clouds for one plant.
        This will read all the files inside the given folder.

        :param path_to_folder: The folder path to the point cloud data files.
        """
        path_to_folder = Path(path_to_folder)
        self.plant_point_cloud_data: list[PlantPointCloud] = _load_point_cloud_set(
            path_to_folder
        )
        """Contains a list of plant point clouds."""
        self.plant_name: str = _extract_plant_name(path_to_folder)
        """The name of the plant extract from the folder name."""

        try:
            morphology = MorphologyData(path_to_folder)
        except ValueError:
            morphology = None

        for plant in self.plant_point_cloud_data:
            if morphology is None:
                plant.morphology = None
                continue

            try:
                morphology.load_json_for_date(plant.caption_date)
                plant.morphology = morphology
            except FileNotFoundError:
                plant.morphology = None

    def paint_each_segment(self, label_type: PlantPointCloudLabelType):
        """Paint each segment on each plant in the set."""
        color_map = None
        for plant in self.plant_point_cloud_data:
            color_map = plant.paint_each_segment(label_type, color_map)
        print("Assigning colors done. ")

    def paint_one_segment(
        self, label: int, label_type: PlantPointCloudLabelType, color: np.ndarray
    ):
        """Paint one segment on each plant in the set."""
        for plant in self.plant_point_cloud_data:
            plant.paint_one_segment(label, label_type, color)

    def apply_each_segment_color(self):
        """Apply the segmented colors to the plant.

        This required to have calculated color first. Call <paint_each_segment> first
        """
        for plant in self.plant_point_cloud_data:
            plant.apply_segmented_colors()

    def reset_color(self):
        """Reset the color of the plant to the original color."""
        for plant in self.plant_point_cloud_data:
            plant.reset_color()
