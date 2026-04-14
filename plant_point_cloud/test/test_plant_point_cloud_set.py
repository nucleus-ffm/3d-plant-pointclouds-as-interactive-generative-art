"""Unit tests for the point cloud loader."""

from datetime import date

import pytest

from plant_point_cloud.datahandling.plant_point_cloud_set import PlantPointCloudSet

path_valid_set = "plant_point_cloud/test/test_data/my_test_plant_set_valid"
path_invalid_set = "plant_point_cloud/test/test_data/my_test_plant_set_invalid"
path_invalid_filenames = (
    "plant_point_cloud/test/test_data/my_test_plant_set_invalid_filenames"
)
path_invalid_folder = "this_folder_does_not_exist"


def test_valid_plant_point_cloud_set():
    """Test a valid set.

    We expect that all capture dates are correct.
    """
    plant_set: PlantPointCloudSet = PlantPointCloudSet(path_valid_set)

    list_of_expected_dates = [
        date.fromisoformat("2025-01-10"),
        date.fromisoformat("2025-01-11"),
    ]

    assert plant_set.plant_name == "my_test_plant_set_valid"
    list_point_clouds = plant_set.plant_point_cloud_data

    # check if the detected capture date are as expected
    for i, point_cloud in enumerate(list_point_clouds):
        assert point_cloud.caption_date == list_of_expected_dates[i]


def test_invalid_plant_point_cloud_set():
    """Test with invalid point cloud set.

    @TODO: This test needs to be extended.
    We can not detect broken .ply file currently.
    """
    PlantPointCloudSet(path_invalid_set)


def test_invalid_filenames_plant_point_cloud_set():
    """Test with invalid filenames set.

    We expect that the fallback date 2000-01-01 is used
    """
    plant_set = PlantPointCloudSet(path_invalid_filenames)
    # 2000-01-01 is the fallback if we can not detect a valid
    # date in the filename
    expected_date: date = date.fromisoformat("2000-01-01")
    assert plant_set.plant_point_cloud_data[0].caption_date == expected_date


def test_with_invalid_folder():
    """Test with invalid folder.

    This should raise a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        PlantPointCloudSet(path_invalid_folder)
