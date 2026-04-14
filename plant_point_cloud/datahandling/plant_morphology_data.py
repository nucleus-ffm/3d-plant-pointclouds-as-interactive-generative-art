"""This module loads and read data from .json file."""

import json
from pathlib import Path


class MorphologyData:
    """Provides access to plant morphology metadata, organ attributes, and time-point information stored in a JSON file."""

    def __init__(self, json_path: Path):
        """Load the JSON morphology file."""
        if json_path.is_file():
            self._json_files = [json_path]
        elif json_path.is_dir():
            self._json_files = sorted(json_path.glob("*.json"))
            if not self._json_files:
                raise ValueError(f"No JSON files found in {json_path}")
        else:
            raise ValueError(f"Invalid JSON path: {json_path}")

        self._json_data = []
        for json_file in self._json_files:
            with open(json_file) as f:
                self._json_data.append(json.load(f))

        self.data = {}
        self.plants = {}

    def load_json_for_date(self, target_date):
        """Load morphology data matching the given date."""
        if hasattr(target_date, "date"):
            target_date = target_date.date()  # datetime → date
        target_date_str = str(target_date)

        for data in self._json_data:
            plants = data.get("plants", {})

            for _plant_id, plant_data in plants.items():
                for _time_idx, time_data in plant_data.items():
                    if time_data.get("date") == target_date_str:
                        self.data = data
                        self.plants = plants
                        return

        raise FileNotFoundError(f"No JSON file found matching date {target_date_str}")

    def get_plant_ids(self):
        """Return a list of all plant IDs contained in the dataset."""
        if not self.plants:
            return []

        return list(self.plants.keys())

    def get_timepoints(self, plant_id: str):
        """Return all recorded time points for the given plant."""
        return list(self.plants.get(str(plant_id), {}).keys())

    def get_organs(self, plant_id: str, time_idx: str):
        """Return all organ groups for a given plant and time point."""
        plant = self.plants.get(str(plant_id), {})
        return plant.get(str(time_idx), {}).get("organs", {})

    def get_organ_record(self, plant_id: str, time_idx: str, organ_label: int):
        """Returns the JSON info of the organ with this label."""
        organs = self.get_organs(plant_id, time_idx)

        for _organ_key, organ_group in organs.items():
            for organ_type, organ_data in organ_group.items():
                if organ_data.get("organ_label") == organ_label:
                    return organ_type, organ_data

        return None, {}
