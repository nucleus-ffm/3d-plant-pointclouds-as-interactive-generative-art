"""The transitions controller computes all the stuff needed to have a smooth transition between two point clouds."""

import queue
from datetime import datetime, timedelta
from multiprocessing import Queue as MultiprocessingQueue
from queue import Queue as ThreadQueue
from threading import Event, Thread

import numpy as np
import open3d as o3d

from plant_point_cloud.datahandling.plant_point_cloud import PlantPointCloud
from plant_point_cloud.datahandling.plant_point_cloud_set import PlantPointCloudSet
from plant_point_cloud.enums.communication_events import CommunicationEventType


class TransitionController(Thread):
    """This class calculates all the transition stuff and puts every calculated frame in a shared queue."""

    def __init__(
        self,
        interpolated_frames: ThreadQueue,
        point_cloud_set: PlantPointCloudSet,
        communication: MultiprocessingQueue,
        thread_killer_global: Event,
        delay_between_transition=5,
        duration_one_transition=2,
        current_plant=0,
        transition_active=True,
    ) -> None:
        """Create new instance of the transition controller."""
        super().__init__()
        self.point_cloud_set: PlantPointCloudSet = point_cloud_set
        self.interpolated_frames: ThreadQueue = interpolated_frames
        """Shared queue between controller and TransitionController.
        Please only put open3d.utility.Vector3dVector objects in it."""
        self.communication = communication
        """Shared queue used as a communication bus between processes and threads."""
        self._thread_killer_global: Event = thread_killer_global
        """This event is shared between all threads and process to 
        terminate the program in total."""
        self.thread_killer_local: Event = Event()
        """This Event is used to just terminate this thread. This thread is
        the only thread, that is listening to this event"""
        self.transition_active = transition_active
        """This optional parameter can be used to define of the transition should
        be active at startup or not."""
        self.delay_between_transition = delay_between_transition
        """Delay in seconds between one transition and another."""
        self.duration_one_transition = duration_one_transition
        """Duration in second of one transition"""
        self.current_plant = current_plant
        """This stores which plant is currently displayed"""

    @staticmethod
    def index_mapping(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """Maps two pointclouds based on their points' indices."""
        mapped_dst = np.empty_like(src_pts)

        for i in range(len(src_pts)):
            mapped_dst[i] = dst_pts[i % len(dst_pts)]
        return mapped_dst

    @staticmethod
    def smoothstep_alpha(t: float) -> float:
        """Creates an alpha for a smoothstep transition. A sigmoid-like interpolation function, widely used in computer graphics to create smooth transitions."""
        return (6 * t**5 - 15 * t**4 + 10 * t**3) ** 2

    @staticmethod
    def _match_array_sizes_tensor(tensor1, tensor2):
        """Add duplicate points to the smaller array to match the size of the larger array.

        Args:
            tensor1: Open3D tensor containing coordinates
            tensor2: Open3D tensor containing coordinates

        Returns:
            Two numpy arrays with the same number of points
        """
        array1 = tensor1.point.positions.numpy()
        array2 = tensor2.point.positions.numpy()
        n1, n2 = len(array1), len(array2)
        sem_label = None
        duplicated_labels = None
        expanded_labels = None

        if n1 == n2:
            return array1, array2

        # Determine which array is smaller
        if n1 < n2:
            smaller = array1
            color = tensor1.point.colors.numpy()
            if tensor1.point.__contains__("scalar_sem_label"):
                sem_label = tensor1.point.scalar_sem_label.numpy()
            target_size = n2
        else:
            smaller = array2
            color = tensor2.point.colors.numpy()
            if tensor2.point.__contains__("scalar_sem_label"):
                sem_label = tensor2.point.scalar_sem_label.numpy()
            target_size = n1

        # Calculate how many points we need to add
        points_to_add = target_size - len(smaller)

        # Create indices to sample points, spreading them evenly
        # This creates indices distributed across the original array
        indices = np.linspace(0, len(smaller) - 1, points_to_add, dtype=int)

        # Get the points to duplicate
        duplicated_points = smaller[indices]
        duplicated_colors = color[indices]
        if sem_label is not None:
            duplicated_labels = sem_label[indices]

        # Concatenate original points with duplicated points
        expanded_array = np.vstack([smaller, duplicated_points])
        expanded_colors = np.vstack([color, duplicated_colors])
        if sem_label is not None:
            expanded_labels = np.vstack([sem_label, duplicated_labels])

        # Return in the same order as input
        if n1 < n2:
            tensor1.point.positions = o3d.core.Tensor(expanded_array)
            tensor1.point.colors = o3d.core.Tensor(expanded_colors)
            if expanded_labels is not None:
                tensor1.point.scalar_sem_label = o3d.core.Tensor(expanded_labels)
            return tensor1, tensor2
        else:
            tensor2.point.positions = o3d.core.Tensor(expanded_array)
            tensor2.point.colors = o3d.core.Tensor(expanded_colors)
            if expanded_labels is not None:
                tensor2.point.scalar_sem_label = o3d.core.Tensor(expanded_labels)
            return tensor1, tensor2

    def _get_action_from_queue(self):
        """Get the next action from the communication queue.

        returns None if there is no new element
        """
        try:
            event = self.communication.get(False)
            # check if this is the right target, and if not, put it back into the queue
            if event["target"] != "transition_controller":
                self.communication.put(event)
                return None, None
            else:
                if "payload" in event:
                    return event["event"], event["payload"]
                else:
                    return event["event"], None
        except queue.Empty:
            return None, None

    def _handle_actions(self, action: CommunicationEventType, payload=None) -> None:
        """Handle actions from communication queue."""
        if action is not None:
            match action:
                case CommunicationEventType.SET_TIME_TRAVEL:
                    self.transition_active = payload["value"]
                case CommunicationEventType.UPDATE_TRANSITION_DELAY:
                    self.delay_between_transition = payload["value"]
                case CommunicationEventType.UPDATE_TRANSITION_DURATION:
                    self.duration_one_transition = payload["value"]
                case _:
                    print("action not yet implemented")

    def _communicate_current_plant(self):
        """Send the current plant to the controller to allow updating the day slider."""
        self.communication.put(
            {
                "target": "controller",
                "event": CommunicationEventType.UPDATE_DAY,
                "payload": {"value": self.current_plant},
            }
        )

    def run(self) -> None:
        """Main loop of the transition controller."""
        # calculate number for frames for the given duration
        # this assumes 60 frames per second
        plant_data = self.point_cloud_set.plant_point_cloud_data
        n_plants: int = len(plant_data)
        self.current_plant = 0
        wait_until = datetime.now()

        while (
            not self._thread_killer_global.is_set()
            and not self.thread_killer_local.is_set()
        ):
            action, payload = self._get_action_from_queue()
            self._handle_actions(action, payload)
            frames: int = round(self.duration_one_transition * 60)

            if self.transition_active and wait_until - datetime.now() <= timedelta(
                seconds=0
            ):
                next_i = (self.current_plant + 1) % n_plants
                # as we are modifying the number of point, we should get a close of the original object to avoid
                # manipulating the original data
                start_pts_tensor_raw = plant_data[self.current_plant].point_data.clone()
                target_pts_tensor_raw = plant_data[next_i].point_data.clone()

                # @Implement: The color could transition into the new colors on the way

                start_pts_tensor, target_pts_tensor = self._match_array_sizes_tensor(
                    start_pts_tensor_raw, target_pts_tensor_raw
                )
                start_pts_colors = start_pts_tensor.point.colors.clone()
                start_pts = start_pts_tensor.point.positions.numpy()
                target_pts = target_pts_tensor.point.positions.numpy()

                # additional target points
                # this is not really necessary currently, as we're extending the source tensor before
                # but we keep this for now for future reference
                extra_pts = (
                    target_pts[len(start_pts) :]
                    if len(target_pts) > len(start_pts)
                    else np.empty((0, 3))
                )
                extra_count = len(extra_pts)
                # random start points for extra_pts from start_pts
                if extra_count > 0:
                    start_indices = np.random.choice(
                        len(start_pts), extra_count, replace=True
                    )
                    extra_start_pts = start_pts[start_indices]
                else:
                    extra_start_pts = np.empty((0, 3))

                mapped_pts = self.index_mapping(start_pts, target_pts).copy()

                start_date = plant_data[self.current_plant].caption_date
                target_date = plant_data[next_i].caption_date
                date_diff = target_date - start_date

                transition_plant = o3d.t.geometry.PointCloud(o3d.core.Tensor(start_pts))
                transition_plant.point.colors = start_pts_colors  # new_colors

                for f in range(frames + 1):
                    if self._thread_killer_global.is_set():
                        break

                    t: float = f / frames
                    alpha = self.smoothstep_alpha(t)
                    tracked_pts = (1 - alpha) * start_pts + alpha * mapped_pts

                    intermediate_date = start_date + alpha * date_diff

                    visible_extra = int(alpha * extra_count)

                    if visible_extra > 0:
                        growing_extra = (1 - alpha) * extra_start_pts[
                            :visible_extra
                        ] + alpha * extra_pts[:visible_extra]
                        tracked_pts = np.vstack((tracked_pts, growing_extra))

                    transition_plant = o3d.t.geometry.PointCloud(
                        o3d.core.Tensor(start_pts)
                    )

                    transition_plant.point.colors = start_pts_colors  # new_colors
                    transition_plant.point.positions = o3d.core.Tensor(tracked_pts)

                    self.interpolated_frames.put(
                        PlantPointCloud(
                            transition_plant, caption_date=intermediate_date
                        )
                    )
                    self.current_plant = next_i
                    self._communicate_current_plant()

                    # calculate the time of the next iteration
                    wait_until = datetime.now() + timedelta(
                        seconds=self.duration_one_transition
                        + self.delay_between_transition
                    )

        print("Transition controller stopped.")
