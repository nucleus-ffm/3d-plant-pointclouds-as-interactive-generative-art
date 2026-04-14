"""'This modules contains the controller of the plant app."""

import queue
import threading
import time
from multiprocessing import Process
from multiprocessing import Queue as MultiprocessingQueue
from os import PathLike
from queue import Queue as ThreadQueue

from plant_point_cloud.enums.communication_events import CommunicationEventType

from .plant_point_cloud import PlantPointCloudLabelType
from .plant_point_cloud_set import PlantPointCloudSet
from .sound_controller import SoundController
from .transition_controller import TransitionController
from .visualizer import Visualizer


class Controller(Process):
    """This class is used to control the entire software.

    The controller should be the only point that manipulates the view, the visualizer etc.
    If this gets to big, we may want to split things up in multiple smaller parts.
    """

    def __init__(self, thread_killer, point_cloud_set_path, communication):
        """The constructs a new controller object for controlling the plant visualizer.

        This controller itself is a process object. Calling `.start` on the controller object will start the process and
        will execute the `run` method.

        :param thread_killer: Flag to kill the thead
        :param point_cloud_set_path: the path to the set of plant point cloud data
        :param communication: The shared queue between all processes to exchanged messages like a bus.
        """
        super().__init__()
        self.visualizer: Visualizer | None = None
        self.thread_killer = thread_killer
        self.communication: MultiprocessingQueue = communication
        self.point_cloud_path: PathLike = point_cloud_set_path
        self.point_cloud_set: PlantPointCloudSet | None = None
        self.demo_mode = False
        self.interpolated_frames: ThreadQueue | None = None
        """This contains the updated point cloud of the transitions controller."""
        self.transition_controller: TransitionController | None = None
        self.sound_controller: SoundController | None = None

    def _create_visualizer(self) -> Visualizer:
        """Create the visualizer."""
        return Visualizer(
            plant_name=self.point_cloud_set.plant_name,
            point_cloud_data=self.point_cloud_set.plant_point_cloud_data[0],
            thread_killer=self.thread_killer,
            communication=self.communication,
            number_of_plants=len(self.point_cloud_set.plant_point_cloud_data) - 1,
        )

    def _create_interpolated_frame_queue(self) -> None:
        """Create the interpolated frame queue.

        Call this after the process has been created otherwise we might get
        issues on win and Mac.
        """
        self.interpolated_frames = ThreadQueue()

    def _get_action_from_queue(self):
        """Get the next action from the communication queue.

        returns None if there is no new element
        """
        try:
            event = self.communication.get(False)
            # check if this is the right target, and if not, put it back into the queue
            # This is not the ideal solution and could be improved in a future version
            if event["target"] != "controller":
                self.communication.put(event)
                return None, None
            else:
                if "payload" in event:
                    return event["event"], event["payload"]
                else:
                    return event["event"], None
        except queue.Empty:
            return None, None

    def _handle_rotation_action(self, action) -> None:
        """Handle a rotation action."""
        azimuth = action["azimuth"]
        altitude = action["altitude"]
        self.visualizer.rotate(azimuth, altitude)

    def _handle_zoom_action(self, action) -> None:
        """Handle the zoom action."""
        zoom = action["zoom"]
        self.visualizer.zoom(zoom)

    def _handle_load_point_cloud_action(self, payload) -> None:
        """Handle the load point cloud action."""
        path = payload["path"]
        self._load_point_cloud_set(path)

        # update the slider limits with the new set
        self.visualizer.update_limits_select_day_slider(
            len(self.point_cloud_set.plant_point_cloud_data) - 1
        )
        self.visualizer.select_day_slider.int_value = 0
        # recalculate the segmentation color
        self.visualizer.segmented_colors_checkbox.checked = False
        threading.Thread(
            target=self._calculate_segmentation_color_assignment, args=[]
        ).start()

        # we loaded a new point cloud set, we have to restart the
        # transition controller
        self._start_transition_controller()

    def _handle_actions(self, action: CommunicationEventType, payload=None) -> None:
        """Handle actions from communication queue."""
        if action is not None:
            match action:
                case CommunicationEventType.ROTATE:
                    self._handle_rotation_action(payload)
                case CommunicationEventType.ZOOM:
                    self._handle_zoom_action(payload)
                case CommunicationEventType.LOAD_POINT_CLOUD_FROM_PATH:
                    self._handle_load_point_cloud_action(payload)
                case CommunicationEventType.SET_DEMO_MODE:
                    self.demo_mode = payload["demo_mode"]
                case CommunicationEventType.TOGGLE_SEGMENTATION_COLOR:
                    apply_color = payload["value"]
                    if apply_color:
                        self.point_cloud_set.apply_each_segment_color()
                        # update the view with the new segmented colors only if the transition is not active
                        # in case the transition is active, the new colors will be applied with the next transition
                        if not self.transition_controller.transition_active:
                            self.interpolated_frames.put(
                                self.point_cloud_set.plant_point_cloud_data[
                                    self.transition_controller.current_plant
                                ]
                            )

                    else:
                        self.point_cloud_set.reset_color()
                        self.interpolated_frames.put(
                            self.point_cloud_set.plant_point_cloud_data[
                                self.transition_controller.current_plant
                            ]
                        )

                case CommunicationEventType.SELECT_DAY:
                    self.transition_controller.current_plant = payload["value"]
                    self.interpolated_frames.put(
                        self.point_cloud_set.plant_point_cloud_data[payload["value"]]
                    )
                    self.visualizer.on_plant_selected(
                        self.point_cloud_set.plant_point_cloud_data[payload["value"]]
                    )
                case CommunicationEventType.UPDATE_DAY:
                    day = payload["value"]
                    self.visualizer.select_day_slider.int_value = day
                    self.visualizer.on_plant_selected(
                        self.point_cloud_set.plant_point_cloud_data[payload["value"]]
                    )

                case CommunicationEventType.GENERATE_NEW_SEGMENTATION_COLOR_SET:
                    threading.Thread(
                        target=self._calculate_segmentation_color_assignment_and_apply,
                        args=[],
                    ).start()
                case CommunicationEventType.TOGGLE_INFORMATION_OVERLAY:
                    new_state = not self.visualizer.leaf_overlay_enabled
                    self.visualizer.set_information_overlay(new_state)
                case _:
                    print("action not yet implemented")

    def _load_point_cloud_set(self, path: PathLike) -> None:
        """Load a set of point clouds from the given path."""
        self.point_cloud_set = PlantPointCloudSet(path)
        self.interpolated_frames.put(self.point_cloud_set.plant_point_cloud_data[0])

    def _start_transition_controller(self) -> None:
        """Start the transition controller.

        If the controller is already running, terminate first and restart
        it again. This must be called after a new point cloud set is loaded
        to refresh the set that is used by the transition controller.
        The transition will start again from plant 0.
        """
        # start the transition controller to calculate the frames between point clouds
        transition_active = True
        delay = 5
        if self.transition_controller is not None:
            # kill the old thread to restart it with a new point could set and the same settings
            self.transition_controller.thread_killer_local.set()
            transition_active = self.transition_controller.transition_active
            delay = self.transition_controller.delay_between_transition
            self.transition_controller.join(timeout=5)
        self.transition_controller = TransitionController(
            self.interpolated_frames,
            self.point_cloud_set,
            self.communication,
            self.thread_killer,
            transition_active=transition_active,
            delay_between_transition=delay,
        )
        self.transition_controller.start()

    def _calculate_segmentation_color_assignment_and_apply(self) -> None:
        """Calculate the color segmentation and applies the new colors directly."""
        self.point_cloud_set.paint_each_segment(PlantPointCloudLabelType.INSTANCE_LABEL)
        if self.visualizer.segmented_colors_checkbox.checked:
            self.point_cloud_set.apply_each_segment_color()
        # update the view with the new segmented colors only if the transition is not active
        # in case the transition is active, the new colors will be applied with the next transition
        if not self.transition_controller.transition_active:
            self.interpolated_frames.put(
                self.point_cloud_set.plant_point_cloud_data[
                    self.transition_controller.current_plant
                ]
            )

    def _calculate_segmentation_color_assignment(self) -> None:
        """Calculate the color segmentation."""
        self.point_cloud_set.paint_each_segment(PlantPointCloudLabelType.INSTANCE_LABEL)

    def run(self) -> None:
        """This method get executed when the process is started.

        :return: None
        """
        print("starting controller")

        # load point cloud and create queue here
        # to avoid issues with multiprocessing when using spawn
        self._create_interpolated_frame_queue()
        self._load_point_cloud_set(self.point_cloud_path)

        # create window with visualizer
        self.visualizer = self._create_visualizer()

        self._start_transition_controller()
        self.sound_controller = SoundController(
            self.thread_killer, self.communication
        ).start()
        threading.Thread(
            target=self._calculate_segmentation_color_assignment, args=[]
        ).start()

        # used for the demo mode
        i = 0.0
        i_max = 5.0
        increasing = True

        while not self.thread_killer.is_set():
            start_time = time.time()

            action, payload = self._get_action_from_queue()
            self._handle_actions(action, payload)

            # this lets the plant spin and rotate automatically
            if self.demo_mode:
                if increasing:
                    if i < i_max:
                        i += 0.01
                    else:
                        increasing = False
                else:
                    if i > 0.02:
                        i -= 0.01
                    else:
                        increasing = True
                self.visualizer.zoom(i)
                self.visualizer.rotate(azimuth=5.0, altitude=5.0)

            # Check if there is a new point cloud in the queue and update the geometry accordingly
            try:
                next_point_cloud = self.interpolated_frames.get(False)
                self.visualizer.update_geometry(next_point_cloud)
            except queue.Empty:
                pass

            # this tick has to be called at every frame to keep the application alive
            self.visualizer.tick()
            # sleep shortly for reduced load, we update 60x per second here
            # we measure the time needed for our computation and if this took more time than 1/60, we do not wait at all
            end_time = time.time()
            time.sleep(max(1.0 / 60.0 - (end_time - start_time), 0))
        self.transition_controller.join()
        self.sound_controller.join(timeout=10)
        self.visualizer.dispose()
        print("Main controller stopped.")
        self.close()
