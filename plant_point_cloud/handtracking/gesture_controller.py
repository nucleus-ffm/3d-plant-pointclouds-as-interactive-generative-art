"""The gesture controller tracks the hand movement."""

import queue
from multiprocessing import Process

from plant_point_cloud.enums.communication_events import CommunicationEventType
from plant_point_cloud.lib.depthai_hand_tracker.HandController import HandController


class GestureController(Process):
    """Class to handle the hand control gestures."""

    def __init__(self, thread_killer, communication):
        """Init the gesture controller."""
        super().__init__()
        self.thread_killer = thread_killer
        self.communication = communication

        self._zoom_level = 0.7
        self._last_zoom_distance = None
        self.MIN_HAND_MOVEMENT = 0.01
        self.ZOOM_SPEED = 0.002
        self.MIN_ZOOM = 0.1
        self.MAX_ZOOM = 3.0
        self.MOVEMENT_DEAD_ZONE = 0.01
        self.ROTATION_SPEED = 5
        self.SMOOTHING_FACTOR = 0.95
        self._zoom_activation_ticks = 0
        self._gesture_activation_threshold = 15
        self._rotate_left_ticks = 0
        self._rotate_right_ticks = 0
        self._tilt_up_ticks = 0
        self._tilt_down_ticks = 0
        self._last_position = None
        self.camera: HandController | None = None
        self.camera_active = True
        self.show_renderer = True
        self.config = None
        self._build_config()

    def _build_config(self):
        self.config = {
            "renderer": {"enable": self.show_renderer},
            "pose_actions": [
                {
                    "name": "rotate_left",
                    "pose": "ONE",
                    "hand": "right",
                    "callback": "rotate_left",
                    "trigger": "continuous",
                    "first_trigger_delay": 0.3,
                },
                {
                    "name": "rotate_right",
                    "pose": "TWO",
                    "hand": "right",
                    "callback": "rotate_right",
                    "trigger": "continuous",
                    "first_trigger_delay": 0.3,
                },
                {
                    "name": "zoom",
                    "pose": "FIVE",
                    "hand": "right",
                    "callback": "zoom",
                    "trigger": "continuous",
                    "first_trigger_delay": 1.0,
                },
                {
                    "name": "tilt_up",
                    "pose": "THREE",
                    "hand": "right",
                    "callback": "tilt_up",
                    "trigger": "continuous",
                    "first_trigger_delay": 0.3,
                },
                {
                    "name": "tilt_down",
                    "pose": "PEACE",
                    "hand": "right",
                    "callback": "tilt_down",
                    "trigger": "continuous",
                    "first_trigger_delay": 0.2,
                },
                {
                    "name": "dynamic_control",
                    "pose": "FOUR",
                    "hand": "right",
                    "callback": "dynamic_control",
                    "trigger": "continuous",
                    "first_trigger_delay": 0.2,
                },
                {
                    "name": "Toggle information overlay",
                    "pose": "FIST",
                    "hand": "right",
                    "callback": "toggle_information",
                    "trigger": "enter",
                    "first_trigger_delay": 0.2,
                },
            ],
        }

    def _get_action_from_queue(self):
        """Get the next action from the communication queue.

        returns None if there is no new element
        """
        try:
            event = self.communication.get(False)
            # check if this is the right target, and if not, put it back into the queue
            if event["target"] != "gesture_controller":
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
                case CommunicationEventType.TOGGLE_CAMERA:
                    value = payload["value"]
                    self.camera_active = value
                case CommunicationEventType.SHOW_CAMERA_RENDERER:
                    value = payload["value"]
                    self.show_renderer = value
                    self._build_config()
                    if value:
                        self.camera.start_renderer()
                    else:
                        self.camera.stop_renderer()

    def _start_camera(self):
        """Start the camera."""
        self.camera = HandController(
            caller_class=self,
            config=self.config,
            thread_killer=self.thread_killer,
        )

    def _stop_camera(self):
        """Stop the camera."""
        if self.camera is not None:
            print("Stop camera")
            self.camera.destroy()
            self.camera = None

    def run(self):
        """Start gesture controller."""
        self._start_camera()

        while not self.thread_killer.is_set():
            action, payload = self._get_action_from_queue()
            self._handle_actions(action, payload)

            if self.camera is None and self.camera_active:
                self._start_camera()

            if self.camera is not None and self.camera_active:
                try:
                    self.camera.tick()
                except RuntimeError as e:
                    print(f"[GestureController] DepthAI runtime error: {e}")

                except Exception as e:
                    print(f"[GestureController] Unexpected error: {e}")

            elif not self.camera_active and self.camera is not None:
                self._stop_camera()

        print("Gesture controller stopped.")


def rotate_left(self, event):
    """Rotate the plant to the left."""
    self._rotate_left_ticks += 1
    if self._rotate_left_ticks < self._gesture_activation_threshold:
        return
    self.communication.put(
        {
            "target": "controller",
            "event": CommunicationEventType.ROTATE,
            "payload": {"azimuth": -10.0, "altitude": 0.0},
        }
    )


def rotate_right(self, event):
    """Rotate the plant to the right."""
    self._rotate_right_ticks += 1
    if self._rotate_right_ticks < self._gesture_activation_threshold:
        return
    self.communication.put(
        {
            "target": "controller",
            "event": CommunicationEventType.ROTATE,
            "payload": {"azimuth": 10.0, "altitude": 0.0},
        }
    )


def tilt_up(self, event):
    """Tilt the plant upward."""
    self._tilt_up_ticks += 1
    if self._tilt_up_ticks < self._gesture_activation_threshold:
        return
    self.communication.put(
        {
            "target": "controller",
            "event": CommunicationEventType.ROTATE,
            "payload": {"azimuth": 0.0, "altitude": 10.0},
        }
    )


def tilt_down(self, event):
    """Tilt the plant downward."""
    self._tilt_down_ticks += 1
    if self._tilt_down_ticks < self._gesture_activation_threshold:
        return
    self.communication.put(
        {
            "target": "controller",
            "event": CommunicationEventType.ROTATE,
            "payload": {"azimuth": 0.0, "altitude": -10.0},
        }
    )


def zoom(self, event):
    """Zoom the camera using hand size.

    The zoom level is controlled by measuring the distance between the wrist
    and the tip of the middle finger. As the hand moves closer to the camera,
    the hand size increases, causing the view to zoom out. Moving
    the hand farther away zooms in.

    """
    self._zoom_activation_ticks += 1
    if self._zoom_activation_ticks < self._gesture_activation_threshold:
        return
    wrist_x, wrist_y = event.hand.landmarks[0, :2]
    middle_tip_x, middle_tip_y = event.hand.landmarks[12, :2]

    distance_from_wrist_to_middle_finger = (
        (middle_tip_x - wrist_x) ** 2 + (middle_tip_y - wrist_y) ** 2
    ) ** 0.5

    if self._last_zoom_distance is None:
        self._last_zoom_distance = distance_from_wrist_to_middle_finger
        return

    size_change = distance_from_wrist_to_middle_finger - self._last_zoom_distance

    # only apply size change if the value is below the threshold
    if abs(size_change) > 80:
        size_change = 0
    self._last_zoom_distance = distance_from_wrist_to_middle_finger

    if abs(size_change) < self.MIN_HAND_MOVEMENT:
        return

    self._zoom_level += size_change * self.ZOOM_SPEED
    self._zoom_level = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom_level))

    self.communication.put(
        {
            "target": "controller",
            "event": CommunicationEventType.ZOOM,
            "payload": {"zoom": float(self._zoom_level)},
        }
    )


def dynamic_control(self, event):
    """Combine rotate and tilt with one gesture."""
    if event.hand is None or event.hand.landmarks is None:
        self._last_position = None
        self._smoothed_dx = 0
        self._smoothed_dy = 0
        return

    wrist_x, wrist_y = event.hand.landmarks[0, :2]
    current_pos = (wrist_x, wrist_y)

    if self._last_position is None:
        self._last_position = current_pos
        self._smoothed_dx = 0
        self._smoothed_dy = 0
        return

    last_x, last_y = self._last_position
    raw_dx = wrist_x - last_x
    raw_dy = wrist_y - last_y

    # only apply movement if change is under the threshold to allow continuous gestures
    if abs(raw_dx) > 100:
        raw_dx = 0
    if abs(raw_dy) > 100:
        raw_dy = 0

    self._last_position = current_pos

    self._smoothed_dx = (
        self.SMOOTHING_FACTOR * self._smoothed_dx + (1 - self.SMOOTHING_FACTOR) * raw_dx
    )

    self._smoothed_dy = (
        self.SMOOTHING_FACTOR * self._smoothed_dy + (1 - self.SMOOTHING_FACTOR) * raw_dy
    )

    dx = self._smoothed_dx
    dy = self._smoothed_dy

    if abs(dx) < self.MOVEMENT_DEAD_ZONE:
        dx = 0.0
    if abs(dy) < self.MOVEMENT_DEAD_ZONE:
        dy = 0.0

    azimuth = dx * self.ROTATION_SPEED
    altitude = -dy * self.ROTATION_SPEED

    if azimuth != 0.0 or altitude != 0.0:
        self.communication.put(
            {
                "target": "controller",
                "event": CommunicationEventType.ROTATE,
                "payload": {
                    "azimuth": float(azimuth),
                    "altitude": float(altitude),
                },
            }
        )


def toggle_information(self, event):
    """Toggle the information overly and sidepanel."""
    self.communication.put(
        {
            "target": "controller",
            "event": CommunicationEventType.TOGGLE_INFORMATION_OVERLAY,
            "payload": {},
        }
    )
