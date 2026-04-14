"""This module features an OSC controller that sends the changing zoom and rotation to an OSC server."""

import queue
import time
from multiprocessing import Queue
from threading import Event, Thread

from pythonosc.udp_client import SimpleUDPClient

from plant_point_cloud.enums.communication_events import CommunicationEventType


class SoundController(Thread):
    """This is a simple OSC sound controller.

     It listens for commands and sends value to the OSC server. This can be used to control
    sound parameters in e.g. Cardinal.
    """

    def __init__(self, thread_killer, communication):
        """Crate a new SoundController object."""
        Thread.__init__(self)
        self.ip = "127.0.0.1"
        """Defines the IP address of the OSC server. This can also be a remote server in the same network."""
        self.port = 2228
        """Dines the port of the OSC server. 2228 is the default."""
        self.client = None
        self.thread_killer: Event = thread_killer
        self.communication: Queue = communication
        self.current_degree = 0

    def _start_client(self):
        """Start the UDP OSC client."""
        self.client = SimpleUDPClient(self.ip, self.port)

    def _get_action_from_queue(self):
        """Get the next action from the communication queue.

        returns None if there is no new element
        """
        try:
            event = self.communication.get(False)
            # check if this is the right target, and if not, put it back into the queue
            if event["target"] != "sound_controller":
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
                case CommunicationEventType.SOUND_ROTATION_POS:
                    value = payload["value"]
                    self.current_degree = (self.current_degree + value) % 360
                    if self.current_degree > 0:
                        value = self.current_degree * 100 / 360
                        self.client.send_message("/host-param", [0, value])
                case CommunicationEventType.SOUND_ZOOM_POS:
                    zoom = payload["value"]
                    if zoom > 0:
                        value = zoom / 80
                    self.client.send_message("/host-param", [1, value])
                case _:
                    print("action not yet implemented")

    def run(self):
        """Main loop of the sound controller.

        This checks for new messages in the communication queue and sends
        commands to the OSC server.
        """
        print("Starting sound controller")

        self._start_client()

        while not self.thread_killer.is_set():
            action, payload = self._get_action_from_queue()
            self._handle_actions(action, payload)
            time.sleep(1 / 60)
