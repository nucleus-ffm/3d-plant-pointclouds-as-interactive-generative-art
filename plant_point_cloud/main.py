"""Main entrypoint for the app.

Starts process controlling handling, GUI, and camera input.
"""

import time
from multiprocessing import Manager, set_start_method

from plant_point_cloud.datahandling.controller import Controller
from plant_point_cloud.handtracking.gesture_controller import GestureController

TIMEOUT = 20


def run():
    """Run the app."""
    # set the process start method to spawn to enforce
    # the same behavior on Linux, Mac and Windows
    set_start_method("spawn")
    manager = Manager()
    thread_killer = manager.Event()
    communication = manager.Queue()
    controller = Controller(
        thread_killer=thread_killer,
        point_cloud_set_path="plant_point_cloud/data/con_soybean_final_dataset",  # soybean_point_clouds",
        communication=communication,
    )
    controller.start()
    gesture_controller = GestureController(
        thread_killer=thread_killer,
        communication=communication,
    )
    gesture_controller.start()
    while not thread_killer.is_set():
        time.sleep(1)
    gesture_controller.join(timeout=TIMEOUT)
    controller.join(timeout=TIMEOUT)

    print("All controller joined. Shutdown manager.")
    manager.shutdown()
    print("Done, Good bye...")


if __name__ == "__main__":
    # this gets executed if the main.py file is executed directly
    run()
