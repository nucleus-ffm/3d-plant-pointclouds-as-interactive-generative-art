"""This module contains the visualizer for 3d plant cloud."""

import colorsys
from multiprocessing import Queue
from pathlib import Path

import numpy as np
from open3d.visualization import gui, rendering

from plant_point_cloud.enums.communication_events import CommunicationEventType

from .focus_detector import FocusDetector, _camera_pose
from .plant_point_cloud import PlantPointCloud, PlantPointCloudLabelType

ROTATION_RADIAN_PER_PIXEL = 0.003  # this value is taken from the open3d implementation
DISTANCE_DEFAULT = 0.7  # this value is taken from the open3d implementation
AZIMUTH_DEFAULT = 15
ALTITUDE_DEFAULT = 15
POINT_SIZE_DEFAULT = 3.0
TRANSITION_DELAY_DEFAULT = 5.0  # time in second between two transitions
TRANSITION_DURATION_DEFAULT = 2.0  # time in second of one transition


class Visualizer:
    """Wrapper around open3d to visualize a 3d plant cloud."""

    """Dict used to store the 3D labels. To allow to delete them with just the name."""

    def __init__(
        self,
        plant_name,
        point_cloud_data: PlantPointCloud,
        thread_killer,
        communication: Queue,
        number_of_plants=1,
    ):
        """Create a new visualizer object. This is a wrapper around open3d.

        To keep this view alive, the update_geometry() method must be called at the desired frame rate.

        :param plant_name: The name of the plant to display. This is used as window name
        :param point_cloud_data: the initial point cloud data to display. This data can be updated at runtime with
        the `update_geometry()` function
        :param thread_killer: Event signal to terminate the viewer as soon as the signal is set.
        """
        super().__init__()
        self.communication: Queue = communication
        self.number_of_plants = number_of_plants
        """This defines the max value of the day select slider"""

        # set the point cloud data
        self._point_cloud_data: PlantPointCloud = point_cloud_data
        self._tensor_pcd = point_cloud_data.point_data
        # implement json file
        self.morphology_json = None
        self.time_idx = None
        self.plant_idx = None
        self._focus = None

        # implement FocusDetector
        self.leaf_overlay_enabled = False
        self._focus_label_name = "focus_label"
        self._focus_counter = 0
        self._focus_threshold_count = 20
        self._last_focused_organ = None

        self._label_store: dict[str, gui.Label3D] = {}
        self._leaf_highlight_colors: dict[int, np.ndarray] = {}
        self._highlighted_label: int | None = None
        self.on_plant_selected(point_cloud_data)
        # create open3d application

        self.app: gui.Application = gui.Application.instance
        self.app.initialize()

        self.window: gui.Application = self.app.create_window(
            title=plant_name, width=1000, height=1000
        )

        self.window.set_on_tick_event(self._on_main_window_tick_event)
        self.window.set_on_close(self._on_main_window_closing)

        # scene widget and rendering, this is the widget that displays the point cloud.
        # due to some bugs in open3d the scene must be added to the window directly
        # if a scene is added to a layout container first, the renderer is never updated
        # and will only display the first frame
        # see https://github.com/isl-org/Open3D/issues/5343
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.set_on_key(self._on_keypress)

        # self.scene.set_on_mouse(self._on_mouse)

        self.true_color: bool = False

        # Materials for rendering points, one with the original color, one with artificial one
        self.material_colorless = rendering.MaterialRecord()
        self.material_colorless.shader = (
            "defaultUnlit"  # good for points; "defaultLit" can be used with lighting
        )
        self.material_colorless.point_size = POINT_SIZE_DEFAULT  # initial point size

        self.material_colorful = rendering.MaterialRecord()
        self.material_colorful.shader = (
            "defaultUnlit"  # good for points; "defaultLit" can be used with lighting
        )
        self.material_colorful.point_size = POINT_SIZE_DEFAULT  # initial point size
        self.material_colorful.base_color = np.array(
            [0.0, 1.0, 0.0, 1.0]
        )  # set plant color to green

        # set material depending on true color setting
        if self.true_color:
            self.scene.scene.add_geometry(
                "plant", self._point_cloud_data.point_data, self.material_colorless
            )
        else:
            self.scene.scene.add_geometry(
                "plant", self._point_cloud_data.point_data, self.material_colorful
            )

        # Setup camera to frame the point cloud
        bbox = self._point_cloud_data.point_data.to_legacy().get_axis_aligned_bounding_box()
        center = bbox.get_center()
        self.scene.setup_camera(60.0, bbox, center)
        self.camera = self.scene.scene.camera

        # add scene to window
        self.window.add_child(self.scene)

        # create a side panel for control buttons
        self.em = self.window.theme.font_size
        """Relative size used to define other sizes. Based on the font size of the parent widget."""
        margin = self.em * 2
        self.panel = gui.Vert(
            self.em * 0.7, gui.Margins(margin, margin, margin, margin)
        )

        self.information_panel = gui.Vert(
            self.em * 0.7, gui.Margins(margin, margin, margin, margin)
        )
        self.information_panel.visible = False

        my_lab = gui.Label("Information Panel")
        self.information_panel.add_child(my_lab)

        # fill the panel with gui widgets
        self._build_side_panel()

        # add panel to window
        self.window.add_child(self.panel)
        self.window.add_child(self.information_panel)

        self._thread_killer = thread_killer

        self.window.set_on_layout(self._on_layout)

        self.center = self._point_cloud_data.point_data.to_legacy().get_center()

        self.mouse_control: bool = False

        # set up the view control parameters

        self.diag = np.linalg.norm(
            self._point_cloud_data.point_data.to_legacy().get_max_bound()
            - self._point_cloud_data.point_data.to_legacy().get_min_bound()
        )
        """Initial camera distance based on bounding box size."""
        self.distance = DISTANCE_DEFAULT * self.diag
        """Initial distance based on the bounding box and the default zoom level."""
        self.fov_deg = 60.0
        """Vertical field-of-view in degree"""
        self.azimuth = AZIMUTH_DEFAULT
        """Azimuth of the view. This defines the rotation around the z-axis."""
        self.altitude = ALTITUDE_DEFAULT
        """Altitude of the view. This defines the rotation of the x-axis."""
        self.up = np.array([0.0, 0.0, 1.0])
        """This defines in the up-vector of the view"""
        self.front = np.array([0.0, 1.0, 0.0])
        """This defines in front vector of the view"""
        self.right = np.cross(self.up, self.front)
        """The right vector is orthogonal to the up and front vector"""

        self.look_at = self._point_cloud_data.point_data.to_legacy().get_center()

    def _build_side_panel(self):
        """Build the UI side panel for the control buttons."""
        # load plant set from storage
        load_plant_set_label = gui.Label("Please select a folder with the plant data")
        self.panel.add_child(load_plant_set_label)
        load_plant_set_button = gui.Button("Load plant set")
        load_plant_set_button.tooltip = (
            "select a folder with the plant point cloud data"
        )
        load_plant_set_button.set_on_clicked(self._on_load_plant_set_clicked)
        self.panel.add_child(load_plant_set_button)

        # add vertical space
        self.panel.add_fixed(self.em * 1.2)

        # toggle switches with controls
        select_features_label = gui.Label("Enable/Disable controls")
        self.panel.add_child(select_features_label)

        hand_gesture_tracking = gui.ToggleSwitch("Hand gesture tracking")
        hand_gesture_tracking.tooltip = "Toggle the hand tracking with the camera"
        hand_gesture_tracking.is_on = True
        hand_gesture_tracking.set_on_clicked(self._toggle_hand_tracking_callback)
        self.hand_gesture_show_renderer = gui.ToggleSwitch("Show camera renderer")
        self.hand_gesture_show_renderer.tooltip = "Toggle the camera renderer"
        self.hand_gesture_show_renderer.is_on = True
        self.hand_gesture_show_renderer.set_on_clicked(
            self._toggle_camera_renderer_callback
        )

        time_travel = gui.ToggleSwitch("Time travel")
        time_travel.is_on = True
        time_travel.tooltip = (
            "Toggle the automatic transition through the set of plants"
        )
        time_travel.set_on_clicked(self._toggle_time_travel_callback)

        self.demo_mode_switch = gui.ToggleSwitch("Enable demo mode")
        self.demo_mode_switch.set_on_clicked(self._set_demo_mode)
        self.demo_mode_switch.tooltip = (
            "Enables the demo mode to rotate and zoom the plant on its own."
        )
        self.enable_mouse_control_switch = gui.ToggleSwitch("Enable mouse control")
        self.enable_mouse_control_switch.tooltip = (
            "Enables mouse control to manually interact with the plant using a mouse."
        )
        self.enable_mouse_control_switch.set_on_clicked(self._set_enable_mouse_control)

        self.panel.add_child(hand_gesture_tracking)
        self.panel.add_child(self.hand_gesture_show_renderer)
        self.panel.add_child(time_travel)
        self.panel.add_child(self.demo_mode_switch)
        self.panel.add_child(self.enable_mouse_control_switch)

        # Checkboxes with features
        select_features_label = gui.Label("Select features")
        self.panel.add_child(select_features_label)

        true_color = gui.Checkbox("True color")
        true_color.tooltip = "Toggle between the original color of the plant, and the artificial selected color"
        true_color.checked = self.true_color
        true_color.set_on_checked(self._toggle_true_color_checkbox)

        self.segmented_colors_checkbox = gui.Checkbox("Show segment colors")
        self.segmented_colors_checkbox.enabled = (
            self.true_color
        )  # this feature requires the true color option to be selected
        self.segmented_colors_checkbox.tooltip = "If enabled if the plant is displayed in different colors for each segment (only available if the data provides segmentation data)"
        self.segmented_colors_checkbox.set_on_checked(self._toggle_segmentation_color)

        generate_new_colors = gui.Button("New")
        generate_new_colors.tooltip = (
            "Generate a new color set for the segmentation colors"
        )
        generate_new_colors.set_on_clicked(self._generate_new_segmentation_color_set)
        color_segmentation_row = gui.Horiz()
        color_segmentation_row.add_child(self.segmented_colors_checkbox)
        color_segmentation_row.add_child(generate_new_colors)

        self.leaf_overlay = gui.Checkbox("Information overlay")
        self.leaf_overlay.set_on_checked(self._on_leaf_overlay_checked)

        self.panel.add_child(true_color)
        self.panel.add_child(color_segmentation_row)  # self.segmented_colors_checkbox
        self.panel.add_child(self.leaf_overlay)

        # add vertical space
        self.panel.add_fixed(self.em * 1.2)

        # Developer options
        developer_options_label = gui.Label("Developer Options")
        self.panel.add_child(developer_options_label)

        sky_box_checkbox = gui.Checkbox("Show sky box")
        sky_box_checkbox.set_on_checked(self._set_show_skybox)
        world_frame_checkbox = gui.Checkbox("Show world frame")
        world_frame_checkbox.set_on_checked(self._set_show_world_frame)

        self.show_information_panel = gui.Checkbox("Show information panel")
        self.show_information_panel.set_on_checked(self.set_show_information_panel)

        self.panel.add_child(sky_box_checkbox)
        self.panel.add_child(world_frame_checkbox)

        self.panel.add_child(self.show_information_panel)

        background_color = gui.ColorEdit()
        background_color.color_value = gui.Color(1.0, 1.0, 1.0, 1.0)
        color_label = gui.Label("Select a background color")
        self.panel.add_child(color_label)
        background_color.set_on_value_changed(self.set_background)

        self.panel.add_child(background_color)

        self.plant_color = gui.ColorEdit()
        self.plant_color.color_value = gui.Color(
            self.material_colorless.base_color[0],
            self.material_colorless.base_color[1],
            self.material_colorless.base_color[2],
            self.material_colorless.base_color[3],
        )
        plant_color_label = gui.Label("Select a plant color")
        self.panel.add_child(plant_color_label)
        self.plant_color.set_on_value_changed(self.update_plant_color)
        self.panel.add_child(self.plant_color)

        point_cloud_size_label = gui.Label("Select a size of the points")
        point_cloud_size = gui.Slider(gui.Slider.Type.DOUBLE)
        point_cloud_size.set_limits(1, 10)
        point_cloud_size.int_value = int(POINT_SIZE_DEFAULT)  # set initial value
        point_cloud_size.set_on_value_changed(self.update_plant_point_size)
        self.panel.add_child(point_cloud_size_label)
        self.panel.add_child(point_cloud_size)

        transition_delay_label = gui.Label("Select a delay between two transitions")
        transition_delay_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        transition_delay_slider.set_limits(1, 50)
        transition_delay_slider.int_value = int(TRANSITION_DELAY_DEFAULT)
        transition_delay_slider.set_on_value_changed(self.update_transition_delay)
        self.panel.add_child(transition_delay_label)
        self.panel.add_child(transition_delay_slider)

        transition_duration_label = gui.Label(
            "Select a duration for each transitions (in seconds)"
        )
        transition_duration_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        transition_duration_slider.set_limits(0.5, 20)
        transition_duration_slider.int_value = int(TRANSITION_DURATION_DEFAULT)
        transition_duration_slider.set_on_value_changed(self.update_transition_duration)
        self.panel.add_child(transition_duration_label)
        self.panel.add_child(transition_duration_slider)

        select_day_label = gui.Label("Slide to travel in time.")
        self.select_day_slider = gui.Slider(gui.Slider.Type.INT)
        self.select_day_slider.set_limits(0, self.number_of_plants)
        self.select_day_slider.int_value = 0
        self.select_day_slider.enabled = False
        self.select_day_slider.set_on_value_changed(self._callback_update_selected_day)
        self.panel.add_child(select_day_label)
        self.panel.add_child(self.select_day_slider)

        # add vertical space
        self.panel.add_fixed(self.em * 0.5)
        # adding an empty label otherwise the vertical space is not applied
        self.panel.add_child(gui.Label(""))

    def _on_layout(self, layout_context):
        """This callback is important for open3D to set the layout of the UI.

        If this callback is not added with `self.window.set_on_layout(self.on_layout)` the UI will
        stay black.

        Here we define the basic layout of the widgets that are directly added to the
        window. In our case the scene and the panel widget. Currently, this splits the window in a panel on the left
        and the scene on the right.
        """
        # Use the window's content_rect directly (robust across versions)
        # r contains the size of the window e.g. r=(0, 0), 1920x1080)
        # you can get the values with:
        # r.x (which is 0), r.y (which is 0), r.width (which is 1920), r.height (which is 1080)
        r = self.window.content_rect

        # Reserve a relative width for the panel on the right, 16% is about 300px for 1920p
        panel_width = 0.16 * r.width
        # reserve a relative width for the information panel on the left, same as the other panel
        information_panel_width = 0.16 * r.width
        # Optionally enforce a minimum width for the scene
        scene_width = max(200, r.width - panel_width - information_panel_width)

        # Panel is on the left
        self.panel.frame = gui.Rect(
            # the Rect is defined from the left to right, 0, 0 is on the left bottom corner
            # start_x, start_y, end_x, end_y
            r.x,
            r.y,
            panel_width,
            r.height,
        )

        # Scene fills middle
        self.scene.frame = gui.Rect(r.x, r.y, r.width, r.height)

        # panel is on the right
        self.information_panel.frame = gui.Rect(
            panel_width + scene_width, r.y, r.width, r.height
        )

    def _on_main_window_closing(self):
        """This gets called when the closing button of the window is clicked.

        This kills all thread and quits the app.
        """
        print("Stopping application. Please wait.")
        self._thread_killer.set()
        return True

    def _on_keypress(self, key_event: gui.KeyEvent):
        """Handles all keypresses inside the 3d scene."""
        key = key_event.key
        # handle single presses
        if not key_event.is_repeat and key_event.type == gui.KeyEvent.DOWN:
            match key:
                case 264:  # arrow left
                    self._select_next_day()
                case 263:  # arrow right
                    self._select_previous_day()

        return gui.SceneWidget.HANDLED

    def _on_mouse(self, mouse_event: gui.MouseEvent):
        print(f"Mouse event: {mouse_event.type.name} {mouse_event.type.value}")
        print(f"position: {mouse_event.x} x {mouse_event.y}")
        return gui.SceneWidget.HANDLED

    def _toggle_hand_tracking_callback(self, checked: bool):
        self.hand_gesture_show_renderer.enabled = checked
        self.communication.put(
            {
                "target": "gesture_controller",
                "event": CommunicationEventType.TOGGLE_CAMERA,
                "payload": {"value": checked},
            }
        )

    def _toggle_camera_renderer_callback(self, checked: bool):
        self.communication.put(
            {
                "target": "gesture_controller",
                "event": CommunicationEventType.SHOW_CAMERA_RENDERER,
                "payload": {"value": checked},
            }
        )

    def _toggle_time_travel_callback(self, checked: bool) -> None:
        self.communication.put(
            {
                "target": "transition_controller",
                "event": CommunicationEventType.SET_TIME_TRAVEL,
                "payload": {"value": checked},
            }
        )
        self.select_day_slider.enabled = not checked

    def _toggle_segmentation_color(self, checked: bool) -> None:
        self.communication.put(
            {
                "target": "controller",
                "event": CommunicationEventType.TOGGLE_SEGMENTATION_COLOR,
                "payload": {"value": checked},
            }
        )

    def _generate_new_segmentation_color_set(self) -> None:
        self.communication.put(
            {
                "target": "controller",
                "event": CommunicationEventType.GENERATE_NEW_SEGMENTATION_COLOR_SET,
                "payload": {},
            }
        )

    def _toggle_true_color_checkbox(self, checked: bool) -> None:
        self.true_color = checked
        self.plant_color.enabled = not checked
        self.segmented_colors_checkbox.enabled = checked

    def set_information_overlay(self, enabled: bool) -> None:
        """Enable or disable the leaf/organ information overlay."""
        self.leaf_overlay_enabled = enabled
        self.leaf_overlay.checked = enabled

        # always show/hide the information panel in sync
        self.information_panel.visible = enabled
        self.show_information_panel.checked = enabled
        self.window.set_needs_layout()

        if not enabled:
            # clean up any active highlight and labels
            if self._highlighted_label is not None:
                self._point_cloud_data.reset_color()
                self._highlighted_label = None
            if self._focus_label_name in self._label_store:
                self.remove_3d_label(self._focus_label_name)
            self.clear_content_of_information_panel()
            self._focus_counter = 0
            self._last_focused_organ = None

    def _select_next_day(self):
        """Move to the next day."""
        new_day = self.select_day_slider.int_value + 1
        if new_day <= self.number_of_plants:
            self._callback_update_selected_day(new_day)
            self.select_day_slider.int_value = new_day

    def _select_previous_day(self):
        """Move to the previous day."""
        new_day = self.select_day_slider.int_value - 1
        if new_day >= 0:
            self._callback_update_selected_day(new_day)
            self.select_day_slider.int_value = new_day

    def _callback_update_selected_day(self, selected_day: float) -> None:
        """Request from the controller to show the selected day."""
        self.communication.put(
            {
                "target": "controller",
                "event": CommunicationEventType.SELECT_DAY,
                "payload": {"value": int(selected_day)},
            }
        )

    def _set_show_skybox(self, checked: bool):
        self.scene.scene.show_skybox(checked)

    def _set_show_world_frame(self, checked: bool):
        self.scene.scene.show_axes(checked)

    def _set_demo_mode(self, checked: bool) -> None:
        self.communication.put(
            {
                "target": "controller",
                "event": CommunicationEventType.SET_DEMO_MODE,
                "payload": {"demo_mode": checked},
            }
        )
        self.enable_mouse_control_switch.enabled = not checked

    def _set_projection_parameters(self):
        self.front = self._normalize(self.front)
        self.right = self._normalize(np.linalg.cross(self.up, self.front))
        eye = self.look_at + self.front * self.distance
        self.scene.look_at(self.center, eye, self.up)

    def _on_load_plant_set_clicked(self):
        """Callback for the button to select a folder with plant data.

        @TODO: the path is sometimes wrong, we need to investigate here.
        This show a FileDialog and allows to select a folder.
        """
        # A Dialog is just a widget, so you make its child a layout just like
        # a Window.
        folder_dialog = gui.FileDialog(
            gui.FileDialog.OPEN_DIR, "Select plant set data", self.window.theme
        )

        # We could set the initial path like this.
        # But this feature seems buggy and returns wrong path
        # after the second the dialog is opened a second time.
        # If this gets fixed by open3D we
        # might, can use this feature again.
        # folder_dialog.set_path("plant_point_cloud")

        # set_on_done and set_on_cancel are required for a filedialog
        folder_dialog.set_on_done(self._on_load_plant_set_clicked_done)
        folder_dialog.set_on_cancel(self._on_load_plant_set_clicked_cancel)
        self.window.show_dialog(folder_dialog)

    def _on_load_plant_set_clicked_done(self, path):
        """Callback for a selected path.

        @param: path: the select path, this is where the plant data should be.
        """
        path = Path(path)
        print(f"load plant set from {path}")
        self.communication.put(
            {
                "target": "controller",
                "event": CommunicationEventType.LOAD_POINT_CLOUD_FROM_PATH,
                "payload": {"path": path},
            }
        )
        self.window.close_dialog()

    def _on_load_plant_set_clicked_cancel(self):
        """Callback for the cancel button of the plant set selection dialog."""
        # this closes the current dialog
        self.window.close_dialog()

    def _update_point_cloud(self) -> bool:
        """Update the point cloud data.

        This removes the old data and adds the data again.
        This keeps the UI running.
        """
        self.scene.scene.remove_geometry("plant")
        if self.true_color or self._highlighted_label is not None:
            self.scene.scene.add_geometry(
                "plant", self._point_cloud_data.point_data, self.material_colorless
            )
        else:
            self.scene.scene.add_geometry(
                "plant", self._point_cloud_data.point_data, self.material_colorful
            )
        return True

    def _on_main_window_tick_event(self):
        """This tick keeps the 3d object running."""
        return self._update_point_cloud()

    def _on_label_show_button(self):
        self.add_3d_label("myLabel", "hello World", np.array([0.0, 0.0, 200.0]))

    def _on_label_update_button(self):
        self.update_3d_label("myLabel", "Hello World updated")

    def _on_label_removed_button(self):
        self.remove_3d_label("myLabel")

    def _set_enable_mouse_control(self, checked):
        self.mouse_control = checked
        self.demo_mode_switch.enabled = not checked

    def _on_add_info_content(self):
        widgets = [gui.Label("Info 1"), gui.Label("Info 2"), gui.Label("Info 3")]
        self.add_content_to_information_panel(widgets)

    def _on_remove_info_content(self):
        self.clear_content_of_information_panel()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """Normalize a vector."""
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n <= 1e-12:
            return v
        return v / n

    def add_3d_label(
        self,
        name: str,
        text: str,
        position: np.ndarray,
        scale: float = 1,
        color: gui.Color = None,
    ) -> None:
        """Add a 3d label to the scene.

        @param: name: name of the label. This is used to identify the label has to be unique.
         This name can be used to remove or update the label later.
        @param: text: the text to be added.
        @param: color: the color of the label as gui.Color object, default is black
        @param: scale: the scale of the label. 1 is the original font size, larger scale will reduce sharpness
        @param: position: the position of the label. 3D position as [x, y, z] array
        """
        if name in self._label_store:
            print(
                "Label already exists. Please choose different name or remove label first"
            )
            return

        label: gui.Label3D = self.scene.add_3d_label(position, text)
        if color is not None:
            label.color = color
        else:
            label.color = gui.Color(0, 0, 0)
        label.scale = scale

        self._label_store[name] = label

    def update_3d_label(
        self,
        name: str,
        text: str = "",
        position: np.ndarray = None,
        scale: float = None,
        color: gui.Color = None,
    ) -> None:
        """Update a 3d label to the scene.

        @param: name: name of the label. This is used to identify the label has to be unique.
        @param: text: the text to be added.
        @param: position: the position of the label. 3D position as [x, y, z] array
        @param: scale: the scale of the text
        @param: color: the color of the label as gui.Color object, default is black
        """
        if name in self._label_store:
            label: gui.Label3D = self._label_store[name]

            if label is None:
                print("No with this name found. Are you sure the name is correct?")
                return
            if text != "":
                label.text = text
            if scale is not None:
                label.scale = scale
            if color is not None:
                label.color = color
            if position is not None:
                label.position = position

    def remove_3d_label(self, name: str) -> None:
        """Removes the 3d label with the given name from the scene again.

        If the label with the given name does not exist, nothing will be removed.

        @param: name: name of the label. This is used to identify the label has to be unique.
        """
        if name in self._label_store:
            label = self._label_store[name]
            self.scene.remove_3d_label(label)
            del self._label_store[name]
        else:
            print("Label not found. Are you sure you have the correct name?")

    def set_view_control(self, control_mode: gui.SceneWidget.Controls):
        """Allows to set the type of view control.

        This allows to decide what the user control when using the mouse to interact with the
        3d object.
        possible values are:
        - gui.SceneWidget.FLY
        - gui.SceneWidget.PICK_POINTS
        - gui.SceneWidget.ROTATE_CAMERA
        - gui.SceneWidget.ROTATE_CAMERA_SPHERE
        - gui.SceneWidget.ROTATE_IBL
        - gui.SceneWidget.ROTATE_SUN
        - gui.SceneWidget.ROTATE_MODEL
        """
        self.scene.set_view_controls(control_mode)

    def set_background(self, color: gui.Color):
        """Set the background color of the scene."""
        self.scene.scene.set_background(
            np.array([color.red, color.green, color.blue, color.alpha]), None
        )

    def update_geometry(
        self, updated_geometry: PlantPointCloud, geometry_name="plant"
    ) -> None:
        """Update the geometry of the visualization.

        This allows changing the point cloud that is displayed.
        :param updated_geometry: the new geometry
        :param geometry_name: the name of the geometry, this allows to have multiple geometry in the scene.
         Defaults to "plant"
        :return: None.
        """
        self._point_cloud_data = updated_geometry
        pcd = updated_geometry.point_data
        self.scene.scene.remove_geometry(geometry_name)
        if self.true_color:
            self.scene.scene.add_geometry(geometry_name, pcd, self.material_colorless)
        else:
            self.scene.scene.add_geometry(geometry_name, pcd, self.material_colorful)

    def tick(self):
        """This tick keeps the application alive and has to be called at the desired refresh rate."""
        # this tick keeps the application alive
        self.app.run_one_tick()

        # update the camera view, this applies the changes
        # that were made with the rotate, zoom etc. methods
        # as this does not allow to use the mouse to control,
        # the user can set the enable mouse control checkbox
        # which disables that we apply the projection parameters
        if not self.mouse_control:
            self._set_projection_parameters()
        self.update_focus()

    def rotate(self, azimuth: float = 0, altitude: float = 0) -> None:
        """Rotate the plant by the given angle.

        see also Horizontal coordinate system: https://en.wikipedia.org/wiki/Horizontal_coordinate_system

        This reimplements the ViewControl logic written by Open3D:
        https://github.com/isl-org/Open3D/blob/07fa91f0172b8239630b2e3b718eb8337a64f4b8/cpp/open3d/visualization/visualizer/ViewControl.cpp#L311

        This reimplementation was necessary as it seems there is no way of using the ViewControl together with
        an Open3DScene which seems to be required for be able to use the GUI features of open3D.

        :param azimuth: the degree of rotation of the azimuth (aka x-axis)
        :param altitude: the degree of altitude of the rotation (aka y-axis)
        :return: None
        """
        # calculate the new azimuth and altitude
        azimuth = azimuth * ROTATION_RADIAN_PER_PIXEL
        altitude = altitude * ROTATION_RADIAN_PER_PIXEL

        self.communication.put(
            {
                "target": "sound_controller",
                "event": CommunicationEventType.SOUND_ROTATION_POS,
                "payload": {"value": self.azimuth},
            }
        )

        # calculate horizontal (azimuth) rotation
        self.front = self._normalize(
            self.front * np.cos(azimuth) - self.right * np.sin(azimuth)
        )
        # recompute right to preserve orthogonality (right = up x front)
        self.right = self._normalize(np.cross(self.up, self.front))
        # calculate vertical (altitude) rotation
        self.front = self._normalize(
            self.front * np.cos(altitude) + self.up * np.sin(altitude)
        )
        # recompute up to preserve orthogonality (up = front x right)
        self.up = self._normalize(np.cross(self.front, self.right))

    def zoom(self, distance: float) -> None:
        """Set the distance to the plant (aka. zoom) by the given amount.

        A larger distance decreases the size of the plant and the other way around.

        A zoom level of 0 is infinity and the plant is not visible anymore.

        :param distance: the level of zoom > 0
        :return: None.
        """
        if distance > 0:
            self.distance = distance * self.diag
            self.communication.put(
                {
                    "target": "sound_controller",
                    "event": CommunicationEventType.SOUND_ZOOM_POS,
                    "payload": {"value": self.distance},
                }
            )
        else:
            print("A zoom level smaller than 0 is no supported")

    def update_plant_color(self, color: gui.Color) -> None:
        """Update the color of the point cloud."""
        self.material_colorful.base_color = np.array(
            [color.red, color.green, color.blue, color.alpha]
        )
        # only apply the color, if the setting is selected
        if self.true_color:
            self.scene.scene.update_material(self.material_colorless)
        else:
            self.scene.scene.update_material(self.material_colorful)

    def update_plant_point_size(self, new_size: float) -> None:
        """Update the size of the point cloud points."""
        self.material_colorless.point_size = new_size
        self.material_colorful.point_size = new_size
        if self.true_color:
            self.scene.scene.update_material(self.material_colorless)
        else:
            self.scene.scene.update_material(self.material_colorful)

    def set_show_information_panel(self, visible: bool) -> None:
        """Show / hide the information panel on the right of the screen."""
        self.information_panel.visible = visible
        self.window.set_on_layout(self._on_layout)
        self.window.set_needs_layout()

        # to keep the checkbox in sync if this method is called from somewhere else
        if self.show_information_panel.checked is not visible:
            self.show_information_panel.checked = visible

    def add_content_to_information_panel(self, list_of_widgets: list) -> None:
        """Add a list of widgets to the information panel on the right of the screen."""
        for widget in list_of_widgets:
            self.information_panel.add_child(widget)
        # set needs layout must be called to update the UI
        self.window.set_needs_layout()

    def clear_content_of_information_panel(self) -> None:
        """Clear the Information panel.

        This remove every widget from the panel.
        """
        for widget in self.information_panel.get_children():
            widget.visible = False

        self.window.set_needs_layout()

    def update_transition_delay(self, new_delay: float) -> None:
        """Request from the transition controller to update the transition delay."""
        self.communication.put(
            {
                "target": "transition_controller",
                "event": CommunicationEventType.UPDATE_TRANSITION_DELAY,
                "payload": {"value": new_delay},
            }
        )

    def update_transition_duration(self, new_duration: float) -> None:
        """Request from the transition controller to update the transition delay."""
        self.communication.put(
            {
                "target": "transition_controller",
                "event": CommunicationEventType.UPDATE_TRANSITION_DURATION,
                "payload": {"value": new_duration},
            }
        )

    def on_plant_selected(self, point_cloud_data: PlantPointCloud) -> None:
        """Called only when a real plant is selected.

        Updates JSON and FocusDetector.
        """
        self._point_cloud_data = point_cloud_data
        self._focus_counter = 0
        self._last_focused_organ = None
        self.morphology_json = point_cloud_data.morphology

        if self.morphology_json is None:
            print(f"No JSON data found for date {point_cloud_data.caption_date}")
            return

        plant_ids = self.morphology_json.get_plant_ids()
        self.plant_idx = plant_ids[0]

        timepoints = self.morphology_json.get_timepoints(self.plant_idx)
        self.time_idx = timepoints[0]

        self._focus = FocusDetector(
            point_cloud_data.point_data, knn=200, distance_threshold=20
        )

    def update_limits_select_day_slider(self, new_limit: int):
        """Update the max value of the select day slider.

        This needs to be updated, if a new point could is loaded which has
        a different amount of plants in it. If the slider is not updated, an index
        occurs if the slider is set to a position greater than the number of plants.
        """
        print(f"Set limit to {new_limit}")
        self.number_of_plants = new_limit
        self.select_day_slider.set_limits(0, new_limit)

    def update_focus(self):
        """Update the focus detection and display organ information.

        Uses organ-based detection with a counter to require sustained focus.
        Shows 3D labels only for leaves, information panel for all organs.
        """
        if not self.leaf_overlay_enabled:
            return
        if self._focus is None:
            return

        focused, position, organ_label = self._focus.compute(self.camera)
        cam_pos, _ = _camera_pose(self.camera)

        def _lose_focus():
            if self._highlighted_label is not None:
                self._point_cloud_data.reset_color()
                self._highlighted_label = None
            self._focus_counter = 0
            self._last_focused_organ = None
            if self._focus_label_name in self._label_store:
                self.remove_3d_label(self._focus_label_name)
            self.clear_content_of_information_panel()

        # while zooming out if a distance of camera to leaf is less than 100, the label disappears
        if focused and position is not None:
            distance_to_leaf = np.linalg.norm(position - cam_pos)
            if distance_to_leaf > 100:
                _lose_focus()
                return

        if not focused or organ_label is None:
            _lose_focus()
            return

        if organ_label == self._last_focused_organ:
            self._focus_counter += 1
        else:
            _lose_focus()  # clears highlight + resets counter
            self._focus_counter = 1
            self._last_focused_organ = organ_label

        if self._focus_counter < self._focus_threshold_count:
            return

        organ_type, organ_info = self.morphology_json.get_organ_record(
            self.plant_idx, self.time_idx, organ_label
        )

        if not organ_info:
            _lose_focus()
            return

        if (
            self._focus_counter == self._focus_threshold_count
            and self._highlighted_label != organ_label
        ):
            self._point_cloud_data.reset_color()
            self._apply_leaf_highlight(organ_label)
            self._highlighted_label = organ_label

        if organ_type == "leaf":
            label_text = (
                f"{organ_type.upper()} {organ_label}\n"
                f"Length: {organ_info.get('length_mm', 'N/A')} mm\n"
                f"Width: {organ_info.get('width_mm', 'N/A')} mm"
            )

            if self._focus_label_name not in self._label_store:
                self.add_3d_label(
                    name=self._focus_label_name,
                    text=label_text,
                    position=position,
                    scale=1.5,
                    color=gui.Color(0, 0, 0),
                )
            else:
                self.update_3d_label(
                    name=self._focus_label_name,
                    text=label_text,
                    position=position,
                )
        else:
            if self._focus_label_name in self._label_store:
                self.remove_3d_label(self._focus_label_name)
        self.clear_content_of_information_panel()

        if organ_type == "leaf":
            widgets = [
                gui.Label(f"Leaf {organ_label}"),
                gui.Label(f"Length: {organ_info.get('length_mm', '')} mm"),
                gui.Label(f"Width: {organ_info.get('width_mm', '')} mm"),
                gui.Label(f"Inclination: {organ_info.get('inclination_deg', '')} deg"),
                gui.Label(""),
                gui.Label(f"Chlorophyll: {organ_info.get('chlorophyl', '')}"),
                gui.Label(f"Carotenoid: {organ_info.get('carotinoid', '')}"),
                gui.Label(f"Anthocyanin: {organ_info.get('anthocyanin', '')}"),
                gui.Label(""),
                gui.Label(
                    f"Temperature: {organ_info.get('leaf_temperature_celsius', '')} C"
                ),
                gui.Label(
                    f"Photosynthesis: {organ_info.get('photosynthesis_rate_micromol_m2_s', '')} umol/m2/s"
                ),
            ]
        elif organ_type == "petiole":
            widgets = [
                gui.Label(f"Petiole {organ_label}"),
                gui.Label(""),
                gui.Label(f"Length: {organ_info.get('length_mm', '')} mm"),
                gui.Label(f"Diameter: {organ_info.get('diameter_mm', '')} mm"),
                gui.Label(f"Inclination: {organ_info.get('inclination_deg', '')} deg"),
            ]
        elif organ_type == "internode":
            widgets = [
                gui.Label(f"Internode {organ_label}"),
                gui.Label(""),
                gui.Label(f"Length: {organ_info.get('length_mm', '')} mm"),
                gui.Label(f"Diameter: {organ_info.get('diameter_mm', '')} mm"),
            ]
        else:
            widgets = [
                gui.Label(f"Organ: {organ_type}"),
                gui.Label(f"Label: {organ_label}"),
            ]

        self.add_content_to_information_panel(widgets)

    def _on_leaf_overlay_checked(self, checked: bool):
        """Callback for enabling or disabling leaf overlay mode."""
        self.leaf_overlay_enabled = checked

        if not checked and self._focus_label_name in self._label_store:
            self.remove_3d_label(self._focus_label_name)
            self.clear_content_of_information_panel()

    def _get_leaf_highlight_color(self, label: int) -> np.ndarray:
        """Return a stable unique color for a label."""
        if label not in self._leaf_highlight_colors:
            # convert label into deterministic hue
            hue = (label * 0.618033988749895) % 1.0  # golden ratio spacing

            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

            self._leaf_highlight_colors[label] = np.array([r, g, b], dtype=np.float32)

        return self._leaf_highlight_colors[label]

    def _apply_leaf_highlight(self, label: int) -> None:
        """Paint the focused leaf segment on the current point cloud."""
        color = self._get_leaf_highlight_color(label)
        # clear any previous highlight first
        self._point_cloud_data.paint_one_segment(
            label, PlantPointCloudLabelType.INSTANCE_LABEL, color
        )

    def dispose(self):
        """Dispose the visualizer.

        This closes the window.
        """
        self.app.quit()
        print("GUI window closed.")
