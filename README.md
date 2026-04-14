# 3D Plant PointClouds as interactive generative ART

The project aims to build a new and interactive way to experience 3D representations of plants.
The visualization of the point clouds of plants allows the user to discover, learn, and marvel
at plants in a completely new way. For that, we are developing a real-time software from scratch
that lets the user control the plant representation using hand gestures.
The software’s core purpose is to interactively and artistically manipulate
the point cloud and display features of the plant, while it grows to an older plant over time,
using transitions between several point clouds.

*This software was developed in a team for a client as a University project. The plant data is still under active research and will be released at a later point in time.*

## Features
**Already implemented features**
- cross-platform GUI using open3D
- manipulating point cloud view through hand gestures (rotation, zoom, and tilt)
- information overlay as labels on point cloud constituents and using a side panel
- point cloud growth

**Planned features**
- post-gesture continuation of viewpoint changes (rotation, zoom, and tilt)
- the addition of sound congruent with point cloud interaction
- implosion/explosion of the point cloud
- manipulation of the point cloud color

## Getting started
This project uses UV for package management. Please refer
to [the official UV website](https://docs.astral.sh/uv/getting-started/installation/) for 
install instructions. 

### UV setup
- if necessary run `uv python install 3.12.0` and after that `uv python pin 3.12.0`
- run `uv sync`
- Select in pycharm as local interpreter 
   - `uv`,
   - `use existing one`
   - select as `uv env use` the venv `app/.venv/bin/python3.12`
- to add more packages to uv, run `uv add <packagename>`

### Running the Software
- Connect the OAK-D camera to your PC
- place your point cloud dataset in /plant_point_cloud/data
- run `uv run -m plant_point_cloud`

### PyCharm Run Configuration
- click on `edit configuration`
- press the `+` on the left
- select `uv run`
- select `module`
- for linux with wayland only(?) add `export XDG_SESSION_TYPE=x11` as environment var (for open3d)
- enter `plant_point_cloud` as module name
- select the correct uv environment (in case there are multiple, check that the Python version is correct)

## Supported Operating Systems
- Linux (x86)
- macOS (arm64)
  - - Minimum recommended macOS version: macOS 15 or newer
- Windows (x86)
  - Windows support is limited due to OpenGL compatibility issues with Open3D. 
  - Visualization may fail depending on OpenGL version support.

## Required Python versions
This project requires Python version 3.12.12. Older versions and newer versions are not supported due to limited support of the dependencies. This could change in the future,  for newer versions of the dependencies.

## Camera Requirements
- An OAK device is required (e.g., OAK-D)

## USB Requirement
- A USB-C port is recommended 
- USB-A may work via an adapter, but can lead to:
   - Unstable connection
   - Insufficient power supply
  
## Raspberry Pi Compatibility
-  Raspberry Pi support is limited due to USB-C limitations.

## Point cloud dataset requirements
This software is able to load point cloud data from a directory. It is built for loading a set
of cloud clouds. If only a single file should be loaded, please create a folder with a single `.ply` file.
The data structure should look like this: 
```
/data
├── soybean_point_clouds
│   ├── 2025-10-17_1820_101464_processed.ply
│   ├── 2025-10-20_1603_101464_processed.ply
│   ├── 2025-10-22_1654_101464_processed.ply
│   ├── 2025-10-24_1452_101464_processed.ply
│   ├── 2025-10-27_1301_101464_processed.ply
│   ├── 2025-10-29_1358_101464_processed.ply
│   ├── 2025-10-31_1518_101464_processed.ply
│   ├── 2025-11-03_1614_101464_processed.ply
│   ├── 2025-11-05_1507_101464_processed.ply
│   ├── 2025-11-10_1859_101464_processed.ply
```

If the folder name should be the plant name and each .ply file should start
with the capture date in the format `YYYY-MM-DD_[...]`. If this is provided, the software will parse the capture date and
will display correct dates in the GUI. If no date is provided, the software will use a fallback value.

## Audio Setup
This software sends OSC commands to localhost on port 2228.
This requires, e.g., having [Cardinal](https://cardinal.kx.studio/) running. 

In Cardinal, open the prepared patch in `/data/cardinal_patches/osc_plant_soundscape.vcv` and enable 
OSC control via `Engine -> Enable OSC remote control`. Select port `2228`. 

This could be extended to a custom headless build and could also, with some minor code adaptations, 
run on an external PC in the same network.

## Contributors
- Many thanks to Julian Röhrig for creating this amazing cardinal patch.

## License
This project is licensed under MIT. 

This project depends on a slightly modified version of [`depthai_hand_tracker`](https://github.com/geaxgx/depthai_hand_tracker/) (MIT license)
