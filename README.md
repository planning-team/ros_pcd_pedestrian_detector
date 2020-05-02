# ROS package for pedestrian pedection

This ROS package provides a rosnode, running Python environment with a build-in PyTorch model for Pedestrian Detection.
The model is aimed to detect Pedestrians within indoor evrimonent on a data collected with Velodyne VLP16 LiDAR.

The node subscribed to the topic `/velodyne_points`. Recives `/sensor_msgs/PointCloud2` with XYZ
and returns `/sensor_msgs/PointCloud2` with XYZ and "probability" to the topic `/velodyne_points_pedestrians`.

### Usage

To run a rosnode make the following:

1. Clone this repository to catkin environment:

```bash
cd ~/catkin_ws/src
git clone https://github.com/gla-seva/ros_pcd_pedestrian_detector.git
```
2. Download pretrained model:
```bash
wget -O ros_pcd_pedestrian_detector/scripts/model/UNet_best.pth --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PU6-fHTE9n7xFma6vS2d_w4shnkKkKNi'
```
