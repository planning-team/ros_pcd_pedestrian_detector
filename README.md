# ROS package for pedestrian pedection

This ROS package provides a rosnode, running Python environment with a build-in PyTorch model for Pedestrian Detection.
The model is aimed to detect Pedestrians within indoor evrimonent on a data collected with Velodyne VLP16 LiDAR.

The node subscribed to the topic `/velodyne_points`. Recives `/sensor_msgs/PointCloud2` with XYZ
and returns `/sensor_msgs/PointCloud2` with XYZ and "probability" to the topic `/velodyne_points_pedestrians`.

![Screenshot from 2020-05-02 13-37-17](https://user-images.githubusercontent.com/38633753/80861996-79779400-8c7a-11ea-8d8b-32930f80fe2f.png)

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

3. Download example .bag from [here](https://lcas.lincoln.ac.uk/nextcloud/index.php/s/KK14C3DZ0ouQx6I)

4. Build the package with:
```bash
cd ~/catkin_ws/
catkin_make
```

5. Run the following nodes (all in separate terminals):   
  a. Start a roscore
```bash
roscore
```
  b. play the bag file:
```bash
rosbag play PATH_TO_YOUR_BAG_FILE
```
  c. This nodelet reads raw data from the /velodyne_packets ROS topic, converts to /sensor_msgs/PointCloud2 format, and republishes to /velodyne_points
```bash
rosrun nodelet nodelet standalone velodyne_pointcloud/CloudNodelet _model:="VLP16" _calibration:="VLP16db.yaml"
```
  d. Our pedestrian detector:
```bash
rosrun ros_pcd_pedestrian_detector pedestrian_detector.py
```
  e. Run RViz to visualise the PointClouds
```bash
rosrun rviz rviz -f velodyne
```
In RViz select Add -> By topic -> /velodyne_points_pedestrians -> PointCloud2


