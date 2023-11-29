#!/usr/bin/env python

import numpy as np

np.int = int
np.float = float

import rospy
import ros_numpy
import onnxruntime as onnxrt

from sensor_msgs.msg import PointCloud2
from tools import preprocess
from tools import dataloader_tools as data_loader


class DetectorNode:
    
    def __init__(self,
                 onnx_path: str,
                 sub_topic: str,
                 pub_topic: str) -> None:
        self._onnx_session = onnxrt.InferenceSession(onnx_path)
        self._onnx_input_p = self._onnx_session.get_inputs()[0].name
        self._onnx_input_n = self._onnx_session.get_inputs()[1].name
        
        self._pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=10)
        self._sub = rospy.Subscriber(sub_topic, PointCloud2, self._callback, queue_size=1, buff_size=2**24)
        
    def _callback(self, cloud_msg) -> None:
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
        xyz = ros_numpy.point_cloud2.get_xyz_points(cloud_arr, dtype=np.float32)

        pcd2img = preprocess.Pcd2ImageTransform().fit_fast(xyz)
        data = pcd2img.transform_fast()
        p, n = data_loader.pointnetize(data[:,:,0:4], n_size=[3,3])
        p = p.transpose(-1, -2, 0, 1)[None, 0:4, ...].astype(np.float32)
        n = n.transpose(-1, -2, 0, 1)[None, 0:3, ...].astype(np.float32)
        
        onnx_inputs= {self._onnx_input_p: p,
                      self._onnx_input_n: n}
        pred = self._onnx_session.run(None, onnx_inputs)[0]
        pred = 1. / (1 + np.exp(-pred))
        
        xyzp = pcd2img.inverse_transform((pred[0, 0, :, :] > 0.5).astype(np.float32)).copy()
        dt = np.dtype({'names':['x','y','z', 'probability'], 'formats':[np.float32, np.float32, np.float32, np.float32]})
        cloud_arr_new = np.frombuffer(xyzp.data, dtype=dt)

        cloud_msg_prob = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_arr_new, frame_id='velodyne')
        self._pub.publish(cloud_msg_prob)


if __name__ == '__main__':
    rospy.init_node('pedestrian_detector', anonymous=True)
    onnx_model_path = rospy.get_param("~onnx_model")
    node = DetectorNode(onnx_path=onnx_model_path,
                        sub_topic="velodyne_points",
                        pub_topic="velodyne_points_pedestrians")
    rospy.spin()
