# SSF-SLAM

Real-Time RGB-D Visual SLAM for Complex Dynamic Environments Based on Semantic and Scene Flow Geometric Information.

- A real-time visual SLAM system combining semantic information and scene flow geometry information, called SSF-SLAM, is proposed. In a new thread, lightweight object detection network Mobilenetv3 is deployed to acquire semantic information. This semantic information is tightly coupled with multi-view geometry, which not only ensures accurate recognition of dynamic targets, but also greatly improves the running speed of the system. 
- To overcome the limitations of common geometric information acquisition methods in complex scenes and calculating camera motion, as well as the instability of ORB-SLAM2 feature matching and camera external parameter calculation, this paper designed a scene flow clustering algorithm based on depth and density. This algorithm first combines depth information and distance information for clustering, and then uses the similarity of the static feature point scene flow to further classify, so that the geometric dynamic feature point area can be quickly and accurately calculated. 
- Extensive comparative experiments were conducted on the TUM RGB-D dataset and the Bonn RGB-D dataset. The results demonstrate that the SSF-SLAM can effectively eliminate dynamic objects, achieving more accurate localization and better robustness in complex dynamic environments, and can make the system run in real time even without GPU acceleration.

## Prerequisites

Tested on Ubuntu 18.04 and 20.04.

**Pangolin **

We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

**Eigen3**

Required by g2o (see below). Download and install instructions can be found at: [http://eigen.tuxfamily.org](http://eigen.tuxfamily.org/). **Required at least 3.1.0**.

**Opencv**

We use [OpenCV](http://opencv.org/) to manipulate images and features. Dowload and install instructions can be found at: [http://opencv.org](http://opencv.org/). **Tested with OpenCV 3.3.2**.

**ROS**

We provide examples to process the live input of a RGB-D camera using ROS melodic. 

```
sudo apt-get install ros-melodic-cv-bridge ros-melodic-tf ros-melodic-message-filters ros-melodic-image-transport ros-melodic-nav-msgs ros-melodic-visualization-msgs
```

**Octomap**

We use octomap to store and process sparse 3D point cloud data.

```
sudo apt install liboctomap-dev octovis
sudo apt install ros-melodic-octomap ros-melodic-octomap-mapping ros-melodic-octomap-msgs ros-melodic-octomap-ros ros-melodic-octomap-rviz-plugins 
```

**NCNN**

NCNN needs to be installed in the Thirdparty folder. Dowload and install instructions can be found at: https://github.com/Tencent/ncnn.

## Build

catkin_ws is a ROS workspace.

```
cd ~/catkin_ws/src/SSF-SLAM/
./ThirdpartyBuild.sh

cd ~/catkin_ws
catkin_make --pkg cv_bridge
catkin_make --pkg image_geometry
catkin_make --pkg octomap_server
catkin_make --pkg ssf-slam
```

Find how to install SSF-SLAM and its dependencies here: **[Installation instructions]()**.

## Run datasets examples

- Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and unzip it. We suggest you download rgbd_dataset_freiburg3_walking_xyz.

- We provide run_tum_walking_xyz script to run TUM example. Change PATH_TO_SEQUENCE and PATH_TO_SEQUENCE/associate.txt in the run_tum_walking_xyz to the sequence directory that you download before, then execute the following command in a new terminal. Execute:

  ```
  cd sg-slam
  ./run_tum_walking_xyz.sh
  ```

## Acknowledgement

Thanks for the great work: [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2 ), [DS-SLAM](https://github.com/ivipsourcecode/DS-SLAM), [SG-SLAM](https://github.com/silencht/SG-SLAM), [RDS-SLAM](https://github.com/yubaoliu/RDS-SLAM), and [ORBSLAM2_with_pointcloud_map](https://github.com/gaoxiang12/ORBSLAM2_with_pointcloud_map,).

