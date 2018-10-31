Structure From Motion
====

  A single camera 3D reconstruction system for drones, created for the UAVConcordia swarm project.
  We start by detecting and tracking features between frames of the camera feed. Then using the
  known motion of the drone, we are able to identify their position in 3D space. Eventually we will
  use an Inertial Navigation System to track the drone's position, and create a complete 3D point cloud
  of the environment.
  
  
## The Math

The basic equation of the system is this:

![distance=focallength\disparity](https://latex.codecogs.com/svg.latex?distance=\frac{focal\\,length*velocity}{disparity})

Where _distance_ is the distance from the camera to the tracked point, _disparity_ is the 3D vector
of the pixel
disparities in the x, y, and radial directions, and _velocity_ is the 3D velocity vector of the camera.

## Requirements
  
  * Numpy:
  
     ```
     pip install numpy
     ```
  * OpenCV 3+
  
    ```
    pip install pyopencv
    ```
    
## Authors
* Drew Wagner - _Initial Work_

See also the list of [contributors](https://github.com/m4lb0t/uav-swarm-project/contributors) who participated in this project.

## Additional Reading
https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
https://ieeexplore.ieee.org/document/6749315
https://etd.ohiolink.edu/rws_etd/document/get/dayton1366386933/inline