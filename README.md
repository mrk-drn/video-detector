# Video Detector

A lightweight real-time object [detection](https://github.com/mrk-drn/video-detector/blob/master/src/VideoDetection.cpp) on camera input.

## Requirements
* Camera (e.g. simple laptop front camera)
* C++ 11/14(+)
* Windows (Linux)
* OpenCV 4.5.0

## Detection
* Faces
* Cats
* Further objects might follow

<img src="demo_images/football.png" width="600">

<img src="demo_images/leonardo_dicaprio_cat.png" width="600">


Pictures of the detected objects are stored in a newly created detections folder.
```C++
struct stat dirInfo;
if (stat("detections", &dirInfo) != 0) {
  if (_mkdir("detections") != 0) {
    std::cout << "Could not create directory for detections." << std::endl;
    return -1;
  }
}
...
imwrite("detections/face_" + std::to_string(i) + ".png", imgCrop);
```

## Reference
The pretrained cascade classifiers are from [OpenCV](https://github.com/opencv/opencv/tree/3.4/data/haarcascades).

