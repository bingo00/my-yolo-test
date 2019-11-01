## demo of yolov3 and opencv4, using CUDA backend
this project is just a demo project for using yolov3 and opencv4 cuda backend

### prepare
make sure you install the opencv4, and compile with cuda on.
you can refer to my csdn if necessary to install opencv4 (in Chinese, sry)
[opencv4 install]https://blog.csdn.net/beingod0/article/details/102860779

reference https://pjreddie.com/darknet/yolo/

download weight from 

```
wget https://pjreddie.com/media/files/yolov3.weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
```
if perfer, place in model or wherever you want

then edit
```
config/model_config.yaml
```
replace the dir
```
model_cfg
model_weight
classname_dir
```
to your own directory, then the prepare is completed

### compile
```
mkdir build
cd build
cmake ..
make
```