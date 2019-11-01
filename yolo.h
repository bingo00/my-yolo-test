#ifndef YOLO_H
#define YOLO_H

#include <iostream>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>


struct YoloBox{
    cv::Rect bbox;
    float score;
    std::string objectClass;
};

class YoloDetect{
public:
    /**
     * @brief  init yolo detector
     * @note   using default path
     * @retval 
     */
    YoloDetect();

    YoloDetect(std::string _yaml_file);
    
    /**
     * @brief  init yolo detector
     * @note   using user path
     * @param  model_cfg_path: the model cfg path that you use 
     * @param  model_bin_path: the model file 
     * @param  classname_dir: the divide classname list file 
     * @retval 
     */
    YoloDetect(std::string model_cfg_path, std::string model_bin_path, std::string classname);
   
    /**
     * @brief  destory the class
     * @note   release the net of dnn
     * @retval 
     */
    ~YoloDetect();
    
    /**
     * @brief  initialize the network and the class voc name
     * @note   init is need before using this class
     * @retval true when success, false when not success
     */
    bool init();

    bool initFromYaml(const std::string _yaml);

    void printYamlInfo();

    /**
     * @brief  detection task of yolo
     * @note   using image and threshold value to detect and classification
     * @param  frame: the input cv::Mat
     * @retval None
     */
    void detect(cv::Mat& frame);

    std::vector<YoloBox> NMS(float thresh, char methodType);

    std::vector<YoloBox> getYoloBoxes();
private:
    cv::String modelConfiguration = "~/darknet/darknet/cfg/yolov3.cfg";
    cv::String modelBinary = "~/darknet/darknet/yolov3.weights";

    std::string classname_dir = "~/temp/my-yolo-surf-tracker/opencv-yolov3/coco.names";

    bool isInit = false;

    cv::dnn::Net net;

    std::vector<std::string> class_name_vec;  

    std::vector<cv::Rect> total_frame_rect;

    std::vector<int> boxes_class_index;
    
    std::vector<YoloBox> yolo_boxes;

    std::string input_layer_ = "data";

    std::vector<std::string> outputBlobName;

    std::string config_yaml_file_ = "~/temp/my-yolo-surf-tracker/config/model_config.yaml";

    int appear_object_class_ = 80;

    float yolo_thresh_ = 0.55;
};

#endif // YOLO_H
