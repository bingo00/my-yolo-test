#include "yolo.h"

YoloDetect::YoloDetect()
{
}

YoloDetect::YoloDetect(std::string _yaml_file)
{
    config_yaml_file_ = _yaml_file;
    initFromYaml(config_yaml_file_);
}

YoloDetect::YoloDetect(std::string model_cfg_path, std::string model_bin_path, std::string classname)
{
    modelConfiguration = model_cfg_path;
    modelBinary = model_bin_path;
    classname_dir = classname;
}

YoloDetect::~YoloDetect()
{
    if (isInit)
        net.~Net();
}

bool YoloDetect::init()
{
    // read net from model dir
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelBinary);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);

    // when using opencv 4 and data > 20191022, the cuda is support
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

    // layer showup
    std::vector<std::string> layers = net.getLayerNames();
    for(std::string layer : layers)
    {
        printf("%s\n", layer.c_str());
    }

    if (net.empty())
    {
        printf("Could not load net...\n");
        return false;
    }

    std::ifstream classNamesFile(classname_dir);
    if (classNamesFile.is_open())
    {
        std::string className = "";
        while (std::getline(classNamesFile, className)){
            class_name_vec.push_back(className);
            printf("className : %s\n", className.c_str());
        }
    }
    else
    {
        printf("open error!\n");
    }

    isInit = true;

    return true;
}

bool YoloDetect::initFromYaml(const std::string _yaml)
{
    YAML::Node yamlConfig = YAML::LoadFile(_yaml);
    modelConfiguration = yamlConfig["model_cfg"].as<std::string>();
    modelBinary = yamlConfig["model_weight"].as<std::string>();
    classname_dir = yamlConfig["classname_dir"].as<std::string>();
    input_layer_ = yamlConfig["input_layer"].as<std::string>();
    outputBlobName = yamlConfig["output_layer"].as<std::vector<std::string>>();
    appear_object_class_ = yamlConfig["appear_object_class"].as<int>();
    yolo_thresh_ = yamlConfig["yolo_thresh"].as<float>();
    printYamlInfo();
}

void YoloDetect::printYamlInfo()
{
    printf("model_cfg %s\n", modelConfiguration.c_str());
    printf("model_weight %s\n", modelBinary.c_str());
    printf("classname_dir %s\n", classname_dir.c_str());
    printf("input_layer %s\n", input_layer_.c_str());
    for(std::string s : outputBlobName)
    {
        printf("outputBlobName %s\n", s.c_str());
    }
    printf("appear_object_class %d\n", appear_object_class_);
    printf("yolo_thresh_ %f\n", yolo_thresh_);
}

void YoloDetect::detect(cv::Mat &frame)
{
    //　将图像转为 yolo network 的输入
    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1 / 255.F, cv::Size(416, 416), cv::Scalar(), true, false);
    net.setInput(inputBlob, input_layer_);

    std::vector<cv::Mat> detectionMats;
    if (outputBlobName.size() < 1)
    {
        // default using yolo v3
        outputBlobName.push_back("yolo_82");
        outputBlobName.push_back("yolo_94");
        outputBlobName.push_back("yolo_106");
    }

    net.forward(detectionMats, outputBlobName);
    std::vector<double> layersTimings;
    double freq = cv::getTickFrequency() / 1000;
    double time = net.getPerfProfile(layersTimings) / freq;
    // print fps on the frame
    std::ostringstream ss;
    ss << "fps : " << 1.0 / time * 1000; //<< "detection time: " << time << " ms"
    putText(frame, ss.str(), cv::Point(20, frame.rows - 20), 0, 0.5, cv::Scalar(0, 0, 255), 2);

    total_frame_rect.clear();
    boxes_class_index.clear();
    yolo_boxes.clear();

    for (cv::Mat detectionMat : detectionMats)
    {
        for (int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;                                       // 这个是物体概率的起始的位置
            const int probability_size = detectionMat.cols - probability_index;    // 物体分类概率的大小
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index); //　获取其概率指针
            // get object class by soft max 找到物体的名称
            size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            // 获取该物体的置信度
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index); // 定位到confidence并提取
            if (confidence > yolo_thresh_)
            { // 当满足置信度阈值时，进行框取处理
                float x = detectionMat.at<float>(i, 0);
                float y = detectionMat.at<float>(i, 1);
                float width = detectionMat.at<float>(i, 2);
                float height = detectionMat.at<float>(i, 3);
                int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
                int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
                int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
                int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
                cv::Rect object(xLeftBottom, yLeftBottom,
                                xRightTop - xLeftBottom,
                                yRightTop - yLeftBottom);

                // 记录选框信息
                if (objectClass < class_name_vec.size())
                {
                    // total_frame_rect.push_back(object);
                    // boxes_class_index.push_back(objectClass);
                    YoloBox temp_box;
                    temp_box.bbox = object;
                    temp_box.score = confidence;
                    temp_box.objectClass = class_name_vec[objectClass];
                    yolo_boxes.push_back(temp_box);
                }
// 框取并标注物体
#if 1
                // rectangle(frame, object, cv::Scalar(0, 0, 255), 2, 8);

                if (objectClass < class_name_vec.size() && objectClass < appear_object_class_)
                {
                    ss.str("");
                    ss << confidence;
                    cv::String conf(ss.str());
                    cv::String label = cv::String(class_name_vec[objectClass]) + ": " + conf;
                    int baseLine = 0;
                    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    cv::rectangle(frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom), cv::Size(labelSize.width, labelSize.height + baseLine)),
                                  cv::Scalar(255, 255, 255), cv::FILLED);
                    rectangle(frame, object, cv::Scalar(0, 0, 255), 2, 8);
                    cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom + labelSize.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }

#endif
            }
        }
    }
    // cv::imshow("frame", frame);
    // NMS after detection
    yolo_boxes = NMS(0.5, 'u');
}

bool CompareBBox(const YoloBox &a, const YoloBox &b)
{
    return a.score > b.score;
}

std::vector<YoloBox> YoloDetect::NMS(float thresh, char methodType)
{
    std::vector<YoloBox> bbox_nms;
    if (yolo_boxes.size() == 0)
    {
        return bbox_nms;
    }
    std::sort(yolo_boxes.begin(), yolo_boxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(yolo_boxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged)
    {
        // 找到没有进行过merge的
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox)
        {
            all_merged = true;
            continue;
        }
        //当前要保留的记录
        bbox_nms.push_back(yolo_boxes[select_idx]);
        mask_merged[select_idx] = 1;
        //准备比对
        YoloBox select_bbox = yolo_boxes[select_idx];
        float area1 = static_cast<float>(select_bbox.bbox.area());
        float x1 = static_cast<float>(select_bbox.bbox.tl().x);
        float y1 = static_cast<float>(select_bbox.bbox.tl().y);
        float x2 = static_cast<float>(select_bbox.bbox.br().x);
        float y2 = static_cast<float>(select_bbox.bbox.br().y);

        select_idx++;
        //#pragma omp parallel for num_threads(threads_num)
        for (int32_t i = select_idx; i < num_bbox; i++)
        {
            if (mask_merged[i] == 1)
                continue;

            YoloBox &bbox_i = yolo_boxes[i];
            float x = std::max<float>(x1, static_cast<float>(bbox_i.bbox.tl().x));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.bbox.tl().y));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.bbox.br().x)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.bbox.br().y)) - y + 1;
            if (w <= 0 || h <= 0)
                continue; //不重合则继续

            float area2 = static_cast<float>(bbox_i.bbox.area());
            float area_intersect = w * h; //重合的部分

            switch (methodType)
            {
            case 'u':
                if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
                    mask_merged[i] = 1;
                break;
            case 'm':
                if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
                    mask_merged[i] = 1;
                break;
            default:
                break;
            }
        }
    }
    // printf("nms process size : before : %d, after : %d\n", yolo_boxes.size(), bbox_nms.size());
    yolo_boxes.clear();
    return bbox_nms;
}

std::vector<YoloBox> YoloDetect::getYoloBoxes()
{
    return yolo_boxes;
}