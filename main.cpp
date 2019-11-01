#include "yolo.h"
#include <unistd.h>


int main(int argc, char **argv)
{
    char video_name[120] = "/home/moi/Videos/201907091341Sr300/front_video_10.avi";
    std::cout << argc << std::endl;
    cv::VideoCapture cap;
    if (argc > 1)
    {
        sprintf(video_name, "%s", argv[1]);
        cap = cv::VideoCapture(argv[1]);
        std::cout << video_name << std::endl;
    }
    else
    {
        printf("no input video source, opening camera\n");
        cap = cv::VideoCapture(0);
    }
    if (!cap.isOpened())
    {
        std::cout << "open video/camera fail!!" << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    cv::Mat current_frame;

    cv::Mat depth;

    bool flag_pause = false;

    // video recorder
    char savevideo_name[40];
    sprintf(savevideo_name, "~/Videos/test%d.avi", 1);
    cv::VideoWriter video(savevideo_name, cv::VideoWriter::fourcc('M', 'P', '4', '2'), 10, cv::Size(640, 480));
    
    // please modify to your own config file dir
    std::string config_file = "/home/moi/temp/my-yolo-surf-tracker/config/model_config.yaml";
    
    YoloDetect yolo(config_file);
    yolo.init();
    
    int count = 0;
    while (true)
    {
        if (!flag_pause)
        {
            cap >> current_frame;
            frame = current_frame(cv::Rect(0, 0, 640, 480));

            // depth = current_frame(cv::Rect(640, 0, 640, 480));
#if 1
            double t = (double)cv::getTickCount();
            yolo.detect(frame);
            std::vector<cv::Rect> boxes;

            std::vector<YoloBox> detboxes = yolo.getYoloBoxes();

            cv::imshow("image", frame );
            video.write(frame);
            // std::cout << " all time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"
            //         << std::endl;
#endif
        }
        char c = cv::waitKey(1);
        if (c == 27)
        {
            break;
        }
        if (c == ' ')
        {
            flag_pause = !flag_pause;
        }
        // usleep(100000);
    }

    video.release();
    return 1;
}
