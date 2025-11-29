#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std::chrono_literals;

class ImagePublisherNode : public rclcpp::Node
{
public:
    ImagePublisherNode() : Node("image_publisher_node")
    {
        // 1. 创建发布者，话题名称必须是 /sensor_img (为了配合你的识别节点)
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/sensor_img", 10);

        // 2. 声明参数：默认图片路径改为 "car.png"
        // 这样如果不在命令行指定路径，程序就默认去找工作空间下的 car.png
        // 注意：把下面的路径改成你 Linux 电脑里某张真实存在的图片的路径！
     this->declare_parameter<std::string>("image_path", "/home/jjj/Desktop/car.png");

        // 3. 获取实际要读取的路径
        std::string image_path;
        this->get_parameter("image_path", image_path);

        RCLCPP_INFO(this->get_logger(), "准备读取图片: %s", image_path.c_str());

        // 4. 读取图片 (支持 png, jpg 等常见格式)
        // 注意：cv::imread 读取进来默认是 BGR 格式
        cv_image_ = cv::imread(image_path, cv::IMREAD_COLOR);

        if (cv_image_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "❌ 错误：找不到图片！");
            RCLCPP_ERROR(this->get_logger(), "请确保 '%s' 文件确实在终端运行的目录下 (通常是 ros2_ws)", image_path.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "✅ 图片读取成功！尺寸: %d x %d", cv_image_.cols, cv_image_.rows);
            // 5. 创建定时器，每 1 秒发布一次 (1000ms)
            timer_ = this->create_wall_timer(
                1000ms, std::bind(&ImagePublisherNode::timer_callback, this));
        }
    }

private:
    void timer_callback()
    {
        if (!cv_image_.empty()) {
            // 6. 转换图片格式：OpenCV (cv::Mat) -> ROS 消息 (sensor_msgs::Image)
            std_msgs::msg::Header header;
            header.frame_id = "camera_optical_frame";
            // 更新时间戳，模拟实时采集
            header.stamp = this->get_clock()->now();

            // 使用 cv_bridge 进行转换，编码格式指定为 "bgr8"
            cv_bridge::CvImage cv_bridge_image(header, "bgr8", cv_image_);
            
            try {
                publisher_->publish(*cv_bridge_image.toImageMsg());
                RCLCPP_INFO(this->get_logger(), "已发送图片 car.png ...");
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "转换出错: %s", e.what());
            }
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::Mat cv_image_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImagePublisherNode>());
    rclcpp::shutdown();
    return 0;
}