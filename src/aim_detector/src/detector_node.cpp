#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <aim_interfaces/msg/aim_info.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <opencv2/dnn.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace std::chrono_literals;

class ArmorDetectorNode : public rclcpp::Node
{
public:
    ArmorDetectorNode() : Node("armor_detector_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sensor_img", 10,
            std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<aim_interfaces::msg::AimInfo>("/aim_target", 10);

        // -------------------------------------------------------------
        // 模型路径 (请确认文件已覆盖为“普通检测版”的 onnx)
        std::string model_path = "/home/jjj/Desktop/ros2_ws/src/aim_detector/model/best.onnx";
        // -------------------------------------------------------------
        
        // 必须和你导出时的尺寸一致 (416)
        input_size_ = cv::Size(416, 416); 

        try {
            net_ = cv::dnn::readNetFromONNX(model_path);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            RCLCPP_INFO(this->get_logger(), "YOLO-Detect (普通版) 模型加载成功!");
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "模型加载失败: %s", e.what());
        }

        camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            1462.3697, 0, 398.59394,
            0, 1469.68385, 110.68997,
            0, 0, 1);
        dist_coeffs_ = (cv::Mat_<double>(1, 5) << 
            0.003518, -0.311778, -0.016581, 0.023682, 0.0000);

        // 装甲板 3D 坐标
        double hw = 0.16 / 2.0; 
        double hh = 0.08 / 2.0; 
        object_points_ = {
            cv::Point3f(-hw, -hh, 0), cv::Point3f(hw, -hh, 0),  
            cv::Point3f(hw, hh, 0),   cv::Point3f(-hw, hh, 0)   
        };
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); }
        catch (cv_bridge::Exception& e) { return; }

        cv::Mat frame = cv_ptr->image;
        if(frame.empty()) return;

        std::vector<cv::Point2f> image_points;
        int detected_type = 0; // 识别到的装甲板 ID

        // 调用普通检测函数
        bool found = detect_armor_yolo(frame, image_points, detected_type);

        if (found) {
            cv::Mat rvec, tvec;
            // 使用方框的四角进行 PnP
            bool success = cv::solvePnP(object_points_, image_points, camera_matrix_, dist_coeffs_, rvec, tvec);

            if (success) {
                cv::Mat p_robot = transform_camera_to_robot(tvec);
                int x_mm = (int)(p_robot.at<double>(0, 0) * 1000);
                int y_mm = (int)(p_robot.at<double>(1, 0) * 1000);
                int z_mm = (int)(p_robot.at<double>(2, 0) * 1000);

                auto result_msg = aim_interfaces::msg::AimInfo();
                result_msg.coordinate = { (int16_t)x_mm, (int16_t)y_mm, (int16_t)z_mm };
                result_msg.type = 7; 
                publisher_->publish(result_msg);

                // 画框
                for(int i=0; i<4; i++) cv::line(frame, image_points[i], image_points[(i+1)%4], cv::Scalar(0,255,0), 2);
                
                std::string text = "ID:" + std::to_string(detected_type) + " Dist:" + std::to_string(x_mm);
                cv::putText(frame, text, image_points[0], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
            }
        }
        cv::imshow("YOLO-Detect", frame);
        cv::waitKey(1);
    }

    // --- 核心：普通检测模型解析 ---
    bool detect_armor_yolo(const cv::Mat& frame, std::vector<cv::Point2f>& detected_points, int& type_id) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, input_size_, cv::Scalar(), true, false);
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        cv::Mat output_data = outputs[0];
        
        // 处理维度问题 (适应 OpenCV 4.5.4)
        int dimensions = output_data.size[1]; 
        int rows = output_data.size[2];       
        
        if (dimensions > rows) {
            output_data = output_data.reshape(1, dimensions);
            cv::transpose(output_data, output_data);
            // 交换后：rows=8400, dimensions=4+类别数
            rows = output_data.size[0];
            dimensions = output_data.size[1];
        } else {
            // 如果原本就是 [8400, 19]
             rows = output_data.size[0];
             dimensions = output_data.size[1];
        }
        
        float* data = (float*)output_data.data;
        float x_factor = (float)frame.cols / input_size_.width;
        float y_factor = (float)frame.rows / input_size_.height;

        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;

        // 遍历 8400 个框
        // 格式: [x, y, w, h, score1, score2, ... score14]
        for (int i = 0; i < rows; ++i) {
            float* classes_scores = data + 4; // 第4位之后全是类别分数
            
            // 找出分数最高的那个类别
            // 假设你有 14 个类别，dimensions 应该是 4 + 14 = 18
            int num_classes = dimensions - 4; 
            cv::Mat scores(1, num_classes, CV_32F, classes_scores);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > 0.5) { 
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back((float)max_class_score);
                class_ids.push_back(class_id_point.x);
            }
            data += dimensions;
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.45, indices);

        if (indices.size() > 0) {
            int idx = indices[0];
            type_id = class_ids[idx];
            cv::Rect best_box = boxes[idx];

            // 提取四角 (近似 PnP)
            detected_points.clear();
            detected_points.push_back(cv::Point2f(best_box.x, best_box.y)); // 左上
            detected_points.push_back(cv::Point2f(best_box.x + best_box.width, best_box.y)); // 右上
            detected_points.push_back(cv::Point2f(best_box.x + best_box.width, best_box.y + best_box.height)); // 右下
            detected_points.push_back(cv::Point2f(best_box.x, best_box.y + best_box.height)); // 左下
            
            return true;
        }
        return false;
    }

    cv::Mat transform_camera_to_robot(const cv::Mat& t_camera) {
        double r_p = 60 * M_PI / 180.0;
        double r_y = 20 * M_PI / 180.0;
        cv::Mat Ry = (cv::Mat_<double>(3, 3) << cos(r_p), 0, sin(r_p), 0, 1, 0, -sin(r_p), 0, cos(r_p));
        cv::Mat Rz = (cv::Mat_<double>(3, 3) << cos(r_y), -sin(r_y), 0, sin(r_y), cos(r_y), 0, 0, 0, 1);
        return Rz * Ry * t_camera + (cv::Mat_<double>(3, 1) << 0.08, 0.0, 0.05);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<aim_interfaces::msg::AimInfo>::SharedPtr publisher_;
    cv::Mat camera_matrix_, dist_coeffs_;
    std::vector<cv::Point3f> object_points_;
    cv::dnn::Net net_;
    cv::Size input_size_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}