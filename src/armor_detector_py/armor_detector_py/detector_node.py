#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch  # <--- 加上这一行！
from ultralytics import YOLO
from aim_interfaces.msg import AimInfo 
import os

class ArmorDetectorNode(Node):
    def __init__(self):
        super().__init__('armor_detector_node')
        
        # 1. 订阅与发布
        self.sub = self.create_subscription(Image, '/sensor_img', self.img_callback, 10)
        self.pub = self.create_publisher(AimInfo, '/aim_target', 10)
        self.bridge = CvBridge()
        
        # 2. 加载模型
        # 获取当前脚本所在目录，确保能找到 best.pt
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # 这一行必须是你的绝对路径！不能有 os.path.dirname 那些东西了
        model_path = '/home/jjj/Desktop/ros2_ws/src/armor_detector_py/armor_detector_py/best.pt'
        self.get_logger().info(f'正在加载模型: {model_path} ...')
        try:
            self.model = YOLO(model_path)
            self.get_logger().info('✅ YOLOv8 模型加载成功！(PyTorch Backend)')
        except Exception as e:
            self.get_logger().error(f'❌ 模型加载失败: {e}')

        # 3. 初始化相机内参 (题目给定)
        self.K = np.array([[1462.3697, 0, 398.59394],
                           [0, 1469.68385, 110.68997],
                           [0, 0, 1]], dtype=np.float64)
        self.D = np.array([0.003518, -0.311778, -0.016581, 0.023682, 0.0000], dtype=np.float64)

        # 4. 定义装甲板 3D 坐标 (宽16cm, 高8cm)
        # 顺序: 左上, 右上, 右下, 左下
        hw, hh = 0.08, 0.04
        self.obj_pts = np.array([
            [-hw, -hh, 0], 
            [hw, -hh, 0],  
            [hw, hh, 0],   
            [-hw, hh, 0]   
        ], dtype=np.float64)

        # 5. 坐标转换所需的旋转矩阵 (预先计算好)
        r_pitch = 60 * np.pi / 180.0
        r_yaw = 20 * np.pi / 180.0
        # Ry (绕Y轴旋转)
        Ry = np.array([[np.cos(r_pitch), 0, np.sin(r_pitch)],
                       [0, 1, 0],
                       [-np.sin(r_pitch), 0, np.cos(r_pitch)]])
        # Rz (绕Z轴旋转)
        Rz = np.array([[np.cos(r_yaw), -np.sin(r_yaw), 0],
                       [np.sin(r_yaw), np.cos(r_yaw), 0],
                       [0, 0, 1]])
        # 复合旋转 R = Rz * Ry
        self.R_robot = Rz @ Ry
        # 平移 T
        self.T_robot = np.array([[0.08], [0.0], [0.05]])

    def img_callback(self, msg):
        try:
            # 将 ROS 图片转换为 OpenCV 格式
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'CV Bridge 转换错误: {e}')
            return

        # --- YOLO 推理 ---
        # verbose=False 防止终端刷屏
        results = self.model(frame, verbose=False)
        
        # 遍历推理结果
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                # 策略: 只取置信度最高的一个目标
                # 找到 conf 最大的索引
                max_conf_idx = torch.argmax(boxes.conf).item() if hasattr(boxes.conf, 'cpu') else np.argmax(boxes.conf.cpu().numpy())
                
                best_box = boxes[max_conf_idx]
                conf = float(best_box.conf)
                cls_id = int(best_box.cls)

                if conf > 0.5: # 阈值过滤
                    # 获取方框坐标 xyxy
                    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                    
                    # 画框 (可视化)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # --- PnP 解算 ---
                    # 因为是检测模型，我们用方框的四个角近似装甲板的四个角
                    # 顺序必须对应: 左上, 右上, 右下, 左下
                    img_pts = np.array([
                        [x1, y1], 
                        [x2, y1], 
                        [x2, y2], 
                        [x1, y2]
                    ], dtype=np.float64)
                    
                    # 核心解算函数
                    success, rvec, tvec = cv2.solvePnP(self.obj_pts, img_pts, self.K, self.D)
                    
                    if success:
                        # --- 坐标转换 (相机系 -> 机器人系) ---
                        # P_robot = R * P_cam + T
                        P_cam = tvec.reshape(3, 1)
                        P_robot = self.R_robot @ P_cam + self.T_robot
                        
                        # 提取坐标 (米 -> 毫米)
                        x_mm = int(P_robot[0, 0] * 1000)
                        y_mm = int(P_robot[1, 0] * 1000)
                        z_mm = int(P_robot[2, 0] * 1000)
                        
                        # --- 发布消息 ---
                        msg_out = AimInfo()
                        msg_out.type = 7 # 题目要求固定输出 7
                        msg_out.coordinate = [x_mm, y_mm, z_mm]
                        self.pub.publish(msg_out)
                        
                        # 在图上显示距离
                        label = f"Target: {cls_id} Dist: {x_mm}mm"
                        cv2.putText(frame, label, (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        self.get_logger().info(f"发布目标: Type=7, Pos=[{x_mm}, {y_mm}, {z_mm}]")

        # 显示画面
        cv2.imshow("Python YOLO Detector", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArmorDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import torch # 延迟导入，防止某些环境冲突
    main()