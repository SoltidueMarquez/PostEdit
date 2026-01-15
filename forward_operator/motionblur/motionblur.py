import numpy as np
import cv2

class Kernel:
    """
    用于生成运动模糊核 (Motion Blur Kernel) 的类。
    
    该类的作用是模拟相机或物体运动产生的模糊效果。它生成一个点扩散函数 (PSF)，
    即一个表示运动轨迹的卷积核矩阵。
    
    参数:
    - size: 核的大小，通常为 (height, width) 的元组，建议使用奇数（如 (31, 31)）。
    - intensity: 运动模糊的强度。在这里对应于运动轨迹（直线）的长度。
    - angle: 运动的方向角度（单位：度，范围 0-180）。如果未提供，将随机生成。
    """
    def __init__(self, size=(31, 31), intensity=3.0, angle=None):
        self.size = size
        self.intensity = intensity
        
        # 如果没有指定角度，则随机生成一个 0 到 180 度之间的角度
        if angle is None:
            self.angle = np.random.uniform(0, 180)
        else:
            self.angle = angle
        
        # 生成模糊核矩阵 (Numpy array)
        self.kernelMatrix = self._generate_motion_blur_kernel()

    def _generate_motion_blur_kernel(self):
        """
        根据指定的强度和角度生成线性运动模糊核。
        """
        h, w = self.size
        # 创建一个全黑（全 0）的背景
        kernel = np.zeros((h, w), dtype=np.float32)
        
        # 计算核的中心位置
        center = (w // 2, h // 2)
        
        # 运动轨迹的长度 (intensity)
        # 确保长度至少为 1
        length = max(1.0, self.intensity)
        
        # 将角度转换为弧度，用于三角函数计算
        angle_rad = np.deg2rad(self.angle)
        
        # 计算相对于中心点的偏移量
        # dx, dy 决定了直线的斜率和长度
        dx = (length / 2) * np.cos(angle_rad)
        dy = (length / 2) * np.sin(angle_rad)
        
        # 计算直线的两个端点坐标 (OpenCV 坐标系中 x 为水平，y 为垂直)
        # p1 和 p2 是以中心点对称分布的
        p1 = (int(center[0] - dx), int(center[1] - dy))
        p2 = (int(center[0] + dx), int(center[1] + dy))
        
        # 在矩阵中画一条白色 (1.0) 的直线，线条宽度为 1
        # 这条线代表了物体在曝光时间内的运动轨迹
        cv2.line(kernel, p1, p2, 1.0, thickness=1)
        
        # 对核进行归一化处理。
        # 归一化的目的是为了在进行卷积（模糊处理）时保持图像的总亮度不变。
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            # 如果因为强度太小而没有画出线，则在中心点设为 1.0 (即不产生模糊)
            kernel[center[1], center[0]] = 1.0
            
        return kernel

    def get_kernel(self):
        """
        返回生成的模糊核矩阵。
        """
        return self.kernelMatrix
