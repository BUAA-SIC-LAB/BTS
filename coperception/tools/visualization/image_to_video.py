import cv2
import os
from pathlib import Path

def create_video_from_images(image_dir, output_path, fps=10, scale_factor=0.5):
    """
    将指定目录下的图片按名称排序并生成视频
    
    参数:
        image_dir (str): 图片目录路径
        output_path (str): 输出视频路径
        fps (int): 视频帧率
        scale_factor (float): 缩放因子，用于调整视频尺寸
    """
    # 获取所有图片文件并按名称排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"在目录 {image_dir} 中没有找到图片文件")
        return
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    height, width = first_image.shape[:2]
    
    # 计算缩放后的尺寸
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # 创建视频写入器，使用H.264编码器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 使用H.264编码
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # 处理每张图片
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            # 调整图片大小
            frame = cv2.resize(frame, (new_width, new_height))
            video_writer.write(frame)
            print(f"处理图片: {image_file}")
    
    # 释放资源
    video_writer.release()
    print(f"视频已生成: {output_path}")

if __name__ == '__main__':
    # 设置图片目录和输出路径
    image_dir = "/home/liuzhenghao/nuScenes_Carla_multi_drone/jupyter/output/scene64_frames"
    output_path = os.path.join(image_dir, "output_video.mp4")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 生成视频，scale_factor=0.5 表示将视频尺寸缩小到原来的一半
    create_video_from_images(image_dir, output_path, fps=10, scale_factor=0.3) 