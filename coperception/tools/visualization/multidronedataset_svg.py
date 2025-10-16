import os
import numpy as np
import svgwrite  # 用于生成纯净的SVG

def generate_editable_svg_only_points(sample, pred_len, voxel_size, area_extents, agent, sample_index):
    """
    生成仅包含黑色点云的SVG（无白色背景元素），确保PPT中只能修改点云颜色
    """
    # 解析样本数据，提取2D体素点云
    (padded_voxel_points, _, _, _, _, _, _, _, _, _, _, _) = sample
    lidar_2d_img = np.max(padded_voxel_points.reshape(256, 256, 13), axis=2)
    height, width = lidar_2d_img.shape  # 256x256

    # 创建SVG绘图对象（只定义尺寸，不添加背景元素）
    svg_path = f"./voxel-svg/{agent}"
    os.makedirs(svg_path, exist_ok=True)
    svg = svgwrite.Drawing(
        filename=f"{svg_path}/agent{agent}_sample{sample_index}.svg",
        size=(f"{width}px", f"{height}px"),  # 尺寸与点云一致
        viewBox=f"0 0 {width} {height}"  # 确保缩放时坐标对应正确
    )

    # 只绘制黑色点云（过滤无效值，避免冗余点）
    threshold = 0.1  # 过滤阈值，可根据数据调整（值>0的点视为有效点云）
    for y in range(height):
        for x in range(width):
            value = lidar_2d_img[y, x]
            if value > threshold:  # 只保留有效点云
                # 每个点绘制为圆形，显式设置为黑色
                svg.add(svg.circle(
                    center=(x, y),  # 坐标对应像素位置
                    r=0.5,  # 点的大小（可调整，建议0.3-0.7）
                    fill="black",  # 点云颜色，PPT中可修改
                    stroke="none"  # 无描边，避免多余线条
                ))

    # 保存SVG
    svg.save()


# 主函数（调用方式与之前一致）
if __name__ == "__main__":
    import os
    from coperception.datasets import V2XSimDet
    from coperception.configs import Config, ConfigGlobal

    split = "test"
    config = Config(binary=True, split=split, use_vis=True)
    config_global = ConfigGlobal(binary=True, split=split)
    data_root = "/home/liuzhenghao/dataset/multi-drone-det"
    
    num_agents = 5
    num_frames = 10
    start = 10
    multidroneDet = V2XSimDet(
        dataset_roots=[f"{data_root}/{split}/agent{i}" for i in range(num_agents)],
        config=config,
        config_global=config_global,
        split=split,
        val=True,
        bound="both"
    )

    for i in range(start, start + num_frames):
        for cnt, sample in enumerate(multidroneDet[i]):
            generate_editable_svg_only_points(
                sample, 
                multidroneDet.pred_len, 
                multidroneDet.voxel_size,
                multidroneDet.area_extents, 
                cnt, 
                i
            )