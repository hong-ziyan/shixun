import rasterio
import numpy as np
import matplotlib.pyplot as plt


def process_sentinel2_image(input_path):
    try:
        # 读取影像元数据
        with rasterio.open(input_path) as src:
            print(f"影像形状: {src.shape}")
            print(f"波段数量: {src.count}")
            print(f"波段描述: {src.descriptions}")

            # 根据波段数量选择RGB组合
            if src.count >= 8:  # 完整数据
                rgb_indices = [4, 3, 2]  # B4,B3,B2
            else:  # 简化数据
                rgb_indices = [3, 2, 1]  # 前3个波段

            # 读取RGB波段
            rgb_data = src.read(rgb_indices)

            # 压缩数值范围
            compressed_data = (rgb_data / 10000.0 * 255).astype(np.uint8)

            # 调整通道顺序为HWC
            rgb_image = np.transpose(compressed_data, (1, 2, 0))

        # 显示结果
        plt.imshow(rgb_image)
        plt.title("Sentinel-2 RGB Image")
        plt.axis('off')
        plt.show()

        return rgb_image

    except rasterio.errors.RasterioIOError as e:
        print(f"文件读取错误: {e}")
    except IndexError as e:
        print(f"波段索引错误: {e}")
        print("提示：请检查数据是否包含所需波段，或调整band_indices参数")
    except Exception as e:
        print(f"处理失败: {e}")


# 使用示例
input_path = "D:/test1/2019_1101_nofire_B2348_B12_10m_roi.tif"  # 替换为实际路径
process_sentinel2_image(input_path)