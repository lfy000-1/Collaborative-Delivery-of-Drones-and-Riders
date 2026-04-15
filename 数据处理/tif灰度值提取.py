import rasterio
import csv
import numpy as np


def extract_single_band_values(tif_path, output_csv):
    """
    提取单波段TIFF的像元值（如灰度值或高程）并提取地理坐标，将NaN替换为0
    参数:
        tif_path: 输入TIFF文件路径
        output_csv: 输出CSV文件路径
    """
    with rasterio.open(tif_path) as src:
        # 读取第一个波段数据
        band = src.read(1)
        transform = src.transform
        nodata = src.nodata

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Row', 'Col', 'X', 'Y', 'T_elevation'])

            for row in range(src.height):
                for col in range(src.width):
                    # 灰度值即是高度
                    gray_value = band[row, col]

                    # 处理nodata值
                    if nodata is not None:
                        if np.isnan(nodata):
                            # 当nodata值为NaN时的特殊处理
                            if np.isnan(gray_value):
                                gray_value = 0.0
                        else:
                            # 正常数值比较
                            if gray_value == nodata:
                                gray_value = 0.0

                    # 处理其他NaN情况
                    if np.isnan(gray_value):
                        gray_value = 0.0

                    # 获取地理坐标
                    x, y = rasterio.transform.xy(transform, row, col)

                    writer.writerow([
                        row + 1,  # 行号从1开始（根据GIS惯例）
                        col + 1,  # 列号从1开始
                        round(x, 10),  # 保留6位小数
                        round(y, 10),
                        round(gray_value, 2)  # 值保留两位小数
                    ])
        print(f"处理完成！结果保存至 {output_csv}")


# 提取巴南区片区地形数据的灰度值
# if __name__ == "__main__":
#    input_file = "巴南区片区地形数据.tif"
#    output_file = "巴南区片区地形高度数据.csv"
#    extract_single_band_values(input_file, output_file)


# 提取巴南区片区地形数据的灰度值
if __name__ == "__main__":
    input_file = "D:\文档\大学文件\竞赛\统计建模\真正的统模\数据\初始数据\投影加剪裁后的地形数据.tif"
    output_file = "D:\文档\大学文件\竞赛\统计建模\真正的统模\数据\初始数据\投影加剪裁后的地形高度数据.csv"
    extract_single_band_values(input_file, output_file)


# 提取巴南区片区建筑物数据的灰度值
# if __name__ == "__main__":
#    input_file = "巴南区片区建筑物数据.tif"
#    output_file = "巴南区片区建筑物高度数据.csv"
#    extract_single_band_values(input_file, output_file)


