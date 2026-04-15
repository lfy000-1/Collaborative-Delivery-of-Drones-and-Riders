import rasterio
import csv

def extract_tif_to_csv(tif_path, csv_path):
    with rasterio.open(tif_path) as src:
        data = src.read(1)                # 读取第一个波段
        transform = src.transform
        height, width = data.shape
        print(f"栅格行数: {height}, 列数: {width}")

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Row', 'Col', 'X', 'Y', 'Value'])   # 表头

            for row in range(height):
                for col in range(width):
                    # 计算像素中心坐标 (使用 +0.5 偏移)
                    x, y = transform * (col + 0.5, row + 0.5)
                    value = data[row, col]
                    writer.writerow([row+1, col+1, x, y, value])

    print(f"已输出 CSV 文件: {csv_path}")

if __name__ == "__main__":
    input_tif = r"D:\文档\大学文件\竞赛\统计建模\真正的统模\数据\初始数据\投影加剪裁后人口密度数据.tif"
    output_csv = r"D:\文档\大学文件\竞赛\统计建模\真正的统模\数据\人口密度数据.csv"
    extract_tif_to_csv(input_tif, output_csv)