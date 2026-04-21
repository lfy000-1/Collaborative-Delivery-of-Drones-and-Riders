# -*- coding: utf-8 -*- 
import geopandas as gpd
import numpy as np
from shapely import make_valid


def calculate_projected_area(gdf, target_crs="EPSG:4547"):
    """智能处理坐标系并计算空间属性"""
    # 自动检测原始坐标系
    original_crs = gdf.crs
    print(f"原始坐标系：{original_crs}")

    # 坐标系转换逻辑
    if original_crs is None:
        raise ValueError("SHP文件缺少坐标系定义，请人工确认坐标系后设置target_crs参数")

    if not original_crs.is_projected:
        print("检测到地理坐标系，正在转换为投影坐标系...")
        try:
            gdf = gdf.to_crs(target_crs)
            print(f"已转换为目标坐标系：{target_crs}")
        except Exception as e:
            raise RuntimeError(f"坐标系转换失败，请人工确认目标坐标系是否正确，错误详情：{str(e)}")

    # 几何修复（必须前置处理）
    gdf.geometry = gdf.geometry.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

    # 计算面积属性
    gdf['Area_m2'] = gdf.geometry.area.round(2)

    # 计算圆柱体参数
    gdf['Radius_m'] = np.sqrt(gdf['Area_m2'] / np.pi).round(2)  # 面积转半径

    # 计算投影坐标系下的中心点
    centroids = gdf.geometry.centroid
    gdf['Center_X'] = centroids.x.round(2)  # 投影坐标X（单位：米）
    gdf['Center_Y'] = centroids.y.round(2)  # 投影坐标Y

    # 计算亩数
    gdf['Area_mu'] = (gdf['Area_m2'] / 666.67).round(2)

    return gdf


def shp2csv(input_shp, output_csv, encoding='auto'):
    """主转换函数"""
    try:
        # 自动检测编码
        if encoding == 'auto':
            try:
                gdf = gpd.read_file(input_shp)
            except UnicodeDecodeError:
                gdf = gpd.read_file(input_shp, encoding='gbk')
                print("检测到GBK编码")
        else:
            gdf = gpd.read_file(input_shp, encoding=encoding)

        # 执行空间计算
        gdf = calculate_projected_area(gdf)

        # 保存CSV（确保包含所有新字段）
        columns_to_save = [
                              'geometry', 'Area_m2', 'Radius_m',
                              'Center_X', 'Center_Y', 'Area_mu'
                          ] + [col for col in gdf.columns if col not in [
            'geometry', 'Area_m2', 'Radius_m',
            'Center_X', 'Center_Y', 'Area_mu'
        ]]

        gdf[columns_to_save].to_csv(output_csv, index=False, encoding='utf-8-sig')

        # 生成统计报告
        total_area = gdf['Area_m2'].sum()
        avg_radius = gdf['Radius_m'].mean()
        print(f"\n转换成功！共处理 {len(gdf)} 个建筑物")
        print(f"总建筑面积：{total_area / 10000:.2f} 公顷")
        print(f"平均建筑半径：{avg_radius:.2f} 米")
        print(f"输出文件：{output_csv}")

    except Exception as e:
        print(f"\n错误发生：{str(e)}")
        print("建议排查步骤：")
        print("1. 确认SHP文件包含完整附属文件（.shp/.dbf/.shx/.prj）")
        print("2. 检查坐标系是否为中国常用投影（如EPSG:4547）")
        print("3. 尝试用GIS软件验证原始数据完整性")


if __name__ == "__main__":
    # 配置参数
    input_shp = "D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据\数据\裁剪后的建筑物数据.shp"  # 输入文件路径
    output_csv = "D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\代码和数据\裁剪后的建筑物数据.csv"  # 输出文件路径

    # 执行转换（遇到编码错误可尝试设置encoding='gbk'）
    shp2csv(input_shp, output_csv, encoding='auto')