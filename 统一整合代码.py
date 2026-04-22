"""
无人机运输系统统一分析程序
整合：地形3D渲染 + HDBSCAN聚类 + 无人机路径规划
基于DJI Matrice 300 RTK参数
"""

import warnings
import os
from pathlib import Path
from numba import jit, prange
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import hdbscan
from sklearn.preprocessing import StandardScaler
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = Path(r"D:\统模-第二次")
OUTPUT_DIR = Path(r"D:\统模-第二次\代码\统一输出")
OUTPUT_DIR.mkdir(exist_ok=True)

TERRAIN_PATH = DATA_DIR / "投影加剪裁后的地形高度数据.xlsx"
BUILDINGS_PATH = DATA_DIR / "裁剪后的建筑物数据_简洁版.xlsx"
POI_DATA_PATH = DATA_DIR / "代码" / "Collaborative-Delivery-of-Drones-and-Riders-main" / "Collaborative-Delivery-of-Drones-and-Riders-main" / "数据处理" / "重庆POI数据.csv"

MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 3
POP_DENSITY_WEIGHT = 2.0
CENTER_WEIGHT = 0.6
DENSITY_WEIGHT = 0.4

UAV_SPEED_MPS = 14.3
UAV_BATTERY_CAPACITY_WH = 588.0
UAV_HOVER_POWER_W = 218.8
UAV_CRUISE_POWER_W = 420.0
UAV_MAX_PAYLOAD_KG = 2.9
UAV_TAKEOFF_LANDING_POWER_W = 1000.0
UAV_SAFE_HEIGHT_M = 20.0

class Terrain:
    def __init__(self, elevation_matrix, x_range=None, y_range=None, x_coords=None, y_coords=None):
        self.elevation = np.asarray(elevation_matrix, dtype=np.float64)
        if self.elevation.ndim != 2:
            raise ValueError("elevation_matrix must be 2D")
        self.ny, self.nx = self.elevation.shape
        if x_coords is not None and y_coords is not None:
            self.x_coords = np.asarray(x_coords, dtype=np.float64)
            self.y_coords = np.asarray(y_coords, dtype=np.float64)
        elif x_range is not None and y_range is not None:
            self.x_coords = np.linspace(x_range[0], x_range[1], self.nx)
            self.y_coords = np.linspace(y_range[0], y_range[1], self.ny)
        else:
            raise ValueError("Provide x_coords/y_coords or x_range/y_range")
        self.x_min, self.x_max = self.x_coords[0], self.x_coords[-1]
        self.y_min, self.y_max = self.y_coords[0], self.y_coords[-1]

    def get_elevation(self, x, y):
        x = float(np.clip(x, self.x_min, self.x_max))
        y = float(np.clip(y, self.y_min, self.y_max))
        xi = float(np.interp(x, self.x_coords, np.arange(self.nx, dtype=float)))
        yi = float(np.interp(y, self.y_coords, np.arange(self.ny, dtype=float)))
        ix = int(np.floor(xi))
        iy = int(np.floor(yi))
        ix = min(max(ix, 0), self.nx - 2)
        iy = min(max(iy, 0), self.ny - 2)
        dx, dy = xi - ix, yi - iy
        h00, h10 = self.elevation[iy, ix], self.elevation[iy, ix + 1]
        h01, h11 = self.elevation[iy + 1, ix], self.elevation[iy + 1, ix + 1]
        return float(h00 * (1 - dx) * (1 - dy) + h10 * dx * (1 - dy) + h01 * (1 - dx) * dy + h11 * dx * dy)

class Obstacle:
    def __init__(self, x, y, z, r):
        self.x, self.y, self.z, self.r = x, y, z, r

@jit(nopython=True, cache=True, forceobj=False)
def calculate_3d_distance(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return np.sqrt(dx*dx + dy*dy + dz*dz)

@jit(nopython=True, cache=True, forceobj=False, parallel=True)
def calculate_route_cost_numba(points_x, points_y, points_z, terrain_arr, x_min, y_min, x_max, y_max, obstacles_x, obstacles_y, obstacles_r, obstacles_top, c_T=1000.0, c_B=100.0):
    n_points = len(points_x)
    if n_points < 2:
        return 0.0, 0.0, 0.0, 0.0

    total_3d_dist = 0.0
    total_straight_dist = 0.0
    total_cost = 0.0
    collision_count = 0

    for i in range(n_points - 1):
        p1x, p1y, p1z = points_x[i], points_y[i], points_z[i]
        p2x, p2y, p2z = points_x[i+1], points_y[i+1], points_z[i+1]

        seg_3d = calculate_3d_distance(p1x, p1y, p1z, p2x, p2y, p2z)
        seg_2d = np.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)
        total_3d_dist += seg_3d
        total_straight_dist += seg_2d

        if seg_2d > 1e-6:
            cos_theta = (p2x-p1x) * 0 + (p2y-p1y) * 0 + (p2z-p1z) * 1
            cos_theta /= (np.sqrt((p2x-p1x)**2 + (p2y-p1y)**2 + (p2z-p1z)**2) + 1e-9)
            angle_penalty = max(0.0, 1.0 - cos_theta) * 20.0
            total_cost += angle_penalty

        for j in range(len(obstacles_x)):
            dx = p1x - obstacles_x[j]
            dy = p1y - obstacles_y[j]
            dist_xy = np.sqrt(dx*dx + dy*dy)
            if dist_xy < obstacles_r[j]:
                if p1z < obstacles_top[j]:
                    total_cost += c_B
                    collision_count += 1

    return total_3d_dist, total_straight_dist, total_cost, float(collision_count)

def load_terrain_and_buildings():
    print("正在读取地形数据...")
    terrain_df = pd.read_excel(TERRAIN_PATH)
    X = terrain_df.iloc[:, 2].values.astype(np.float64)
    Y = terrain_df.iloc[:, 3].values.astype(np.float64)
    Z = terrain_df.iloc[:, 4].values.astype(np.float64)

    grid_res = 200
    grid_x, grid_y = np.mgrid[X.min():X.max():complex(0, grid_res), Y.min():Y.max():complex(0, grid_res)]
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='linear', fill_value=np.nanmin(Z))

    terrain = Terrain(grid_z.T, x_range=(X.min(), X.max()), y_range=(Y.min(), Y.max()))

    print("正在读取建筑物数据...")
    buildings_df = pd.read_excel(BUILDINGS_PATH)
    radii = buildings_df['Radius_m'].values.astype(np.float64)
    centers_x = buildings_df['Center_X'].values.astype(np.float64)
    centers_y = buildings_df['Center_Y'].values.astype(np.float64)
    heights = buildings_df['Elevation'].values.astype(np.float64)

    obstacles = []
    for i in range(len(centers_x)):
        if radii[i] >= 1 and heights[i] > 0:
            terrain_z = terrain.get_elevation(centers_x[i], centers_y[i])
            obs = Obstacle(centers_x[i], centers_y[i], float(heights[i]), float(radii[i]))
            obstacles.append(obs)

    print(f"加载完成: 地形数据点 {len(X)}, 建筑物 {len(obstacles)}")
    return terrain, X, Y, Z, grid_x, grid_y, grid_z, buildings_df, obstacles

def perform_clustering(terrain):
    print("\n正在进行HDBSCAN聚类...")
    x_min, x_max = terrain.x_min, terrain.x_max
    y_min, y_max = terrain.y_min, terrain.y_max
    print(f"地图范围: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")

    original_cluster_path = Path(r"D:\统模-第二次\代码\Collaborative-Delivery-of-Drones-and-Riders-main\Collaborative-Delivery-of-Drones-and-Riders-main\数据处理\候选点_适中.csv")

    if original_cluster_path.exists():
        print(f"找到原始聚类文件，加载并等比例映射...")
        candidate_df = pd.read_csv(original_cluster_path)

        if 'X' in candidate_df.columns and 'Y' in candidate_df.columns:
            orig_x_col, orig_y_col = 'X', 'Y'
        elif '投影X' in candidate_df.columns and '投影Y' in candidate_df.columns:
            orig_x_col, orig_y_col = '投影X', '投影Y'
        else:
            raise ValueError(f"原始聚类文件缺少坐标列，可用列: {candidate_df.columns.tolist()}")

        orig_x_min, orig_x_max = candidate_df[orig_x_col].min(), candidate_df[orig_x_col].max()
        orig_y_min, orig_y_max = candidate_df[orig_y_col].min(), candidate_df[orig_y_col].max()
        print(f"原始聚类坐标范围: X=[{orig_x_min:.0f}, {orig_x_max:.0f}], Y=[{orig_y_min:.0f}, {orig_y_max:.0f}]")

        map_width = x_max - x_min
        map_height = y_max - y_min

        scaled_points = []
        for _, row in candidate_df.iterrows():
            orig_x = row[orig_x_col]
            orig_y = row[orig_y_col]

            scaled_x = x_min + (orig_x - orig_x_min) / (orig_x_max - orig_x_min + 1e-9) * map_width
            scaled_y = y_min + (orig_y - orig_y_min) / (orig_y_max - orig_y_min + 1e-9) * map_height
            scaled_z = terrain.get_elevation(scaled_x, scaled_y)

            scaled_points.append({
                "cluster_id": row.get('cluster_id', len(scaled_points)),
                "X": scaled_x,
                "Y": scaled_y,
                "Z": scaled_z,
                "点数量": row.get('点数量', 1)
            })

        result_df = pd.DataFrame(scaled_points)
        point_names = [chr(ord("A") + i) for i in range(len(result_df))]
        result_df["起降点名称"] = point_names
        print(f"聚类完成: 发现 {len(result_df)} 个起降点")
        return result_df

def find_obstacle_free_path(start_x, start_y, start_z, end_x, end_y, end_z, terrain, obstacles, obs_x, obs_y, obs_r, obs_top, kdtree=None):
    """使用KD树加速的避障路径规划"""
    path = [(start_x, start_y, start_z)]

    dx = end_x - start_x
    dy = end_y - start_y
    dz = end_z - start_z
    total_dist = np.sqrt(dx*dx + dy*dy + dz*dz)

    if total_dist < 1:
        return path + [(end_x, end_y, end_z)]

    n_check_points = max(10, int(total_dist / 20))
    bypass_threshold = total_dist * 0.15
    min_bypass_dist = 15.0

    for i in range(1, n_check_points):
        t = i / n_check_points
        px = start_x + dx * t
        py = start_y + dy * t
        pz = start_z + dz * t

        terrain_z = terrain.get_elevation(px, py)
        pz = max(pz, terrain_z + UAV_SAFE_HEIGHT_M)

        if kdtree is not None:
            nearby_indices = kdtree.query_ball_point([px, py], r=bypass_threshold)
        else:
            nearby_indices = list(range(len(obs_x)))

        collision = False
        for k in nearby_indices:
            dist_xy = np.sqrt((px - obs_x[k])**2 + (py - obs_y[k])**2)
            if dist_xy < obs_r[k] * 1.2 and pz < obs_top[k]:
                collision = True
                break

        if collision:
            perp_x, perp_y = -dy / (total_dist + 1e-9), dx / (total_dist + 1e-9)

            bypass_dist = max(min_bypass_dist, obs_r[k] * 2.0)

            left_x, left_y = px + perp_x * bypass_dist, py + perp_y * bypass_dist
            right_x, right_y = px - perp_x * bypass_dist, py - perp_y * bypass_dist

            left_clear = True
            if kdtree is not None:
                left_nearby = kdtree.query_ball_point([left_x, left_y], r=bypass_threshold)
                for k2 in left_nearby:
                    if np.sqrt((left_x - obs_x[k2])**2 + (left_y - obs_y[k2])**2) < obs_r[k2] * 1.3:
                        left_clear = False
                        break
            else:
                for k2 in range(len(obs_x)):
                    if np.sqrt((left_x - obs_x[k2])**2 + (left_y - obs_y[k2])**2) < obs_r[k2] * 1.3:
                        left_clear = False
                        break

            right_clear = True
            if kdtree is not None:
                right_nearby = kdtree.query_ball_point([right_x, right_y], r=bypass_threshold)
                for k2 in right_nearby:
                    if np.sqrt((right_x - obs_x[k2])**2 + (right_y - obs_y[k2])**2) < obs_r[k2] * 1.3:
                        right_clear = False
                        break
            else:
                for k2 in range(len(obs_x)):
                    if np.sqrt((right_x - obs_x[k2])**2 + (right_y - obs_y[k2])**2) < obs_r[k2] * 1.3:
                        right_clear = False
                        break

            if left_clear and not right_clear:
                bypass_x, bypass_y = left_x, left_y
            elif right_clear and not left_clear:
                bypass_x, bypass_y = right_x, right_y
            elif left_clear and right_clear:
                if np.sqrt((left_x - end_x)**2 + (left_y - end_y)**2) < np.sqrt((right_x - end_x)**2 + (right_y - end_y)**2):
                    bypass_x, bypass_y = left_x, left_y
                else:
                    bypass_x, bypass_y = right_x, right_y
            else:
                bypass_x, bypass_y = px + perp_x * bypass_dist * 1.8, py + perp_y * bypass_dist * 1.8

            bypass_z = terrain.get_elevation(bypass_x, bypass_y) + UAV_SAFE_HEIGHT_M * 1.3
            path.append((bypass_x, bypass_y, bypass_z))

    path.append((end_x, end_y, end_z))

    final_path = [path[0]]
    for pt in path[1:]:
        if len(final_path) > 0:
            last_pt = final_path[-1]
            dist = np.sqrt((pt[0] - last_pt[0])**2 + (pt[1] - last_pt[1])**2 + (pt[2] - last_pt[2])**2)
            if dist > 5:
                final_path.append(pt)
        else:
            final_path.append(pt)

    return final_path

@jit(nopython=True, cache=True, forceobj=False)
def calculate_path_cost_with_penalty(path_x, path_y, path_z, obs_x, obs_y, obs_r, obs_top):
    """
    计算路径总代价（考虑高度、转向、障碍物距离惩罚）

    惩罚项:
    1. 高度惩罚: 超过60m开始惩罚,120m严格限制
    2. 转向惩罚: 转角越大惩罚越高
    3. 障碍物距离惩罚: 距离障碍物越近惩罚越高
    """
    n = len(path_x)
    if n < 2:
        return 0.0, 0, 0.0, 0.0, 0.0

    total_distance = 0.0
    total_height_penalty = 0.0
    total_turn_penalty = 0.0
    total_obstacle_penalty = 0.0
    n_violations = 0

    HEIGHT_SOFT_LIMIT = 60.0
    HEIGHT_HARD_LIMIT = 120.0
    HEIGHT_PENALTY_SCALE = 50.0
    TURN_PENALTY_SCALE = 30.0
    OBSTACLE_SAFE_DIST = 10.0
    OBSTACLE_PENALTY_SCALE = 100.0

    prev_dx, prev_dy, prev_dz = 0.0, 0.0, 0.0

    for i in range(n - 1):
        dx = path_x[i+1] - path_x[i]
        dy = path_y[i+1] - path_y[i]
        dz = path_z[i+1] - path_z[i]
        seg_dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        total_distance += seg_dist

        avg_z = (path_z[i] + path_z[i+1]) / 2.0
        if avg_z > HEIGHT_SOFT_LIMIT:
            height_excess = avg_z - HEIGHT_SOFT_LIMIT
            if avg_z > HEIGHT_HARD_LIMIT:
                height_excess = HEIGHT_HARD_LIMIT - HEIGHT_SOFT_LIMIT
            total_height_penalty += height_excess * HEIGHT_PENALTY_SCALE / seg_dist

        if i > 0:
            dot = dx * prev_dx + dy * prev_dy + dz * prev_dz
            norm_curr = seg_dist
            norm_prev = np.sqrt(prev_dx*prev_dx + prev_dy*prev_dy + prev_dz*prev_dz)
            if norm_curr > 1e-9 and norm_prev > 1e-9:
                cos_angle = dot / (norm_curr * norm_prev)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                turn_angle = np.arccos(cos_angle)
                total_turn_penalty += turn_angle * TURN_PENALTY_SCALE / seg_dist

        px = (path_x[i] + path_x[i+1]) / 2.0
        py = (path_y[i] + path_y[i+1]) / 2.0
        pz = (path_z[i] + path_z[i+1]) / 2.0

        for j in range(len(obs_x)):
            dist_xy = np.sqrt((px - obs_x[j])**2 + (py - obs_y[j])**2)
            if dist_xy < obs_r[j] + OBSTACLE_SAFE_DIST:
                clearance = dist_xy - obs_r[j]
                if clearance < 0:
                    total_obstacle_penalty += OBSTACLE_PENALTY_SCALE * 10
                    n_violations += 1
                elif clearance < OBSTACLE_SAFE_DIST:
                    total_obstacle_penalty += OBSTACLE_PENALTY_SCALE * (OBSTACLE_SAFE_DIST - clearance) / OBSTACLE_SAFE_DIST / seg_dist

        prev_dx, prev_dy, prev_dz = dx, dy, dz

    total_penalty = total_height_penalty + total_turn_penalty + total_obstacle_penalty
    return total_distance, n_violations, total_penalty, total_height_penalty, total_turn_penalty

def calculate_transport_routes(terrain, obstacles, landing_points):
    print("\n正在计算所有运输路线（含避障路径规划）...")
    n_points = len(landing_points)
    results = []
    all_paths = {}

    terrain_arr = terrain.elevation
    x_min, y_min = terrain.x_min, terrain.y_min
    x_max, y_max = terrain.x_max, terrain.y_max

    obs_x = np.array([o.x for o in obstacles], dtype=np.float64)
    obs_y = np.array([o.y for o in obstacles], dtype=np.float64)
    obs_r = np.array([o.r for o in obstacles], dtype=np.float64)
    obs_top = np.array([terrain.get_elevation(o.x, o.y) + o.z for o in obstacles], dtype=np.float64)

    route_count = n_points * (n_points - 1)
    print(f"共 {n_points} 个起降点，预计算 {route_count} 条避障路径...")

    print("正在构建KD树加速障碍物查询...")
    obstacle_coords = np.column_stack((obs_x, obs_y))
    kdtree = cKDTree(obstacle_coords)

    start_time = time.time()
    path_compute_time = 0

    for i, row_i in landing_points.iterrows():
        for j, row_j in landing_points.iterrows():
            if i == j:
                continue

            p1x, p1y, p1z = row_i['X'], row_i['Y'], row_i['Z'] + UAV_SAFE_HEIGHT_M
            p2x, p2y, p2z = row_j['X'], row_j['Y'], row_j['Z'] + UAV_SAFE_HEIGHT_M

            t0 = time.time()
            path = find_obstacle_free_path(p1x, p1y, p1z, p2x, p2y, p2z, terrain, obstacles, obs_x, obs_y, obs_r, obs_top, kdtree)
            path_compute_time += time.time() - t0

            path_x = np.array([p[0] for p in path], dtype=np.float64)
            path_y = np.array([p[1] for p in path], dtype=np.float64)
            path_z = np.array([p[2] for p in path], dtype=np.float64)

            seg_3d_dist, n_collisions, total_penalty, height_penalty, turn_penalty = calculate_path_cost_with_penalty(
                path_x, path_y, path_z, obs_x, obs_y, obs_r, obs_top)

            seg_2d = np.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)

            flight_time_h = seg_3d_dist / UAV_SPEED_MPS / 3600.0
            flight_time_min = seg_3d_dist / UAV_SPEED_MPS / 60.0
            energy_wh = UAV_CRUISE_POWER_W * flight_time_h
            battery_pct = (energy_wh / UAV_BATTERY_CAPACITY_WH) * 100.0

            n_waypoints = len(path)
            n_avoidance = max(0, n_waypoints - 2)

            results.append({
                '起点': row_i['起降点名称'],
                '终点': row_j['起降点名称'],
                '直线距离_m': round(seg_2d, 1),
                '实际距离_m': round(seg_3d_dist, 1),
                '飞行时间_分钟': round(flight_time_min, 2),
                '耗电量_百分比': round(min(battery_pct, 999.99), 2),
                '避障次数': n_avoidance,
                '路径代价': round(total_penalty, 2),
                '高度惩罚': round(height_penalty, 2),
                '转向惩罚': round(turn_penalty, 2)
            })

            all_paths[(row_i['起降点名称'], row_j['起降点名称'])] = path

    elapsed = time.time() - start_time
    print(f"路径规划计算完成: {len(results)} 条路线")
    print(f"  - 路径计算耗时: {path_compute_time:.2f} 秒")
    print(f"  - 总耗时: {elapsed:.2f} 秒")

    return pd.DataFrame(results), all_paths

def create_3d_visualization(terrain, X, Y, Z, grid_x, grid_y, grid_z, buildings_df, landing_points, route_results, all_paths=None):
    print("\n正在生成3D可视化...")
    plotter = pv.Plotter(off_screen=False)

    print("正在创建3D地形...")
    terrain_mesh = pv.StructuredGrid(grid_x, grid_y, grid_z)
    z_min, z_max = np.nanmin(grid_z), np.nanmax(grid_z)
    z_norm = (grid_z - z_min) / (z_max - z_min + 1e-9)

    terrain_colors = np.zeros((grid_z.shape[0], grid_z.shape[1], 3), dtype=np.float32)
    for i in range(grid_z.shape[0]):
        for j in range(grid_z.shape[1]):
            z = z_norm[i, j]
            if z < 0.2:
                terrain_colors[i, j] = [0.25, 0.55, 0.28]
            elif z < 0.4:
                terrain_colors[i, j] = [0.35, 0.68, 0.35]
            elif z < 0.6:
                terrain_colors[i, j] = [0.50, 0.65, 0.30]
            elif z < 0.8:
                terrain_colors[i, j] = [0.65, 0.58, 0.25]
            else:
                terrain_colors[i, j] = [0.80, 0.75, 0.55]

    terrain_mesh['terrain_colors'] = terrain_colors.reshape(-1, 3)
    plotter.add_mesh(terrain_mesh, rgb=True, scalars='terrain_colors', opacity=0.95, smooth_shading=True)

    print("正在创建建筑物模型...")
    radii = buildings_df['Radius_m'].values
    centers_x = buildings_df['Center_X'].values
    centers_y = buildings_df['Center_Y'].values
    heights = buildings_df['Elevation'].values

    height_bins = [0, 30, 60, 100, 150]
    building_colors_by_height = [
        [0.20, 0.45, 0.20],
        [0.30, 0.65, 0.35],
        [0.50, 0.75, 0.40],
        [0.75, 0.60, 0.20],
        [0.85, 0.35, 0.15],
        [0.90, 0.15, 0.10]
    ]

    buildings_blocks = []
    for cx, cy, r, h in zip(centers_x, centers_y, radii, heights):
        if r < 1 or h <= 0:
            continue
        base_z = terrain.get_elevation(cx, cy)
        center_z = base_z + h / 2

        h_idx = 0
        for idx, threshold in enumerate(height_bins[1:], 0):
            if h < threshold:
                h_idx = idx
                break
        else:
            h_idx = len(building_colors_by_height) - 1

        building = pv.Cylinder(
            center=(cx, cy, center_z),
            direction=(0, 0, 1),
            radius=r,
            height=h,
            resolution=8
        )
        buildings_blocks.append((building, building_colors_by_height[h_idx]))

    if buildings_blocks:
        print(f"正在批量添加 {len(buildings_blocks)} 个建筑物...")
        batch_size = 2000
        for batch_start in range(0, len(buildings_blocks), batch_size):
            batch_end = min(batch_start + batch_size, len(buildings_blocks))
            batch = buildings_blocks[batch_start:batch_end]
            meshes = [b[0] for b in batch]
            colors = [b[1] for b in batch]
            multi_block = pv.MultiBlock(meshes)
            plotter.add_mesh(multi_block, color='gray', opacity=0.90, smooth_shading=True)

    print("正在添加起降点...")
    for _, point in landing_points.iterrows():
        px, py, pz = point['X'], point['Y'], point['Z']
        marker = pv.Sphere(radius=12, center=(px, py, pz + 8))
        plotter.add_mesh(marker, color='red', opacity=1.0)
        plotter.add_point_labels([(px, py, pz + 25)], [point['起降点名称']], font_size=14, bold=True)

    print("正在添加避障运输路线...")
    if all_paths is not None and len(all_paths) > 0:
        route_idx = 0
        for (start_name, end_name), path in all_paths.items():
            if len(path) < 2:
                continue

            path_x = np.array([p[0] for p in path])
            path_y = np.array([p[1] for p in path])
            path_z = np.array([p[2] for p in path])

            points = np.column_stack((path_x, path_y, path_z))
            route_line = pv.lines_from_points(points)

            hue = (route_idx % 20) / 20.0
            r, g, b = hue, 0.5, 1.0 - hue * 0.5
            plotter.add_mesh(route_line, color=(r, g, b), line_width=3, opacity=0.8)
            route_idx += 1

        print(f"已添加 {len(all_paths)} 条避障路线")

    plotter.set_background("#87CEEB")
    plotter.camera_position = [
        (X.min() - 200, Y.min() - 200, Z.max() + 300),
        (X.mean(), Y.mean(), Z.mean()),
        (0, 0, 1)
    ]

    print("\n" + "="*80)
    print("3D可视化窗口已打开，关闭后程序结束")
    print("操作说明:")
    print("  - 左键拖动: 旋转视角")
    print("  - 右键拖动或滚轮: 缩放")
    print("  - 中键拖动: 平移")
    print("="*80)

    plotter.show()

def save_results(landing_points, route_results):
    landing_csv = OUTPUT_DIR / "起降点列表.csv"
    route_csv = OUTPUT_DIR / "运输路线结果.csv"
    landing_points.to_csv(landing_csv, index=False, encoding='utf-8-sig')
    route_results.to_csv(route_csv, index=False, encoding='utf-8-sig')

    txt_path = OUTPUT_DIR / "运输路线完整报告.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("无人机运输系统 - 完整运输路线报告\n".center(90) + "\n")
        f.write("=" * 100 + "\n\n")

        f.write("【基本信息】\n")
        f.write(f"起降点数量: {len(landing_points)}\n")
        f.write(f"起降点列表: {', '.join(landing_points['起降点名称'].tolist())}\n")
        f.write(f"路线总数: {len(route_results)}\n\n")

        f.write("【起降点坐标信息】\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'名称':<8} {'X坐标':<18} {'Y坐标':<18} {'海拔(m)':<12} {'聚类点数':<10}\n")
        f.write("-" * 80 + "\n")
        for _, row in landing_points.iterrows():
            f.write(f"{row['起降点名称']:<8} {row['X']:<18.2f} {row['Y']:<18.2f} {row['Z']:<12.2f} {row['点数量']:<10}\n")
        f.write("\n")

        f.write("【运输路线详细表】\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'起点':<8} {'终点':<8} {'直线距离':<12} {'实际距离':<12} {'飞行时间':<12} {'耗电量':<12} {'代价':<10} {'高度惩罚':<10} {'转向惩罚':<10}\n")
        f.write(f"{'':.<8} {'':.<8} {'(m)':.<12} {'(m)':.<12} {'(分钟)':.<12} {'(%)':.<12} {'':.<10} {'':.<10} {'':.<10}\n")
        f.write("-" * 100 + "\n")

        for _, route in route_results.iterrows():
            f.write(f"{route['起点']:<8} {route['终点']:<8} {route['直线距离_m']:<12.1f} {route['实际距离_m']:<12.1f} {route['飞行时间_分钟']:<12.2f} {route['耗电量_百分比']:<12.2f} {route.get('路径代价', 0):<10.2f} {route.get('高度惩罚', 0):<10.2f} {route.get('转向惩罚', 0):<10.2f}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("报告生成完毕\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n结果已保存至: {OUTPUT_DIR}")
    print(f"  - CSV: {landing_csv.name}, {route_csv.name}")
    print(f"  - TXT: {txt_path.name}")

def print_route_summary(route_results, landing_points):
    print("\n" + "="*100)
    print("无人机运输路线汇总表".center(90))
    print("="*100)
    print(f"\n起降点数量: {len(landing_points)}")
    print(f"起降点列表: {', '.join(landing_points['起降点名称'].tolist())}")
    print(f"路线总数: {len(route_results)}")

    print("\n" + "-"*100)
    print(f"{'起点':^8} {'终点':^8} {'直线距离':^12} {'实际距离':^12} {'飞行时间':^12} {'耗电量(%)':^12}")
    print("-"*100)

    for _, row in route_results.iterrows():
        print(f"{row['起点']:^8} {row['终点']:^8} {row['直线距离_m']:>10.1f}m {row['实际距离_m']:>10.1f}m {row['飞行时间_分钟']:>10.2f}分钟 {row['耗电量_百分比']:>10.2f}%")

    print("\n" + "="*100)
    print("各起降点信息".center(90))
    print("="*100)
    print(f"\n{'名称':^8} {'X坐标':^15} {'Y坐标':^15} {'海拔(m)':^12} {'聚类点数':^10}")
    print("-"*60)
    for _, row in landing_points.iterrows():
        print(f"{row['起降点名称']:^8} {row['X']:>14.2f} {row['Y']:>14.2f} {row['Z']:>10.2f} {row['点数量']:>8}")

def main():
    print("="*80)
    print("无人机运输系统统一分析程序".center(70))
    print("基于DJI Matrice 300 RTK参数".center(70))
    print("="*80)

    terrain, X, Y, Z, grid_x, grid_y, grid_z, buildings_df, obstacles = load_terrain_and_buildings()
    landing_points = perform_clustering(terrain)
    route_results, all_paths = calculate_transport_routes(terrain, obstacles, landing_points)
    save_results(landing_points, route_results)
    print_route_summary(route_results, landing_points)
    create_3d_visualization(terrain, X, Y, Z, grid_x, grid_y, grid_z, buildings_df, landing_points, route_results, all_paths)

if __name__ == "__main__":
    main()
