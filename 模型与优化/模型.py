"""无人机路径代价模型。

该模块包含：
1. 地形插值模型（Terrain）
2. 建筑物障碍物模型（Obstacle）
3. 路径总代价计算器（UAVPathCostCalculator）
4. 从 CSV 自动加载地形与建筑数据的工具函数
"""

import csv
from dataclasses import dataclass
from math import atan2, pi, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Obstacle:
    """圆柱体障碍物模型（建筑物简化）。

    属性:
        x: 障碍物中心点 X 坐标（投影坐标，单位通常为米）
        y: 障碍物中心点 Y 坐标（投影坐标，单位通常为米）
        z: 建筑物相对地面高度（米）
        r: 建筑物等效半径（米）
    """

    x: float
    y: float
    z: float
    r: float


class Terrain:
    """地形高程模型，支持规则网格上的双线性插值。"""

    def __init__(
        self,
        elevation_matrix: np.ndarray,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        x_coords: Optional[Sequence[float]] = None,
        y_coords: Optional[Sequence[float]] = None,
    ):
        """初始化地形对象。

        参数:
            elevation_matrix: 地形高程矩阵，形状为 (ny, nx)
            x_range: 当不提供 x_coords 时使用，表示 X 方向坐标范围 (min_x, max_x)
            y_range: 当不提供 y_coords 时使用，表示 Y 方向坐标范围 (min_y, max_y)
            x_coords: 每一列对应的实际 X 坐标数组（长度需等于 nx）
            y_coords: 每一行对应的实际 Y 坐标数组（长度需等于 ny）

        说明:
            - 优先使用 x_coords/y_coords
            - 若使用 x_range/y_range，则按线性均匀网格自动生成坐标。
        """

        self.elevation = np.asarray(elevation_matrix, dtype=float)
        if self.elevation.ndim != 2:
            raise ValueError("elevation_matrix must be 2D")

        self.ny, self.nx = self.elevation.shape
        if self.nx < 2 or self.ny < 2:
            raise ValueError("elevation_matrix must be at least 2x2 for interpolation")

        if x_coords is not None and y_coords is not None:
            x_arr = np.asarray(x_coords, dtype=float)
            y_arr = np.asarray(y_coords, dtype=float)
            if x_arr.size != self.nx:
                raise ValueError(f"x_coords length {x_arr.size} != nx {self.nx}")
            if y_arr.size != self.ny:
                raise ValueError(f"y_coords length {y_arr.size} != ny {self.ny}")
            if np.any(np.diff(x_arr) <= 0):
                raise ValueError("x_coords must be strictly increasing")
            if np.any(np.diff(y_arr) <= 0):
                raise ValueError("y_coords must be strictly increasing")
            self.x_coords = x_arr
            self.y_coords = y_arr
        elif x_range is not None and y_range is not None:
            x_min, x_max = x_range
            y_min, y_max = y_range
            if x_max <= x_min or y_max <= y_min:
                raise ValueError("x_range/y_range must be (min, max) with max > min")
            self.x_coords = np.linspace(float(x_min), float(x_max), self.nx)
            self.y_coords = np.linspace(float(y_min), float(y_max), self.ny)
        else:
            raise ValueError("Provide either (x_coords, y_coords) or (x_range, y_range)")

        self.x_min = float(self.x_coords[0])
        self.x_max = float(self.x_coords[-1])
        self.y_min = float(self.y_coords[0])
        self.y_max = float(self.y_coords[-1])

    def get_elevation(self, x: float, y: float) -> float:
        """查询任意坐标点 (x, y) 的地形高程。

        参数:
            x: 查询点 X 坐标
            y: 查询点 Y 坐标

        返回:
            双线性插值后的地形高程值。

        说明:
            - 超出边界会先裁剪到地形范围内。
            - 使用网格坐标与实际坐标映射后进行双线性插值。
        """

        x = float(np.clip(x, self.x_min, self.x_max))
        y = float(np.clip(y, self.y_min, self.y_max))

        xi = float(np.interp(x, self.x_coords, np.arange(self.nx, dtype=float)))
        yi = float(np.interp(y, self.y_coords, np.arange(self.ny, dtype=float)))

        ix = int(np.floor(xi))
        iy = int(np.floor(yi))

        ix = min(max(ix, 0), self.nx - 2)
        iy = min(max(iy, 0), self.ny - 2)

        dx = xi - ix
        dy = yi - iy

        h00 = self.elevation[iy, ix]
        h10 = self.elevation[iy, ix + 1]
        h01 = self.elevation[iy + 1, ix]
        h11 = self.elevation[iy + 1, ix + 1]

        h0 = h00 * (1.0 - dx) + h10 * dx
        h1 = h01 * (1.0 - dx) + h11 * dx
        return float(h0 * (1.0 - dy) + h1 * dy)


class UAVPathCostCalculator:
    """无人机路径代价计算器。"""

    def __init__(
        self,
        terrain: Terrain,
        obstacles: List[Obstacle],
        c_T: float = 1000.0,
        c_B: float = 100.0,
        c_hr: float = 20.0,
        c_theta: float = 20.0 / pi,
        k_nearest: int = 4,
        ignore_first_collision: bool = True,
        ignore_last_collision: bool = True,
    ):
        """初始化路径代价计算器。

        参数:
            terrain: 地形对象
            obstacles: 障碍物列表
            c_T: 地形约束惩罚系数
            c_B: 障碍物碰撞惩罚系数
            c_hr: 飞行高度奖励系数（高于建筑顶部时给负代价奖励）
            c_theta: 转角约束惩罚系数
            k_nearest: 每个路径点仅评估最近的 k 个障碍物，用于提速
            ignore_first_collision: 是否忽略全路径的第一次碰撞惩罚
            ignore_last_collision: 是否忽略全路径的最后一次碰撞惩罚
        """

        self.terrain = terrain
        self.obstacles = obstacles
        self.c_T = c_T
        self.c_B = c_B
        self.c_hr = c_hr
        self.c_theta = c_theta
        self.k_nearest = max(1, int(k_nearest))
        self.ignore_first_collision = bool(ignore_first_collision)
        self.ignore_last_collision = bool(ignore_last_collision)

        if obstacles:
            self._obs_x = np.array([obs.x for obs in obstacles], dtype=float)
            self._obs_y = np.array([obs.y for obs in obstacles], dtype=float)
            self._obs_r = np.array([obs.r for obs in obstacles], dtype=float)
            self._obs_top = np.array(
                [self.terrain.get_elevation(obs.x, obs.y) + obs.z for obs in obstacles],
                dtype=float,
            )
        else:
            self._obs_x = np.empty(0, dtype=float)
            self._obs_y = np.empty(0, dtype=float)
            self._obs_r = np.empty(0, dtype=float)
            self._obs_top = np.empty(0, dtype=float)

    def _nearest_obstacle_indices(self, point: Tuple[float, float]) -> np.ndarray:
        """返回距离给定点最近的障碍物索引集合。

        参数:
            point: 当前参考点 (x, y)

        返回:
            按距离从近到远排序的障碍物索引数组，长度不超过 k_nearest。
        """

        if self._obs_x.size == 0:
            return np.empty(0, dtype=int)

        px, py = point
        dist2 = (self._obs_x - px) ** 2 + (self._obs_y - py) ** 2
        k = min(self.k_nearest, dist2.size)
        nearest = np.argpartition(dist2, k - 1)[:k]
        nearest = nearest[np.argsort(dist2[nearest])]
        return nearest

    @staticmethod
    def _segment_intersect_cylinder(
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        cx: float,
        cy: float,
        r: float,
    ) -> bool:
        """判断线段与圆柱体水平投影是否相交。

        参数:
            p1: 线段起点 (x, y, H)
            p2: 线段终点 (x, y, H)
            cx: 圆心 X 坐标
            cy: 圆心 Y 坐标
            r: 半径

        返回:
            若线段与圆（障碍物水平投影）相交或接触，返回 True；否则 False。
        """

        x1, y1, _ = p1
        x2, y2, _ = p2

        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy

        a = dx * dx + dy * dy
        if a < 1e-12:
            return (fx * fx + fy * fy) <= r * r

        t = -(fx * dx + fy * dy) / a
        t = max(0.0, min(1.0, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        dist2 = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
        return dist2 <= r * r

    def terrain_cost(self, path: List[Tuple[float, float, float]]) -> float:
        """计算地形约束代价。

        规则:
            - 若 H > 120，超高部分按 c_T 惩罚。
            - 若 DH(x,y) < H <= 120，代价为 0。
            - 若 H <= DH(x,y)，低于地形部分按 c_T 惩罚。

        参数:
            path: 路径点列表，每个点为 (x, y, H)

        返回:
            地形约束总成本。
        """

        total = 0.0
        for x, y, H in path:
            dh = self.terrain.get_elevation(x, y)
            if H > 120.0:
                cost = (H - 120.0) * self.c_T
            elif dh < H <= 120.0:
                cost = 0.0
            else:
                cost = (dh - H) * self.c_T
            total += cost
        return float(total)

    def obstacle_collision_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算障碍物相关代价（碰撞惩罚 + 高度奖励）。

        参数:
            path: 中间路径点列表 (x, y, H)
            start: 起点 (x, y, H)
            end: 终点 (x, y, H)

        返回:
            障碍物总代价。

        说明:
            - 对每个航段，仅检查最近 k 个障碍物。
            - 若航段与障碍物相交，则叠加碰撞惩罚 c_B。
            - 若当前点高度高于建筑顶部，可获得 -c_hr 奖励。
            - 若 ignore_first_collision=True，则全路径第一次碰撞惩罚会被忽略。
            - 若 ignore_last_collision=True，则全路径最后一次碰撞惩罚会被忽略。
        """

        full_path = [start] + path + [end]
        collision_flags: List[bool] = []
        height_rewards: List[float] = []

        for i in range(1, len(full_path)):
            p_prev = full_path[i - 1]
            p_curr = full_path[i]

            nearest_idx = self._nearest_obstacle_indices((p_curr[0], p_curr[1]))
            for obs_idx in nearest_idx:
                obs = self.obstacles[int(obs_idx)]

                collide = False
                if (p_curr[0] - obs.x) ** 2 + (p_curr[1] - obs.y) ** 2 <= obs.r ** 2:
                    collide = True
                if (p_prev[0] - obs.x) ** 2 + (p_prev[1] - obs.y) ** 2 <= obs.r ** 2:
                    collide = True
                if self._segment_intersect_cylinder(p_prev, p_curr, obs.x, obs.y, obs.r):
                    collide = True

                H_i = p_curr[2]
                building_top = float(self._obs_top[int(obs_idx)])
                hr = -self.c_hr if H_i >= building_top else 0.0

                collision_flags.append(collide)
                height_rewards.append(hr)

        collision_indices = [idx for idx, is_collision in enumerate(collision_flags) if is_collision]
        ignored_collision_indices = set()

        if self.ignore_first_collision and collision_indices:
            ignored_collision_indices.add(collision_indices[0])
        if self.ignore_last_collision and collision_indices:
            ignored_collision_indices.add(collision_indices[-1])

        total_B = 0.0
        for idx, collide in enumerate(collision_flags):
            if collide and idx not in ignored_collision_indices:
                total_B += self.c_B
            total_B += height_rewards[idx]

        return float(total_B)

    def flight_distance_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算飞行总路程代价。

        参数:
            path: 中间路径点列表
            start: 起点
            end: 终点

        返回:
            起点 -> 路径点 -> 终点 的三维欧式距离总和。
        """

        if not path:
            return 0.0

        dist = sqrt(
            (path[0][0] - start[0]) ** 2
            + (path[0][1] - start[1]) ** 2
            + (path[0][2] - start[2]) ** 2
        )

        for i in range(1, len(path)):
            dist += sqrt(
                (path[i][0] - path[i - 1][0]) ** 2
                + (path[i][1] - path[i - 1][1]) ** 2
                + (path[i][2] - path[i - 1][2]) ** 2
            )

        dist += sqrt(
            (end[0] - path[-1][0]) ** 2
            + (end[1] - path[-1][1]) ** 2
            + (end[2] - path[-1][2]) ** 2
        )
        return float(dist)

    def altitude_variation_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算高度变化代价。

        参数:
            path: 中间路径点列表
            start: 起点
            end: 终点

        返回:
            相邻航段高度差绝对值之和。
        """

        if not path:
            return 0.0

        delta = abs(path[0][2] - start[2])
        for i in range(1, len(path)):
            delta += abs(path[i][2] - path[i - 1][2])
        delta += abs(end[2] - path[-1][2])

        return float(delta)

    def turning_angle_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算转角约束代价。

        参数:
            path: 中间路径点列表
            start: 起点
            end: 终点

        返回:
            相邻航段航向角变化绝对值之和，再乘以 c_theta。
        """

        points = [start] + path + [end]
        if len(points) < 3:
            return 0.0

        angles = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            angles.append(atan2(dy, dx))

        total_angle_change = 0.0
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i - 1]
            diff = abs((diff + pi) % (2 * pi) - pi)
            total_angle_change += diff

        return float(total_angle_change * self.c_theta)

    def total_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算路径总成本。

        参数:
            path: 中间路径点列表
            start: 起点
            end: 终点

        返回:
            地形代价 + 障碍物代价 + 路程代价 + 高度变化代价 + 转角代价。
        """

        tc = self.terrain_cost(path)
        oc = self.obstacle_collision_cost(path, start, end)
        fc = self.flight_distance_cost(path, start, end)
        ac = self.altitude_variation_cost(path, start, end)
        angc = self.turning_angle_cost(path, start, end)
        return float(tc + oc + fc + ac + angc)


def _ensure_required_columns(
    actual_columns: Sequence[str], required_columns: Sequence[str], csv_path: Path
) -> None:
    """校验 CSV 是否包含必需字段。

    参数:
        actual_columns: CSV 实际字段名列表
        required_columns: 必需字段名列表
        csv_path: CSV 文件路径（用于报错信息）

    异常:
        ValueError: 当缺少必需字段时抛出。
    """

    missing = [col for col in required_columns if col not in actual_columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}. "
            f"Available columns: {list(actual_columns)}"
        )


def load_terrain_from_csv(csv_path: str) -> Terrain:
    """从地形 CSV 构建 Terrain 对象。

    参数:
        csv_path: 地形数据路径，需包含列 Row, Col, X, Y, T_elevation

    返回:
        Terrain 对象

    说明:
        - 自动识别网格尺寸
        - 若有缺失格点，会用邻域均值（或全局均值）填补。
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Terrain CSV not found: {path}")

    samples: List[Tuple[int, int, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        _ensure_required_columns(reader.fieldnames, ["Row", "Col", "X", "Y", "T_elevation"], path)

        for line_no, row in enumerate(reader, start=2):
            try:
                r = int(float(row["Row"]))
                c = int(float(row["Col"]))
                x = float(row["X"])
                y = float(row["Y"])
                h = float(row["T_elevation"])
            except Exception as exc:
                raise ValueError(f"Invalid terrain record at line {line_no}: {row}") from exc
            samples.append((r, c, x, y, h))

    if not samples:
        raise ValueError(f"{path} has no terrain data rows")

    row_ids = sorted({r for r, _, _, _, _ in samples})
    col_ids = sorted({c for _, c, _, _, _ in samples})
    row_to_idx = {r: i for i, r in enumerate(row_ids)}
    col_to_idx = {c: i for i, c in enumerate(col_ids)}

    ny = len(row_ids)
    nx = len(col_ids)

    elev_sum = np.zeros((ny, nx), dtype=float)
    elev_count = np.zeros((ny, nx), dtype=int)

    x_sum: Dict[int, float] = {c: 0.0 for c in col_ids}
    x_count: Dict[int, int] = {c: 0 for c in col_ids}
    y_sum: Dict[int, float] = {r: 0.0 for r in row_ids}
    y_count: Dict[int, int] = {r: 0 for r in row_ids}

    for r, c, x, y, h in samples:
        i = row_to_idx[r]
        j = col_to_idx[c]
        elev_sum[i, j] += h
        elev_count[i, j] += 1

        x_sum[c] += x
        x_count[c] += 1
        y_sum[r] += y
        y_count[r] += 1

    elevation = np.full((ny, nx), np.nan, dtype=float)
    valid = elev_count > 0
    elevation[valid] = elev_sum[valid] / elev_count[valid]

    x_coords = np.array([x_sum[c] / x_count[c] for c in col_ids], dtype=float)
    y_coords = np.array([y_sum[r] / y_count[r] for r in row_ids], dtype=float)

    x_order = np.argsort(x_coords)
    y_order = np.argsort(y_coords)

    x_coords = x_coords[x_order]
    y_coords = y_coords[y_order]
    elevation = elevation[np.ix_(y_order, x_order)]

    if np.isnan(elevation).any():
        global_mean = float(np.nanmean(elevation))
        if np.isnan(global_mean):
            raise ValueError(f"All terrain elevations are NaN in {path}")

        nan_positions = np.argwhere(np.isnan(elevation))
        for iy, ix in nan_positions:
            neighbors = []
            if iy > 0 and not np.isnan(elevation[iy - 1, ix]):
                neighbors.append(elevation[iy - 1, ix])
            if iy < elevation.shape[0] - 1 and not np.isnan(elevation[iy + 1, ix]):
                neighbors.append(elevation[iy + 1, ix])
            if ix > 0 and not np.isnan(elevation[iy, ix - 1]):
                neighbors.append(elevation[iy, ix - 1])
            if ix < elevation.shape[1] - 1 and not np.isnan(elevation[iy, ix + 1]):
                neighbors.append(elevation[iy, ix + 1])
            elevation[iy, ix] = float(np.mean(neighbors)) if neighbors else global_mean

    return Terrain(elevation_matrix=elevation, x_coords=x_coords, y_coords=y_coords)


def load_obstacles_from_csv(csv_path: str, min_radius: float = 0.0) -> List[Obstacle]:
    """从建筑物 CSV 构建障碍物列表。

    参数:
        csv_path: 建筑数据路径，需包含列 Center_X, Center_Y, Elevation, Radius_m
        min_radius: 最小半径阈值，半径 <= 该值的记录会被过滤

    返回:
        Obstacle 列表

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 表头缺失、字段非法或无可用障碍物
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Building CSV not found: {path}")

    obstacles: List[Obstacle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        _ensure_required_columns(reader.fieldnames, ["Center_X", "Center_Y", "Elevation", "Radius_m"], path)

        for line_no, row in enumerate(reader, start=2):
            try:
                x = float(row["Center_X"])
                y = float(row["Center_Y"])
                z = float(row["Elevation"])
                r = float(row["Radius_m"])
            except Exception as exc:
                raise ValueError(f"Invalid obstacle record at line {line_no}: {row}") from exc

            if r <= min_radius:
                continue

            obstacles.append(Obstacle(x=x, y=y, z=z, r=r))

    if not obstacles:
        raise ValueError(
            f"No valid obstacles loaded from {path}. "
            "Check Radius_m and other numeric fields."
        )

    return obstacles


def build_calculator_from_csv(
    terrain_csv_path: str,
    building_csv_path: str,
    **calculator_kwargs: Any,
) -> UAVPathCostCalculator:
    """从 CSV 一步构建 UAVPathCostCalculator。

    参数:
        terrain_csv_path: 地形 CSV 路径
        building_csv_path: 建筑 CSV 路径
        **calculator_kwargs: 透传给 UAVPathCostCalculator 的参数

    返回:
        初始化完成的 UAVPathCostCalculator 实例。
    """

    terrain = load_terrain_from_csv(terrain_csv_path)
    obstacles = load_obstacles_from_csv(building_csv_path)
    return UAVPathCostCalculator(terrain=terrain, obstacles=obstacles, **calculator_kwargs)


def _default_data_paths() -> Tuple[Path, Path]:
    """返回项目默认数据路径。"""

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "数据"
    terrain_csv = data_dir / "投影加剪裁后的地形高度数据.csv"
    building_csv = data_dir / "裁剪后的建筑物数据_简洁版.csv"
    return terrain_csv, building_csv


def main() -> None:
    """模块演示入口。

    逻辑:
        1. 若默认数据文件存在，则加载真实数据并计算示例路径代价。
        2. 否则回退到随机模拟数据，保证脚本可独立运行。
    """

    terrain_csv, building_csv = _default_data_paths()

    if terrain_csv.exists() and building_csv.exists():
        calculator = build_calculator_from_csv(
            terrain_csv_path=str(terrain_csv),
            building_csv_path=str(building_csv),
            c_T=1000.0,
            c_B=100.0,
            c_hr=20.0,
            c_theta=20.0 / pi,
            k_nearest=4,
        )

        terrain = calculator.terrain
        obstacles = calculator.obstacles

        x_span = terrain.x_max - terrain.x_min
        y_span = terrain.y_max - terrain.y_min

        start_xy = (terrain.x_min + 0.1 * x_span, terrain.y_min + 0.1 * y_span)
        end_xy = (terrain.x_min + 0.9 * x_span, terrain.y_min + 0.9 * y_span)

        start = (
            start_xy[0],
            start_xy[1],
            terrain.get_elevation(start_xy[0], start_xy[1]) + 90.0,
        )
        end = (
            end_xy[0],
            end_xy[1],
            terrain.get_elevation(end_xy[0], end_xy[1]) + 90.0,
        )

        path = []
        for t in (0.25, 0.45, 0.65, 0.8):
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            h = terrain.get_elevation(x, y) + 95.0 + 10.0 * np.sin(t * pi)
            path.append((x, y, float(h)))

        total = calculator.total_cost(path, start, end)

        print("Loaded real dataset successfully.")
        print(f"Terrain grid: {terrain.ny} x {terrain.nx}")
        print(f"Obstacle count: {len(obstacles)}")
        print(f"X range: [{terrain.x_min:.3f}, {terrain.x_max:.3f}]")
        print(f"Y range: [{terrain.y_min:.3f}, {terrain.y_max:.3f}]")
        print(f"Total cost: {total:.2f}")
        print(f"Terrain cost: {calculator.terrain_cost(path):.2f}")
        print(f"Obstacle cost: {calculator.obstacle_collision_cost(path, start, end):.2f}")
        print(f"Distance cost: {calculator.flight_distance_cost(path, start, end):.2f}")
        print(f"Altitude cost: {calculator.altitude_variation_cost(path, start, end):.2f}")
        print(f"Turning cost: {calculator.turning_angle_cost(path, start, end):.2f}")
    else:
        np.random.seed(42)
        demo_elevation = np.random.uniform(0, 50, size=(128, 128))
        terrain = Terrain(demo_elevation, x_range=(0, 1000), y_range=(0, 1000))

        obstacles = [
            Obstacle(x=300, y=400, z=30, r=25),
            Obstacle(x=600, y=200, z=45, r=30),
            Obstacle(x=800, y=700, z=20, r=20),
        ]

        calculator = UAVPathCostCalculator(terrain, obstacles)

        start_point = (100.0, 100.0, 60.0)
        end_point = (900.0, 900.0, 55.0)
        test_path = [
            (200.0, 200.0, 70.0),
            (350.0, 400.0, 80.0),
            (500.0, 300.0, 75.0),
            (700.0, 600.0, 90.0),
        ]

        total = calculator.total_cost(test_path, start_point, end_point)
        print("Fallback to demo data.")
        print(f"Total cost: {total:.2f}")


if __name__ == "__main__":
    main()
