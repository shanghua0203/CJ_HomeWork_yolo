"""
===========================================
雜訊過濾模組 (filter.py)
===========================================

專門過濾乒乓球軌跡中的雜訊，確保座標合理性

主要功能：
1. 濾除超出畫面範圍的座標
2. 濾除瞬間跳動過大的點（相鄰兩幀距離異常）
3. 平滑軌跡座標

作者：Python 影像辨識工程師
"""

import numpy as np
from typing import List, Tuple, Optional


class TrajectoryFilter:
    """
    軌跡雜訊過濾器
    過濾不合理的座標，確保軌跡平滑連貫
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        max_jump: int = 100,
        max_missing_frames: int = 2
    ):
        """
        初始化過濾器

        參數說明：
        - frame_width: 影片畫面寬度
        - frame_height: 影片畫面高度
        - max_jump: 相鄰兩幀最大允許位移（pixels），超過視為雜訊
        - max_missing_frames: 允許球消失的最多幀數
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_jump = max_jump
        self.max_missing_frames = max_missing_frames

    def is_valid_coordinate(self, x: int, y: int) -> bool:
        """
        檢查座標是否在合理範圍內

        參數：
        - x, y: 影像座標

        回傳：
        - True 如果座標在畫面範圍內
        """
        return 0 <= x < self.frame_width and 0 <= y < self.frame_height

    def is_jump_too_large(
        self,
        prev_x: Optional[int],
        prev_y: Optional[int],
        curr_x: int,
        curr_y: int
    ) -> bool:
        """
        檢查兩幀之間的位移是否過大

        參數：
        - prev_x, prev_y: 上一幀座標
        - curr_x, curr_y: 當前幀座標

        回傳：
        - True 如果位移超過 max_jump
        """
        if prev_x is None or prev_y is None:
            return False

        # 計算歐氏距離
        distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)

        return bool(distance > self.max_jump)

    def filter_single_coordinate(
        self,
        x: int,
        y: int,
        prev_x: Optional[int] = None,
        prev_y: Optional[int] = None
    ) -> Optional[Tuple[int, int]]:
        """
        過濾單個座標點

        參數：
        - x, y: 待檢查的座標
        - prev_x, prev_y: 上一個有效的座標（可選）

        回傳：
        - (x, y) 如果座標有效，否則回傳 None
        """
        # 檢查 1：座標是否在畫面範圍內
        if not self.is_valid_coordinate(x, y):
            return None

        # 檢查 2：與上一幀的位移是否過大
        if prev_x is not None and prev_y is not None:
            if self.is_jump_too_large(prev_x, prev_y, x, y):
                return None

        return (x, y)

    def filter_trajectory(
        self,
        trajectory: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        過濾一串軌跡座標

        參數：
        - trajectory: [(x1, y1), (x2, y2), ...] 原始軌跡

        回傳：
        - 過濾後的軌跡
        """
        if not trajectory:
            return []

        filtered = []
        prev_x, prev_y = None, None

        for x, y in trajectory:
            # 檢查座標有效性
            if not self.is_valid_coordinate(x, y):
                continue  # 跳過無效座標，不更新上一個座標

            # 檢查位移是否過大（只和有效的上一個座標比較）
            if prev_x is not None and prev_y is not None:
                if self.is_jump_too_large(prev_x, prev_y, x, y):
                    continue  # 跳過異常座標，不更新上一個座標

            filtered.append((x, y))
            prev_x, prev_y = x, y  # 只有通過檢查才更新

        return filtered

    def interpolate_missing(
        self,
        trajectory: List[Optional[Tuple[int, int]]],
        frame_indices: List[int]
    ) -> List[Optional[Tuple[int, int]]]:
        """
        對掉幀進行線性插值補間

        當球消失 1-2 幀後重新出現時，用插值填補遺失的座標

        參數：
        - trajectory: 原始軌跡（可能包含 None）
        - frame_indices: 對應的幀編號

        回傳：
        - 補間後的完整軌跡
        """
        if not trajectory:
            return []

        # 如果沒有提供 frame_indices，使用索引作為幀編號
        if not frame_indices:
            frame_indices = list(range(len(trajectory)))

        # 找出有效的座標點
        valid_points = []
        valid_indices = []

        for i, point in enumerate(trajectory):
            if point is not None and point[0] is not None and point[1] is not None:
                valid_points.append(point)
                valid_indices.append(frame_indices[i] if i < len(frame_indices) else i)

        if not valid_points:
            return [None] * len(trajectory)

        # 對每一幀計算插值座標
        result = []

        for i in range(len(trajectory)):
            current_frame = frame_indices[i] if i < len(frame_indices) else i

            # 找前後有效的點
            prev_point = None
            prev_frame = None
            next_point = None
            next_frame = None

            for j, idx in enumerate(valid_indices):
                if idx <= current_frame:
                    prev_point = valid_points[j]
                    prev_frame = idx
                if idx >= current_frame and next_point is None:
                    next_point = valid_points[j]
                    next_frame = idx

            # 計算插值座標
            if prev_point is not None and next_point is not None:
                # 兩邊都有有效點，進行線性插值
                if next_frame != prev_frame:
                    t = (current_frame - prev_frame) / (next_frame - prev_frame)
                    interp_x = int(prev_point[0] + t * (next_point[0] - prev_point[0]))
                    interp_y = int(prev_point[1] + t * (next_point[1] - prev_point[1]))
                    result.append((interp_x, interp_y))
                else:
                    result.append(prev_point)
            elif prev_point is not None:
                result.append(prev_point)
            elif next_point is not None:
                result.append(next_point)
            else:
                result.append(None)

        return result


def smooth_trajectory(
    trajectory: List[Tuple[int, int]],
    window_size: int = 3
) -> List[Tuple[int, int]]:
    """
    使用移動平均平滑軌跡

    參數：
    - trajectory: 原始軌跡
    - window_size: 平滑視窗大小（奇數）

    回傳：
    - 平滑後的軌跡
    """
    if len(trajectory) < window_size:
        return trajectory

    if window_size % 2 == 0:
        window_size += 1  # 確保為奇數

    half_window = window_size // 2
    smoothed = []

    for i in range(len(trajectory)):
        start = max(0, i - half_window)
        end = min(len(trajectory), i + half_window + 1)

        # 計算該區間的平均座標
        window_points = trajectory[start:end]
        avg_x = int(np.mean([p[0] for p in window_points]))
        avg_y = int(np.mean([p[1] for p in window_points]))

        smoothed.append((avg_x, avg_y))

    return smoothed


def remove_outliers(
    trajectory: List[Tuple[int, int]],
    threshold: float = 2.0
) -> List[Tuple[int, int]]:
    """
    移除統計離群值

    使用 Z-score 方法，移除偏離平均距離過大的點

    參數：
    - trajectory: 原始軌跡
        threshold: Z-score 閾值（預設 2.0）

    回傳：
    - 移除離群值後的軌跡
    """
    if len(trajectory) < 3:
        return trajectory

    # 計算每個點到中心點的距離
    center_x = np.mean([p[0] for p in trajectory])
    center_y = np.mean([p[1] for p in trajectory])

    distances = [
        np.sqrt((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2)
        for p in trajectory
    ]

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    if std_dist == 0:
        return trajectory

    # 濾除 Z-score 超過閾值的點
    filtered = []
    for i, point in enumerate(trajectory):
        z_score = (distances[i] - mean_dist) / std_dist
        if abs(z_score) <= threshold:
            filtered.append(point)

    return filtered


# ===========================================
# 如果直接執行此檔案，進行測試
# ===========================================
if __name__ == "__main__":
    print("軌跡過濾模組測試")
    print("=" * 40)

    # 建立過濾器
    filter_obj = TrajectoryFilter(frame_width=640, frame_height=480, max_jump=100)

    # 測試 1：有效座標
    result = filter_obj.is_valid_coordinate(100, 200)
    print(f"測試 1 - 座標 (100, 200) 是否有效：{result}")

    # 測試 2：超出範圍
    result = filter_obj.is_valid_coordinate(700, 200)
    print(f"測試 2 - 座標 (700, 200) 是否有效：{result}")

    # 測試 3：跳動過大
    result = filter_obj.is_jump_too_large(100, 100, 300, 300)
    print(f"測試 3 - 從 (100,100) 跳到 (300,300) 是否過大：{result}")

    # 測試 4：過濾軌跡
    trajectory = [(100, 100), (110, 110), (500, 500), (120, 120)]
    filtered = filter_obj.filter_trajectory(trajectory)
    print(f"測試 4 - 原始軌跡：{trajectory}")
    print(f"         過濾後：{filtered}")