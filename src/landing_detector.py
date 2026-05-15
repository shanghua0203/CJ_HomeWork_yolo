"""
===========================================
落點判斷模組 (landing_detector.py)
===========================================

分析乒乓球軌跡中的 Y 軸變化，判斷落點位置

作者：Python 影像辨識工程師
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LandingPoint:
    """
    落點資料類別
    儲存一個落點的完整資訊
    """
    x: int
    y: int
    frame_index: int
    lowest_y: int
    trajectory_segment: List[Tuple[int, int]]


class LandingDetector:
    """
    乒乓球落點偵測器
    分析軌跡找出球反弾的落點
    """

    def __init__(
        self,
        y_reversal_threshold: int = 5,
        min_fall_distance: int = 20,
        min_trajectory_length: int = 3
    ):
        """
        初始化落點偵測器
        """
        self.y_reversal_threshold = y_reversal_threshold
        self.min_fall_distance = min_fall_distance
        self.min_trajectory_length = min_trajectory_length
        self._processed_index = 0

    def detect_y_reversal(self, y_values: List[int]) -> List[int]:
        """
        偵測 Y 軸反呑點
        分析 Y 值序列，找出變大後突然變小的位置
        """
        if len(y_values) < 3:
            return []

        reversals = []

        for i in range(1, len(y_values) - 1):
            prev_y = y_values[i - 1]
            curr_y = y_values[i]
            next_y = y_values[i + 1]

            if curr_y > prev_y and next_y < curr_y:
                reversal_size = curr_y - next_y
                if reversal_size >= self.y_reversal_threshold:
                    reversals.append(i)

        return reversals

    def find_lowest_before_reversal(
        self,
        trajectory: List[Tuple[int, int]],
        reversal_index: int
    ) -> Tuple[int, int, int]:
        """
        找出反呑前的最低點
        """
        segment = trajectory[:reversal_index + 1]

        if not segment:
            return (-1, -1, -1)

        max_y = -1
        lowest_point = segment[0]
        lowest_idx = 0

        for i, point in enumerate(segment):
            if point[1] > max_y:
                max_y = point[1]
                lowest_point = point
                lowest_idx = i

        return (lowest_point[0], lowest_point[1], lowest_idx)

    def is_valid_landing(
        self,
        trajectory: List[Tuple[int, int]],
        landing_point_index: int
    ) -> bool:
        """
        檢查落點是否有效
        """
        if len(trajectory) < self.min_trajectory_length:
            return False

        if landing_point_index > 0:
            start_y = trajectory[0][1]
            end_y = trajectory[landing_point_index][1]
            fall_distance = end_y - start_y

            if fall_distance < self.min_fall_distance:
                return False

        return True

    def detect(
        self,
        trajectory: List[Tuple[int, int]],
        frame_indices: Optional[List[int]] = None
    ) -> List[LandingPoint]:
        """
        偵測所有落點（僅分析新加入的軌跡點，防止重複偵測）
        """
        if len(trajectory) < self.min_trajectory_length:
            return []

        if frame_indices is None:
            frame_indices = list(range(len(trajectory)))

        start_index = self._processed_index
        if start_index >= len(trajectory):
            return []

        new_trajectory = trajectory[start_index:]
        new_frame_indices = frame_indices[start_index:]

        if len(new_trajectory) < 3:
            return []

        y_values = [p[1] for p in new_trajectory]
        reversal_indices = self.detect_y_reversal(y_values)

        landing_points = []

        for rev_idx in reversal_indices:
            x, y, _ = self.find_lowest_before_reversal(new_trajectory, rev_idx)

            if x < 0:
                continue

            landing_idx = start_index + rev_idx
            full_segment = trajectory[:landing_idx + 1]

            if not self.is_valid_landing(full_segment, landing_idx):
                continue

            segment = trajectory[:landing_idx + 1]

            landing = LandingPoint(
                x=x,
                y=y,
                frame_index=frame_indices[landing_idx],
                lowest_y=y,
                trajectory_segment=segment
            )

            landing_points.append(landing)

        if landing_points:
            self._processed_index = len(trajectory)

        return landing_points

    def reset(self):
        """重置已處理索引"""
        self._processed_index = 0

    def get_first_landing(
        self,
        trajectory: List[Tuple[int, int]],
        frame_indices: Optional[List[int]] = None
    ) -> Optional[LandingPoint]:
        """取得第一個落點"""
        landings = self.detect(trajectory, frame_indices)
        return landings[0] if landings else None

    def get_last_landing(
        self,
        trajectory: List[Tuple[int, int]],
        frame_indices: Optional[List[int]] = None
    ) -> Optional[LandingPoint]:
        """取得最後一個落點"""
        landings = self.detect(trajectory, frame_indices)
        return landings[-1] if landings else None


def analyze_y_trend(y_values: List[int]) -> Dict:
    """分析 Y 軸趨勢"""
    if not y_values:
        return {
            "count": 0, "min": None, "max": None,
            "mean": None, "is_falling": False, "is_rising": False
        }

    y_array = np.array(y_values)

    return {
        "count": len(y_values),
        "min": int(np.min(y_array)),
        "max": int(np.max(y_array)),
        "mean": float(np.mean(y_array)),
        "is_falling": y_values[-1] > y_values[0],
        "is_rising": y_values[-1] < y_values[0]
    }


def find_y_extremes(y_values: List[int]) -> Tuple[Optional[int], Optional[int]]:
    """找出 Y 軸的極值點"""
    if not y_values:
        return None, None

    min_idx = int(np.argmin(y_values))
    max_idx = int(np.argmax(y_values))

    return (min_idx, max_idx)
