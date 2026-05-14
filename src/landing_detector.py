"""
===========================================
落點判斷模組 (landing_detector.py)
===========================================

分析乒乓球軌跡中的 Y 軸變化，判斷落點位置

主要功能：
1. 分析 Y 軸座標的變化趋势
2. 找出 Y 軸變大後突然變小的「反弾點」
3. 判斷球撞擊桌面的「落點」

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
    # 落點座標
    x: int
    y: int

    # 對應的幀編號
    frame_index: int

    # 反弾前的最低 Y 值
    lowest_y: int

    # 該落點來源的軌跡片段
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

        參數說明：
        - y_reversal_threshold: Y 軸反転閾值（pixels）
          當 Y 軸從變大變成變小，且變化幅度超過此值，視為反弾
        - min_fall_distance: 最小下落距離
          落點前至少需要下跌的距離，過短的軌跡不視為有效落點
        - min_trajectory_length: 最小軌跡長度
          用於判斷落點的軌跡至少需要有多少個點
        """
        self.y_reversal_threshold = y_reversal_threshold
        self.min_fall_distance = min_fall_distance
        self.min_trajectory_length = min_trajectory_length

    def detect_y_reversal(
        self,
        y_values: List[int]
    ) -> List[int]:
        """
        偵測 Y 軸反転點

        分析 Y 值序列，找出「變大後突然變小」的位置

        參數：
        - y_values: Y 軸座標列表

        回傳：
        - 反転點的索引列表
        """
        if len(y_values) < 3:
            return []

        reversals = []

        for i in range(1, len(y_values) - 1):
            prev_y = y_values[i - 1]
            curr_y = y_values[i]
            next_y = y_values[i + 1]

            # 判斷模式：Y 變大（下跌）然後變小（反弾）
            # 在影像中，Y 變大 = 球往下移動
            if curr_y > prev_y and next_y < curr_y:
                # 計算反転幅度
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
        找出反転前的最低點

        在反転點之前找到 Y 值最大的點（即畫面中最下面的位置）

        參數：
        - trajectory: 軌跡座標
        - reversal_index: 反転點索引

        回傳：
        - (x, y, 索引)
        """
        segment = trajectory[:reversal_index + 1]

        if not segment:
            return (-1, -1, -1)

        # 找出 Y 最大的點（畫面中最下面）
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

        判斷條件：
        1. 落點前有足夠的下落距離
        2. 軌跡長度足夠

        參數：
        - trajectory: 完整軌跡
        - landing_point_index: 落點索引

        回傳：
        - True 如果落點有效
        """
        # 檢查軌跡長度
        if len(trajectory) < self.min_trajectory_length:
            return False

        # 檢查下落距離
        # 比較落點前第一個點和最低點的 Y 差
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
        偵測所有落點

        主入口函數，分析軌跡找出所有有效的落點

        參數：
        - trajectory: 軌跡座標 [(x1, y1), (x2, y2), ...]
        - frame_indices: 對應的幀編號（可選）

        回傳：
        - LandingPoint 列表
        """
        if len(trajectory) < self.min_trajectory_length:
            return []

        # 如果沒有提供幀編號，使用索引
        if frame_indices is None:
            frame_indices = list(range(len(trajectory)))

        # 取出 Y 值序列
        y_values = [p[1] for p in trajectory]

        # 偵測反転點
        reversal_indices = self.detect_y_reversal(y_values)

        # 收集所有落點
        landing_points = []

        for rev_idx in reversal_indices:
            # 找出反転前的最低點
            x, y, local_idx = self.find_lowest_before_reversal(
                trajectory, rev_idx
            )

            if x < 0:
                continue

            # 檢查是否有效落點
            if not self.is_valid_landing(trajectory, local_idx):
                continue

            # 取得該落點的軌跡片段
            segment_end = local_idx + 1
            segment = trajectory[:segment_end]

            # 建立落點物件
            landing = LandingPoint(
                x=x,
                y=y,
                frame_index=frame_indices[local_idx],
                lowest_y=y,
                trajectory_segment=segment
            )

            landing_points.append(landing)

        return landing_points

    def get_first_landing(
        self,
        trajectory: List[Tuple[int, int]],
        frame_indices: Optional[List[int]] = None
    ) -> Optional[LandingPoint]:
        """
        取得第一個落點

        參數：
        - trajectory: 軌跡座標
        - frame_indices: 對應的幀編號

        回傳：
        - 第一個有效的落點，沒有則回傳 None
        """
        landings = self.detect(trajectory, frame_indices)

        if landings:
            return landings[0]
        return None

    def get_last_landing(
        self,
        trajectory: List[Tuple[int, int]],
        frame_indices: Optional[List[int]] = None
    ) -> Optional[LandingPoint]:
        """
        取得最後一個落點

        參數：
        - trajectory: 軌跡座標
        - frame_indices: 對應的幀編號

        回傳：
        - 最後一個有效的落點，沒有則回傳 None
        """
        landings = self.detect(trajectory, frame_indices)

        if landings:
            return landings[-1]
        return None


def analyze_y_trend(y_values: List[int]) -> Dict:
    """
    分析 Y 軸趨勢

    計算一段時間內 Y 值的基本統計

    參數：
    - y_values: Y 軸座標列表

    回傳：
    - 包含統計資訊的字典
    """
    if not y_values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "is_falling": False,
            "is_rising": False
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


def find_y_extremes(
    y_values: List[int]
) -> Tuple[Optional[int], Optional[int]]:
    """
    找出 Y 軸的極值點

    找出 Y 值最大（最低點）和最小（最高點）的索引

    參數：
    - y_values: Y 軸座標列表

    回傳：
    - (最低點索引, 最高點索引)
    """
    if not y_values:
        return None, None

    min_idx = int(np.argmin(y_values))
    max_idx = int(np.argmax(y_values))

    return (min_idx, max_idx)


# ===========================================
# 如果直接執行此檔案，進行測試
# ===========================================
if __name__ == "__main__":
    print("落點判斷模組測試")
    print("=" * 40)

    # 建立偵測器
    detector = LandingDetector(
        y_reversal_threshold=5,
        min_fall_distance=20
    )

    # 模擬軌跡：球從上往下，然後反弾
    # Y 值變大 = 往下移動
    trajectory = [
        (100, 50),   # 幀 0
        (110, 100),  # 幀 1 - 繼續下跌
        (115, 150),  # 幀 2 - 繼續下跌
        (120, 180),  # 幀 3 - 最低點（接觸桌面）
        (125, 160),  # 幀 4 - 反弾！
        (130, 130),  # 幀 5 - 繼續上升
    ]

    # 偵測落點
    landings = detector.detect(trajectory)

    print(f"偵測到 {len(landings)} 個落點")
    for i, landing in enumerate(landings):
        print(f"  落點 {i + 1}: 座標 ({landing.x}, {landing.y}), 幀 {landing.frame_index}")

    # 分析 Y 趨勢
    y_values = [p[1] for p in trajectory]
    trend = analyze_y_trend(y_values)
    print(f"\nY 軸趨勢分析：{trend}")