"""
===========================================
軌跡追蹤模組 (tracker.py)
===========================================

記錄乒乓球連續幀的座標，並繪製軌跡線

主要功能：
1. 持續追蹤並記錄球的位置
2. 處理掉幀問題（球消失 1-2 格的插值連線）
3. 繪製軌跡線

作者：Python 影像辨識工程師
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class TrackingState:
    """
    追蹤狀態資料類別
    用於儲存當前追蹤的所有狀態資訊
    """
    # 軌跡座標列表
    trajectory: List[Tuple[int, int]] = field(default_factory=list)

    # 對應的幀編號
    frame_indices: List[int] = field(default_factory=list)

    # 連續消失的幀數計數
    missing_frames: int = 0

    # 是否正在追蹤
    is_tracking: bool = False

    # 上一個有效的座標
    last_valid_point: Optional[Tuple[int, int]] = None

    # 上一個有效的幀編號
    last_valid_frame: int = -1


class BallTracker:
    """
    乒乓球軌跡追蹤器
    持續追蹤球的位置並記錄軌跡
    """

    def __init__(
        self,
        max_missing_frames: int = 2,
        min_trajectory_length: int = 2,
        max_trajectory_length: int = 5000,
        trajectory_color: Tuple[int, int, int] = (0, 255, 0),
        trajectory_thickness: int = 2,
        ball_radius: int = 5,
        ball_color: Tuple[int, int, int] = (0, 0, 255)
    ):
        """
        初始化追蹤器

        參數說明：
        - max_missing_frames: 允許球消失的最大幀數（掉幀容忍度）
        - min_trajectory_length: 最少需要的軌跡點數量
        - max_trajectory_length: 軌跡點數量上限，超過時自動截斷舊資料
        - trajectory_color: 軌跡線顏色 (B, G, R)
        - trajectory_thickness: 軌跡線粗細
        - ball_radius: 球標記半徑
        - ball_color: 球標記顏色
        """
        self.max_missing_frames = max_missing_frames
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.trajectory_color = trajectory_color
        self.trajectory_thickness = trajectory_thickness
        self.ball_radius = ball_radius
        self.ball_color = ball_color

        # 建立追蹤狀態
        self.state = TrackingState()

    def reset(self):
        """
        重置追蹤狀態
        清除所有記錄，重新開始追蹤
        """
        self.state = TrackingState()

    def update(
        self,
        detection: Optional[Tuple[int, int]],
        frame_number: int
    ) -> bool:
        """
        更新追蹤狀態

        參數：
        - detection: 偵測到的座標 (x, y)，如果沒偵測到則為 None
        - frame_number: 當前幀編號

        回傳：
        - True 如果有新增軌跡點，否則為 False
        """
        has_new_point = False

        if detection is not None:
            if hasattr(detection, '__len__') and len(detection) >= 2:
                x, y = detection[0], detection[1]
            else:
                x, y = detection

            # 有偵測到球
            self.state.is_tracking = True
            self.state.missing_frames = 0
            self.state.last_valid_point = (x, y)
            self.state.last_valid_frame = frame_number

            # 加入軌跡
            self.state.trajectory.append((x, y))
            self.state.frame_indices.append(frame_number)
            has_new_point = True

            # 截斷舊資料，防止記憶體無限制增長
            if len(self.state.trajectory) > self.max_trajectory_length:
                excess = len(self.state.trajectory) - self.max_trajectory_length
                self.state.trajectory = self.state.trajectory[excess:]
                self.state.frame_indices = self.state.frame_indices[excess:]

        else:
            # 沒偵測到球
            self.state.missing_frames += 1

            # 如果超出容忍範圍，停止追蹤
            if self.state.missing_frames > self.max_missing_frames:
                self.state.is_tracking = False

        return has_new_point

    def interpolate_dropped_frames(
        self,
        detection: Optional[Tuple[int, int]],
        frame_number: int
    ) -> List[Tuple[int, int]]:
        """
        對掉幀進行插值處理

        當球消失時，根據前後座標進行線性插值補間

        參數：
        - detection: 當前偵測結果
        - frame_number: 當前幀編號

        回傳：
        - 插值後的座標列表
        """
        if detection is not None:
            return [detection]

        if self.state.missing_frames == 0:
            return []

        if self.state.last_valid_point is None:
            return []

        # 如果消失幀數過多，不進行插值
        if self.state.missing_frames > self.max_missing_frames:
            return []

        # 找出最後一個有效點和即將出現的有效點（假設球會回來）
        # 這裡我們用最後一個有效點作為預測
        last_x, last_y = self.state.last_valid_point

        # 簡單的插值：假設球停在原地
        return [(last_x, last_y)]

    def get_trajectory(self) -> List[Tuple[int, int]]:
        """
        取得當前軌跡

        回傳：
        - 軌跡座標列表
        """
        return self.state.trajectory.copy()

    def get_frame_indices(self) -> List[int]:
        """
        取得對應的幀編號

        回傳：
        - 幀編號列表
        """
        return self.state.frame_indices.copy()

    def is_tracking(self) -> bool:
        """
        檢查是否正在追蹤

        回傳：
        - True 如果正在追蹤
        """
        return self.state.is_tracking

    def get_statistics(self) -> Dict:
        """
        取得追蹤統計資訊

        回傳：
        - 包含各種統計數據的字典
        """
        trajectory = self.state.trajectory

        if len(trajectory) < 2:
            return {
                "total_points": len(trajectory),
                "total_frames": len(self.state.frame_indices),
                "missing_frames": sum(1 for f in self.state.frame_indices if f is None),
                "is_tracking": self.state.is_tracking,
                "average_speed": 0,
                "min_y": trajectory[0][1] if trajectory else None,
                "max_y": trajectory[0][1] if trajectory else None
            }

        # 計算 Y 軸統計（用於落點分析）
        y_values = [p[1] for p in trajectory]

        # 計算平均速度
        total_distance = 0
        total_frames = 0

        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            distance = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            total_distance += distance
            total_frames += 1

        avg_speed = total_distance / total_frames if total_frames > 0 else 0

        return {
            "total_points": len(trajectory),
            "total_frames": len(self.state.frame_indices),
            "missing_frames": self.state.missing_frames,
            "is_tracking": self.state.is_tracking,
            "average_speed": avg_speed,
            "min_y": min(y_values),
            "max_y": max(y_values)
        }

    def draw_trajectory(
        self,
        frame: np.ndarray,
        start_index: int = 0,
        show_ball_markers: bool = True
    ) -> np.ndarray:
        """
        在影像上繪製軌跡線

        參數：
        - frame: 原始影像
        - start_index: 開始繪製的軌跡點索引
        - show_ball_markers: 是否顯示球的位置標記

        回傳：
        - 繪製後的影像
        """
        if len(self.state.trajectory) < 2:
            return frame

        # 複製影像以避免修改原始資料
        output = frame.copy()

        trajectory = self.state.trajectory[start_index:]

        if len(trajectory) < 2:
            return output

        # 繪製軌跡線
        for i in range(1, len(trajectory)):
            pt1 = trajectory[i - 1]
            pt2 = trajectory[i]

            cv2.line(
                output,
                pt1,
                pt2,
                self.trajectory_color,
                self.trajectory_thickness
            )

        # 繪製球的位置標記
        if show_ball_markers:
            for i, (x, y) in enumerate(trajectory):
                # 最後一個點用不同顏色標示
                if i == len(trajectory) - 1:
                    cv2.circle(output, (x, y), self.ball_radius + 3, self.ball_color, -1)
                else:
                    cv2.circle(output, (x, y), self.ball_radius, self.trajectory_color, -1)

        return output

    def draw_trajectory_with_predictions(
        self,
        frame: np.ndarray,
        predictions: List[Tuple[int, int]],
        show_prediction_lines: bool = True
    ) -> np.ndarray:
        """
        繪製軌跡線（含預測線段）

        用於顯示掉幀時的插值預測線段

        參數：
        - frame: 原始影像
        - predictions: 插值預測的座標列表
        - show_prediction_lines: 是否顯示預測線段

        回傳：
        - 繪製後的影像
        """
        output = self.draw_trajectory(frame)

        if not show_prediction_lines or not predictions:
            return output

        # 用虛線繪製預測線段
        trajectory = self.state.trajectory
        prediction_color = (255, 255, 0)  # 黃色

        for i, pred in enumerate(predictions):
            if len(trajectory) > 0:
                last_point = trajectory[-1]
                cv2.line(
                    output,
                    last_point,
                    pred,
                    prediction_color,
                    1,
                    lineType=cv2.LINE_AA
                )

        return output


def connect_trajectory_gaps(
    trajectory: List[Tuple[int, int]],
    frame_indices: List[int],
    max_gap: int = 2
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    連接軌跡中的斷點

    當球消失 1-2 幀時，用線性插值填補缺口

    參數：
    - trajectory: 原始軌跡
    - frame_indices: 對應的幀編號
    - max_gap: 最大允許的間隔幀數

    回傳：
    - (填補後的軌跡, 填補後的幀編號)
    """
    if len(trajectory) < 2:
        return trajectory, frame_indices

    result_trajectory = []
    result_indices = []

    for i in range(len(trajectory)):
        result_trajectory.append(trajectory[i])
        result_indices.append(frame_indices[i])

        if i < len(trajectory) - 1:
            current_frame = frame_indices[i]
            next_frame = frame_indices[i + 1]
            gap = next_frame - current_frame

            # 如果間隔在 max_gap 範圍內（不含本身的一幀），進行插值
            # 例如 gap=2 表示間隔 2 幀（中間有一幀需插值）
            if gap > 1 and gap <= max_gap + 1:
                # 進行插值填補
                current_point = trajectory[i]
                next_point = trajectory[i + 1]

                for j in range(1, gap):
                    t = j / gap
                    interp_x = int(current_point[0] + t * (next_point[0] - current_point[0]))
                    interp_y = int(current_point[1] + t * (next_point[1] - current_point[1]))
                    result_trajectory.append((interp_x, interp_y))
                    result_indices.append(current_frame + j)

    return result_trajectory, result_indices


def calculate_trajectory_speed(
    trajectory: List[Tuple[int, int]],
    frame_indices: List[int],
    fps: float = 30.0
) -> List[float]:
    """
    計算軌跡中每個點的速度

    參數：
    - trajectory: 軌跡座標
    - frame_indices: 對應的幀編號
    - fps: 影片幀率

    回傳：
    - 每個點的速度列表（pixels/second）
    """
    if len(trajectory) < 2:
        return [0.0] * len(trajectory)

    speeds = [0.0]

    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]

        # 計算距離
        distance = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)

        # 計算時間差
        if i < len(frame_indices):
            frame_diff = frame_indices[i] - frame_indices[i - 1]
            time_diff = frame_diff / fps
        else:
            time_diff = 1.0 / fps

        # 計算速度
        if time_diff > 0:
            speed = distance / time_diff
        else:
            speed = 0.0

        speeds.append(speed)

    return speeds


# ===========================================
# 如果直接執行此檔案，進行測試
# ===========================================
if __name__ == "__main__":
    print("軌跡追蹤模組測試")
    print("=" * 40)

    # 建立追蹤器
    tracker = BallTracker(max_missing_frames=2)

    # 模擬偵測序列
    detections = [
        (100, 100),   # 幀 0
        (110, 110),   # 幀 1
        None,         # 幀 2（掉幀）
        None,         # 幀 3（掉幀）
        (120, 120),   # 幀 4
        (130, 130),   # 幀 5
    ]

    # 更新追蹤
    for frame_num, detection in enumerate(detections):
        tracker.update(detection, frame_num)
        print(f"幀 {frame_num}: {detection} -> 追蹤中: {tracker.is_tracking()}")

    # 取得軌跡
    trajectory = tracker.get_trajectory()
    print(f"\n軌跡：{trajectory}")
    print(f"統計：{tracker.get_statistics()}")

    # 測試繪製
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output = tracker.draw_trajectory(test_frame)

    print("\n測試完成！")