"""
===========================================
視覺化輸出模組 (visualizer.py)
===========================================

整合所有視覺化輸出功能：軌跡、落點、2D 鳥瞰圖

主要功能：
1. 在原影片上繪製軌跡線與落點標記
2. 產生 2D 鳥瞰圖並標示落點
3. 輸出結果影片或圖片

作者：Python 影像辨識工程師
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from tracker import BallTracker
from landing_detector import LandingPoint, LandingDetector
from perspective import PerspectiveTransformer, TableCorners


@dataclass
class VisualizationConfig:
    """
    視覺化設定檔
    儲存所有視覺化相關的設定參數
    """
    # 軌跡線顏色 (B, G, R)
    trajectory_color: Tuple[int, int, int] = (0, 255, 0)

    # 軌跡線粗細
    trajectory_thickness: int = 2

    # 落點顏色
    landing_color: Tuple[int, int, int] = (0, 0, 255)

    # 落點半徑
    landing_radius: int = 10

    # 當前球位置顏色
    ball_color: Tuple[int, int, int] = (255, 0, 255)

    # 當前球半徑
    ball_radius: int = 8

    # 文字顏色
    text_color: Tuple[int, int, int] = (255, 255, 255)

    # 文字大小
    text_scale: float = 0.7

    # 是否顯示資訊面板
    show_info_panel: bool = True

    # 是否顯示軌跡
    show_trajectory: bool = True

    # 是否顯示落點
    show_landing_points: bool = True

    # 是否顯示偵測框（預設開啟）
    show_detection_box: bool = True

    # 偵測框顏色 (B, G, R)
    detection_box_color: Tuple[int, int, int] = (0, 255, 0)

    # 偵測框粗細
    detection_box_thickness: int = 2


class TrajectoryVisualizer:
    """
    軌跡視覺化器
    在影片上繪製軌跡線與落點
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        output_width: int = 800,
        output_height: int = 600
    ):
        """
        初始化視覺化器

        參數：
        - config: 視覺化設定
        - output_width: 鳥瞰圖輸出寬度
        - output_height: 鳥瞰圖輸出高度
        """
        self.config = config or VisualizationConfig()
        self.output_width = output_width
        self.output_height = output_height

    def draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[int, int]],
        start_index: int = 0
    ) -> np.ndarray:
        """
        在影像上繪製軌跡線

        參數：
        - frame: 原始影像
        - trajectory: 軌跡座標列表
        - start_index: 開始繪製的索引

        回傳：
        - 繪製後的影像
        """
        if not self.config.show_trajectory:
            return frame

        if len(trajectory) < 2:
            return frame

        output = frame.copy()

        # 繪製軌跡線
        for i in range(start_index + 1, len(trajectory)):
            pt1 = trajectory[i - 1]
            pt2 = trajectory[i]

            cv2.line(
                output,
                pt1,
                pt2,
                self.config.trajectory_color,
                self.config.trajectory_thickness
            )

        # 繪製軌跡點
        for i, (x, y) in enumerate(trajectory[start_index:]):
            if i == len(trajectory[start_index:]) - 1:
                # 最後一個點（當前位置）
                cv2.circle(
                    output,
                    (x, y),
                    self.config.ball_radius + 2,
                    self.config.ball_color,
                    -1
                )
            else:
                cv2.circle(output, (x, y), 3, self.config.trajectory_color, -1)

        return output

    def draw_detection_box(
        self,
        frame: np.ndarray,
        x1: int, y1: int,
        x2: int, y2: int,
        confidence: float = 1.0,
        label: str = "ball"
    ) -> np.ndarray:
        """
        在影像上繪製偵測框

        參數：
        - frame: 原始影像
        - x1, y1, x2, y2: 偵測框座標
        - confidence: 信心值
        - label: 標籤名稱

        回傳：
        - 繪製後的影像
        """
        if not self.config.show_detection_box:
            return frame

        output = frame.copy()

        # 繪製偵測框
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            self.config.detection_box_color,
            self.config.detection_box_thickness
        )

        # 繪製信心值文字
        text = f"{label}: {confidence:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # 文字背景
        cv2.rectangle(
            output,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            self.config.detection_box_color,
            -1
        )

        # 文字內容
        cv2.putText(
            output,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return output

    def draw_detection_boxes(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int, int, float, str]]
    ) -> np.ndarray:
        """
        在影像上繪製多個偵測框

        參數：
        - frame: 原始影像
        - detections: [(center_x, center_y, x1, y1, x2, y2, conf, class_name), ...]

        回傳：
        - 繪製後的影像
        """
        output = frame.copy()

        for det in detections:
            center_x, center_y, x1, y1, x2, y2, conf, class_name = det
            output = self.draw_detection_box(
                output, x1, y1, x2, y2, conf, class_name
            )

        return output

    def draw_landing_points(
        self,
        frame: np.ndarray,
        landing_points: List[LandingPoint],
        transform: Optional[PerspectiveTransformer] = None
    ) -> np.ndarray:
        """
        在影像上繪製落點

        參數：
        - frame: 原始影像
        - landing_points: 落點列表
        - transform: 透視變換器（可選，用於顯示鳥瞰圖落點）

        回傳：
        - 繪製後的影像
        """
        if not self.config.show_landing_points:
            return frame

        output = frame.copy()

        for i, landing in enumerate(landing_points):
            x, y = landing.x, landing.y

            # 繪製落點圓圈
            cv2.circle(
                output,
                (x, y),
                self.config.landing_radius,
                self.config.landing_color,
                2
            )

            # 繪製中心點
            cv2.circle(
                output,
                (x, y),
                3,
                self.config.landing_color,
                -1
            )

            # 繪製編號
            cv2.putText(
                output,
                f"L{i + 1}",
                (x + 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale,
                self.config.landing_color,
                2
            )

        return output

    def draw_info_panel(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[int, int]],
        landing_points: List[LandingPoint],
        frame_number: int = 0
    ) -> np.ndarray:
        """
        繪製資訊面板

        參數：
        - frame: 原始影像
        - trajectory: 軌跡座標
        - landing_points: 落點列表
        - frame_number: 當前幀編號

        回傳：
        - 繪製後的影像
        """
        if not self.config.show_info_panel:
            return frame

        output = frame.copy()
        h, w = frame.shape[:2]

        # 建立半透明面板
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)

        # 混合面板
        alpha = 0.7
        output[h - panel_height:h] = cv2.addWeighted(
            output[h - panel_height:h],
            alpha,
            panel,
            1 - alpha,
            0
        )

        # 繪製文字
        y_offset = h - panel_height + 30

        info_lines = [
            f"Frame: {frame_number}",
            f"Trajectory Points: {len(trajectory)}",
            f"Landing Points: {len(landing_points)}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                output,
                line,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.config.text_color,
                1
            )

        # 顯示當前球位置
        if trajectory:
            last_point = trajectory[-1]
            cv2.putText(
                output,
                f"Ball Position: ({last_point[0]}, {last_point[1]})",
                (10, y_offset + len(info_lines) * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.config.text_color,
                1
            )

        return output


class BirdEyeVisualizer:
    """
    鳥瞰圖視覺化器
    在 2D 鳥瞰圖上標示落點
    """

    def __init__(
        self,
        transformer: PerspectiveTransformer,
        table_color: Tuple[int, int, int] = (100, 100, 100),
        landing_color: Tuple[int, int, int] = (0, 0, 255),
        grid_color: Tuple[int, int, int] = (50, 50, 50)
    ):
        """
        初始化鳥瞰圖視覺化器

        參數：
        - transformer: 透視變換器
        - table_color: 桌面顏色
        - landing_color: 落點顏色
        - grid_color: 網格顏色
        """
        self.transformer = transformer
        self.table_color = table_color
        self.landing_color = landing_color
        self.grid_color = grid_color

    def create_bird_eye_view(
        self,
        original_frame: Optional[np.ndarray] = None,
        background_color: Tuple[int, int, int] = (200, 200, 200)
    ) -> np.ndarray:
        """
        建立鳥瞰圖底圖

        參數：
        - original_frame: 原始影像（用於透視變換）
        - background_color: 背景顏色

        回傳：
        - 鳥瞰圖影像
        """
        w = self.transformer.output_width
        h = self.transformer.output_height

        if original_frame is not None:
            bird_eye = self.transformer.transform_image(original_frame)
        else:
            bird_eye = np.ones((h, w, 3), dtype=np.uint8)
            bird_eye[:] = background_color

        return bird_eye

    def draw_grid(
        self,
        bird_eye: np.ndarray,
        grid_size: int = 100
    ) -> np.ndarray:
        """
        在鳥瞰圖上繪製網格

        參數：
        - bird_eye: 鳥瞰圖
        - grid_size: 網格大小（pixels）

        回傳：
        - 繪製後的鳥瞰圖
        """
        output = bird_eye.copy()
        h, w = bird_eye.shape[:2]

        # 繪製垂直線
        for x in range(0, w, grid_size):
            cv2.line(output, (x, 0), (x, h), self.grid_color, 1)

        # 繪製水平線
        for y in range(0, h, grid_size):
            cv2.line(output, (0, y), (w, y), self.grid_color, 1)

        return output

    def draw_landing_points(
        self,
        bird_eye: np.ndarray,
        landing_points: List[LandingPoint],
        radius: int = 15
    ) -> np.ndarray:
        """
        在鳥瞰圖上繪製落點

        參數：
        - bird_eye: 鳥瞰圖
        - landing_points: 落點列表
        - radius: 落點半徑

        回傳：
        - 繪製後的鳥瞰圖
        """
        output = bird_eye.copy()
        w = self.transformer.output_width
        h = self.transformer.output_height

        for i, landing in enumerate(landing_points):
            # 將原視角座標轉換為鳥瞰圖座標
            bx, by = self.transformer.transform_point(landing.x, landing.y)

            # 檢查是否在鳥瞰圖範圍內
            if 0 <= bx < w and 0 <= by < h:
                # 繪製大圓圈
                cv2.circle(output, (bx, by), radius, self.landing_color, 2)

                # 繪製中心點
                cv2.circle(output, (bx, by), 3, self.landing_color, -1)

                # 繪製編號
                cv2.putText(
                    output,
                    f"L{i + 1}",
                    (bx + radius + 5, by),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.landing_color,
                    2
                )

        return output


def create_side_by_side_view(
    original: np.ndarray,
    bird_eye: np.ndarray,
    labels: bool = True
) -> np.ndarray:
    """
    建立左右併排視圖

    參數：
    - original: 原視角影像
    - bird_eye: 鳥瞰圖
    - labels: 是否顯示標籤

    回傳：
    - 併排後的影像
    """
    h = max(original.shape[0], bird_eye.shape[0])
    w = original.shape[1] + bird_eye.shape[1]

    # 建立空白畫布
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 放置原圖
    oh, ow = original.shape[:2]
    canvas[:oh, :ow] = original

    # 放置鳥瞰圖
    bh, bw = bird_eye.shape[:2]
    canvas[:bh, ow:ow + bw] = bird_eye

    # 添加標籤
    if labels:
        cv2.putText(
            canvas,
            "Original View",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            "Bird's Eye View",
            (ow + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    return canvas


def create_stacked_view(
    top: np.ndarray,
    bottom: np.ndarray
) -> np.ndarray:
    """
    建立上下堆疊視圖

    參數：
    - top: 上方影像（原視角）
    - bottom: 下方影像（鳥瞰圖）

    回傳：
    - 堆疊後的影像
    """
    w = max(top.shape[1], bottom.shape[1])
    h = top.shape[0] + bottom.shape[0]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 放置原圖（上）
    tw = top.shape[1]
    canvas[:top.shape[0], :tw] = top

    # 放置鳥瞰圖（下）
    bw = bottom.shape[1]
    canvas[top.shape[0]:, :bw] = bottom

    return canvas


# ===========================================
# 如果直接執行此檔案，進行測試
# ===========================================
if __name__ == "__main__":
    print("視覺化輸出模組測試")
    print("=" * 40)

    # 建立設定
    config = VisualizationConfig()

    print(f"軌跡顏色：{config.trajectory_color}")
    print(f"落點顏色：{config.landing_color}")

    # 建立視覺化器
    visualizer = TrajectoryVisualizer(config=config)

    # 建立鳥瞰圖視覺化器（需要先設定轉換器）
    from perspective import TableCorners

    corners = TableCorners(
        top_left=(100, 100),
        top_right=(500, 100),
        bottom_left=(100, 400),
        bottom_right=(500, 400)
    )

    transformer = PerspectiveTransformer()
    transformer.set_corners(corners)

    bird_eye_viz = BirdEyeVisualizer(transformer=transformer)

    # 建立測試鳥瞰圖
    bird_eye = bird_eye_viz.create_bird_eye_view()
    bird_eye = bird_eye_viz.draw_grid(bird_eye)

    print(f"\n鳥瞰圖尺寸：{bird_eye.shape}")

    print("\n測試完成！")