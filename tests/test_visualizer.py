"""
===========================================
視覺化模組單元測試 (test_visualizer.py)
===========================================

測試項目：
1. 視覺化設定
2. 軌跡繪製
3. 落點繪製
4. 資訊面板
5. 鳥瞰圖產生
6. 多視圖組合

作者：Python 影像辨識工程師
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualizer import (
    VisualizationConfig,
    TrajectoryVisualizer,
    BirdEyeVisualizer,
    create_side_by_side_view,
    create_stacked_view
)
from perspective import PerspectiveTransformer, TableCorners
from landing_detector import LandingPoint


class TestVisualizationConfig:
    """測試視覺化設定"""

    def test_default_config(self):
        """測試 1：預設設定"""
        config = VisualizationConfig()

        assert config.trajectory_color == (0, 255, 0)
        assert config.trajectory_thickness == 2
        assert config.landing_color == (0, 0, 255)
        assert config.landing_radius == 10

    def test_custom_config(self):
        """測試 2：自訂設定"""
        config = VisualizationConfig(
            trajectory_color=(255, 0, 0),
            trajectory_thickness=3,
            landing_radius=20
        )

        assert config.trajectory_color == (255, 0, 0)
        assert config.trajectory_thickness == 3
        assert config.landing_radius == 20


class TestTrajectoryVisualizer:
    """測試軌跡視覺化器"""

    def test_init(self):
        """測試 3：初始化"""
        viz = TrajectoryVisualizer()

        assert viz.config is not None
        assert viz.output_width == 800
        assert viz.output_height == 600

    def test_draw_trajectory_empty(self):
        """測試 4：繪製空軌跡"""
        viz = TrajectoryVisualizer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = viz.draw_trajectory(frame, [])

        assert result.shape == frame.shape

    def test_draw_trajectory_single_point(self):
        """測試 5：單點軌跡"""
        viz = TrajectoryVisualizer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(100, 100)]

        result = viz.draw_trajectory(frame, trajectory)

        assert result.shape == frame.shape

    def test_draw_trajectory_multiple_points(self):
        """測試 6：多點軌跡"""
        viz = TrajectoryVisualizer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(100, 100), (200, 200), (300, 300)]

        result = viz.draw_trajectory(frame, trajectory)

        assert result.shape == frame.shape

    def test_draw_trajectory_disabled(self):
        """測試 7：停用軌跡繪製"""
        config = VisualizationConfig(show_trajectory=False)
        viz = TrajectoryVisualizer(config=config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(100, 100), (200, 200)]

        result = viz.draw_trajectory(frame, trajectory)

        # 應該與原圖相同
        assert np.array_equal(result, frame)


class TestLandingPointsDrawing:
    """測試落點繪製"""

    def test_draw_landing_points_empty(self):
        """測試 8：繪製空落點列表"""
        viz = TrajectoryVisualizer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = viz.draw_landing_points(frame, [])

        assert result.shape == frame.shape

    def test_draw_landing_points_single(self):
        """測試 9：單一落點"""
        viz = TrajectoryVisualizer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landing_points = [
            LandingPoint(
                x=320, y=400,
                frame_index=10,
                lowest_y=400,
                trajectory_segment=[]
            )
        ]

        result = viz.draw_landing_points(frame, landing_points)

        assert result.shape == frame.shape

    def test_draw_landing_points_multiple(self):
        """測試 10：多個落點"""
        viz = TrajectoryVisualizer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landing_points = [
            LandingPoint(x=320, y=400, frame_index=10, lowest_y=400, trajectory_segment=[]),
            LandingPoint(x=400, y=350, frame_index=30, lowest_y=350, trajectory_segment=[])
        ]

        result = viz.draw_landing_points(frame, landing_points)

        assert result.shape == frame.shape


class TestInfoPanel:
    """測試資訊面板"""

    def test_draw_info_panel(self):
        """測試 11：繪製資訊面板"""
        viz = TrajectoryVisualizer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(100, 100), (200, 200)]
        landing_points = [
            LandingPoint(x=320, y=400, frame_index=10, lowest_y=400, trajectory_segment=[])
        ]

        result = viz.draw_info_panel(frame, trajectory, landing_points, frame_number=5)

        assert result.shape == frame.shape

    def test_draw_info_panel_disabled(self):
        """測試 12：停用資訊面板"""
        config = VisualizationConfig(show_info_panel=False)
        viz = TrajectoryVisualizer(config=config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(100, 100)]
        landing_points = []

        result = viz.draw_info_panel(frame, trajectory, landing_points)

        assert result.shape == frame.shape


class TestBirdEyeVisualizer:
    """測試鳥瞰圖視覺化器"""

    def test_init(self):
        """測試 13：初始化"""
        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer = PerspectiveTransformer()
        transformer.set_corners(corners)

        viz = BirdEyeVisualizer(transformer=transformer)

        assert viz.transformer is not None

    def test_create_bird_eye_view_no_image(self):
        """測試 14：建立鳥瞰圖（無原圖）"""
        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer = PerspectiveTransformer()
        transformer.set_corners(corners)

        viz = BirdEyeVisualizer(transformer=transformer)
        bird_eye = viz.create_bird_eye_view()

        assert bird_eye.shape[0] == 600
        assert bird_eye.shape[1] == 800

    def test_create_bird_eye_view_with_image(self):
        """測試 15：建立鳥瞰圖（有空影像）"""
        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer = PerspectiveTransformer()
        transformer.set_corners(corners)

        viz = BirdEyeVisualizer(transformer=transformer)
        original = np.zeros((480, 640, 3), dtype=np.uint8)

        bird_eye = viz.create_bird_eye_view(original_frame=original)

        assert bird_eye.shape[0] == 600
        assert bird_eye.shape[1] == 800

    def test_draw_grid(self):
        """測試 16：繪製網格"""
        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer = PerspectiveTransformer()
        transformer.set_corners(corners)

        viz = BirdEyeVisualizer(transformer=transformer)
        bird_eye = viz.create_bird_eye_view()
        bird_eye = viz.draw_grid(bird_eye, grid_size=100)

        assert bird_eye.shape == (600, 800, 3)

    def test_draw_landing_points_empty(self):
        """測試 17：繪製空落點"""
        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer = PerspectiveTransformer()
        transformer.set_corners(corners)

        viz = BirdEyeVisualizer(transformer=transformer)
        bird_eye = viz.create_bird_eye_view()
        result = viz.draw_landing_points(bird_eye, [])

        assert result.shape == bird_eye.shape

    def test_draw_landing_points_single(self):
        """測試 18：繪製單一落點"""
        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer = PerspectiveTransformer()
        transformer.set_corners(corners)

        viz = BirdEyeVisualizer(transformer=transformer)
        bird_eye = viz.create_bird_eye_view()

        landing_points = [
            LandingPoint(x=320, y=400, frame_index=10, lowest_y=400, trajectory_segment=[])
        ]

        result = viz.draw_landing_points(bird_eye, landing_points)

        assert result.shape == bird_eye.shape


class TestMultiView:
    """測試多視圖組合"""

    def test_side_by_side(self):
        """測試 19：左右併排"""
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        bird_eye = np.zeros((600, 800, 3), dtype=np.uint8)

        result = create_side_by_side_view(original, bird_eye)

        # 寬度 = 原圖寬 + 鳥瞰圖寬
        assert result.shape[1] == 640 + 800
        # 高度 = 最大高度
        assert result.shape[0] == 600

    def test_side_by_side_with_labels(self):
        """測試 20：併排視圖（含標籤）"""
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        bird_eye = np.zeros((600, 800, 3), dtype=np.uint8)

        result = create_side_by_side_view(original, bird_eye, labels=True)

        assert result.shape[1] == 640 + 800

    def test_stacked_view(self):
        """測試 21：上下堆疊"""
        top = np.zeros((480, 640, 3), dtype=np.uint8)
        bottom = np.zeros((300, 400, 3), dtype=np.uint8)

        result = create_stacked_view(top, bottom)

        # 高度 = 兩個高度相加
        assert result.shape[0] == 480 + 300
        # 寬度 = 最大寬度
        assert result.shape[1] == 640


class TestEdgeCases:
    """測試邊界情況"""

    def test_draw_trajectory_all_disabled(self):
        """測試 22：全部停用"""
        config = VisualizationConfig(
            show_trajectory=False,
            show_info_panel=False,
            show_landing_points=False
        )
        viz = TrajectoryVisualizer(config=config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(100, 100), (200, 200)]
        landing_points = [
            LandingPoint(x=320, y=400, frame_index=10, lowest_y=400, trajectory_segment=[])
        ]

        result = viz.draw_trajectory(frame, trajectory)
        result = viz.draw_landing_points(result, landing_points)
        result = viz.draw_info_panel(result, trajectory, landing_points)

        assert result.shape == frame.shape

    def test_large_trajectory(self):
        """測試 23：大量軌跡點"""
        viz = TrajectoryVisualizer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(i * 10, i * 10) for i in range(100)]

        result = viz.draw_trajectory(frame, trajectory)

        assert result.shape == frame.shape


# ===========================================
# 主程式入口
# ===========================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])