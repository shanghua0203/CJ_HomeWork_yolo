"""
===========================================
系統測試 (test_system.py)
===========================================

端到端整合測試
測試整個分析流程的正確性

作者：Python 影像辨識工程師
"""

import pytest
import numpy as np
import os
import sys
import cv2
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import BallDetector
from filter import TrajectoryFilter, smooth_trajectory
from tracker import BallTracker, connect_trajectory_gaps
from landing_detector import LandingDetector, LandingPoint
from perspective import (
    PerspectiveTransformer,
    TableCorners,
    sort_four_points,
    auto_detect_table_corners
)
from visualizer import (
    TrajectoryVisualizer,
    BirdEyeVisualizer,
    VisualizationConfig,
    create_side_by_side_view
)


class TestIntegratedPipeline:
    """測試整合流程"""

    def test_detector_with_synthetic_image(self):
        """測試 1：偵測器處理合成影像"""
        detector = BallDetector(confidence=0.1)

        # 建立測試影像（隨機雜訊）
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 偵測不應該崩潰
        result = detector.detect(frame)

        # 結果應該是 None 或有效的座標
        assert result is None or isinstance(result, tuple)

    def test_filter_and_tracker_integration(self):
        """測試 2：過濾器與追蹤器整合"""
        # 建立過濾器與追蹤器
        filter_obj = TrajectoryFilter(max_jump=50)
        tracker = BallTracker(max_missing_frames=2)

        # 模擬偵測序列
        detections = [
            (100, 100),   # 幀 0
            (105, 105),   # 幀 1
            (110, 110),   # 幀 2
            None,         # 幀 3（掉幀）
            (115, 115),   # 幀 4
            (120, 120),   # 幀 5
            (500, 500),   # 幀 6（異常點）
            (125, 125),   # 幀 7
        ]

        # 更新追蹤器
        for i, detection in enumerate(detections):
            tracker.update(detection, i)

        # 取得軌跡
        trajectory = tracker.get_trajectory()

        # 過濾軌跡
        filtered = filter_obj.filter_trajectory(trajectory)

        # 驗證異常點被過濾
        assert len(filtered) < len(detections)
        assert (500, 500) not in filtered

    def test_landing_detection_integration(self):
        """測試 3：落點偵測整合"""
        # 建立落點偵測器
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        # 模擬乒乓球軌跡：從上往下，然後反弾
        trajectory = [
            (100, 50),
            (110, 100),
            (115, 150),
            (120, 200),
            (125, 180),  # 反弾點
            (130, 150),
        ]

        # 偵測落點
        landings = detector.detect(trajectory)

        # 應該偵測到至少一個落點
        assert len(landings) >= 1

    def test_perspective_transform_integration(self):
        """測試 4：透視變換整合"""
        # 建立透視變換器
        transformer = PerspectiveTransformer()

        # 設定角落
        corners = TableCorners(
            top_left=(100, 100),
            top_right=(500, 100),
            bottom_left=(100, 400),
            bottom_right=(500, 400)
        )
        transformer.set_corners(corners)

        # 測試點轉換
        original = (300, 250)
        transformed = transformer.transform_point(*original)
        back = transformer.inverse_transform_point(*transformed)

        # 轉換後應該能還原
        assert abs(back[0] - original[0]) < 1
        assert abs(back[1] - original[1]) < 1

    def test_full_analysis_flow(self):
        """測試 5：完整分析流程"""
        # 步驟 1：建立偵測器
        detector = BallDetector(confidence=0.1)

        # 步驟 2：建立過濾器
        filter_obj = TrajectoryFilter(max_jump=100)

        # 步驟 3：建立追蹤器
        tracker = BallTracker(max_missing_frames=2)

        # 步驟 4：建立落點偵測器
        landing_detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        # 步驟 5：建立透視變換器
        transformer = PerspectiveTransformer()
        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer.set_corners(corners)

        # 步驟 6：建立視覺化器
        viz = TrajectoryVisualizer()
        bird_eye_viz = BirdEyeVisualizer(transformer=transformer)

        # 模擬處理流程
        # 建立測試影像
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 偵測
        detection = detector.detect(frame)

        # 更新追蹤
        tracker.update(detection, 0)
        trajectory = tracker.get_trajectory()

        # 過濾
        filtered = filter_obj.filter_trajectory(trajectory)

        # 落點偵測
        if len(filtered) >= 3:
            landings = landing_detector.detect(filtered)
        else:
            landings = []

        # 視覺化
        output = viz.draw_trajectory(frame, filtered)
        output = viz.draw_landing_points(output, landings)

        # 鳥瞰圖
        bird_eye = bird_eye_viz.create_bird_eye_view()
        bird_eye = bird_eye_viz.draw_landing_points(bird_eye, landings)

        # 驗證流程完成
        assert output.shape == frame.shape
        assert bird_eye.shape[0] == 600
        assert bird_eye.shape[1] == 800


class TestEndToEndScenarios:
    """測試端到端情境"""

    def test_scenario_ball_bouncing(self):
        """測試 6：球反弾情境"""
        # 模擬球反弾的完整軌跡
        trajectory = [
            (320, 50),   # 起始點
            (315, 100),
            (310, 150),
            (305, 200),
            (300, 250),  # 最低點（接觸桌面）
            (295, 220),  # 反弾
            (290, 180),
            (285, 140),
            (280, 100),
            (275, 50),   # 到達高點
        ]

        # 建立偵測器
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=50
        )

        # 偵測落點
        landings = detector.detect(trajectory)

        # 應該偵測到落點
        assert len(landings) >= 1

        # 落點應該在最低點附近
        if landings:
            landing = landings[0]
            assert landing.y >= 200  # 落點 Y 應該較大（畫面下方）

    def test_scenario_multiple_bounces(self):
        """測試 7：多次反弾情境"""
        # 模擬多次反弾
        trajectory = [
            (320, 50),
            (310, 150),
            (300, 250),  # 第一次落點
            (290, 200),
            (280, 150),
            (270, 250),  # 第二次落點
            (260, 200),
            (250, 150),
            (240, 250),  # 第三次落點
            (230, 200),
        ]

        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=50
        )

        landings = detector.detect(trajectory)

        # 應該偵測到多個落點
        assert len(landings) >= 2

    def test_scenario_with_dropped_frames(self):
        """測試 8：帶掉幀的情境"""
        # 模擬掉幀
        detector = BallTracker(max_missing_frames=2)
        filter_obj = TrajectoryFilter(max_jump=50)

        detections = [
            (100, 100),
            (110, 110),
            None,  # 掉幀
            (120, 120),
            (130, 130),
        ]

        for i, det in enumerate(detections):
            detector.update(det, i)

        trajectory = detector.get_trajectory()
        filtered = filter_obj.filter_trajectory(trajectory)

        # 驗證追蹤正常
        assert len(trajectory) >= 3

    def test_scenario_trajectory_smoothing(self):
        """測試 9：軌跡平滑化情境"""
        # 有雜訊的軌跡
        noisy_trajectory = [
            (100, 100),
            (105, 105),
            (150, 110),  # 雜訊
            (120, 120),
            (125, 125),
        ]

        # 平滑化
        smoothed = smooth_trajectory(noisy_trajectory, window_size=3)

        # 驗證平滑化成功
        assert len(smoothed) == len(noisy_trajectory)

    def test_scenario_perspective_transform(self):
        """測試 10：透視變換情境"""
        transformer = PerspectiveTransformer()
        corners = TableCorners(
            top_left=(100, 100),
            top_right=(540, 100),
            bottom_left=(100, 380),
            bottom_right=(540, 380)
        )
        transformer.set_corners(corners)

        # 測試多點轉換
        points = [(320, 240), (200, 300), (400, 200)]
        transformed = transformer.transform_trajectory(points)

        assert len(transformed) == len(points)

        # 反轉換
        back_transformed = [
            transformer.inverse_transform_point(x, y)
            for x, y in transformed
        ]

        # 驗證還原正確（容許 2 pixels 誤差）
        for i, (orig, back) in enumerate(zip(points, back_transformed)):
            assert abs(orig[0] - back[0]) < 2
            assert abs(orig[1] - back[1]) < 2


class TestVisualizationIntegration:
    """測試視覺化整合"""

    def test_combined_view(self):
        """測試 11：合併視圖"""
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        bird_eye = np.zeros((600, 800, 3), dtype=np.uint8)

        combined = create_side_by_side_view(original, bird_eye)

        # 寬度應該是兩張圖相加
        assert combined.shape[1] == 640 + 800

    def test_trajectory_visualization(self):
        """測試 12：軌跡視覺化"""
        viz = TrajectoryVisualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        trajectory = [
            (100, 100),
            (110, 110),
            (120, 120),
        ]

        output = viz.draw_trajectory(frame, trajectory)

        assert output.shape == frame.shape

    def test_landing_visualization(self):
        """測試 13：落點視覺化"""
        viz = TrajectoryVisualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        landings = [
            LandingPoint(x=320, y=400, frame_index=10, lowest_y=400, trajectory_segment=[]),
            LandingPoint(x=400, y=350, frame_index=30, lowest_y=350, trajectory_segment=[])
        ]

        output = viz.draw_landing_points(frame, landings)

        assert output.shape == frame.shape


class TestErrorHandling:
    """測試錯誤處理"""

    def test_empty_trajectory_handling(self):
        """測試 14：空軌跡處理"""
        detector = LandingDetector()
        landings = detector.detect([])

        assert len(landings) == 0

    def test_single_point_trajectory(self):
        """測試 15：單點軌跡處理"""
        detector = LandingDetector()
        landings = detector.detect([(100, 100)])

        assert len(landings) == 0

    def test_invalid_transform_without_corners(self):
        """測試 16：沒有設定角落的轉換"""
        transformer = PerspectiveTransformer()

        # 應該回傳原座標
        result = transformer.transform_point(100, 100)
        assert result == (100, 100)


class TestPerformance:
    """測試效能相關"""

    def test_large_trajectory_processing(self):
        """測試 17：大量軌跡點處理"""
        # 建立 1000 個點的軌跡
        trajectory = [(i * 10 % 640, i * 10 % 480) for i in range(1000)]

        # 過濾
        filter_obj = TrajectoryFilter(max_jump=100)
        filtered = filter_obj.filter_trajectory(trajectory)

        # 落點偵測
        detector = LandingDetector()
        landings = detector.detect(filtered)

        # 應該能處理而不崩潰
        assert isinstance(landings, list)

    def test_rapid_frame_processing(self):
        """測試 18：快速幀處理"""
        tracker = BallTracker(max_missing_frames=2)

        # 模擬快速處理 100 幀
        for i in range(100):
            if i % 10 == 0:
                tracker.update(None, i)  # 偶爾掉幀
            else:
                tracker.update((i * 6 % 640, i * 5 % 480), i)

        trajectory = tracker.get_trajectory()
        assert len(trajectory) > 0


# ===========================================
# 主程式入口
# ===========================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])