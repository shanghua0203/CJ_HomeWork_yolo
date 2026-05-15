"""
===========================================
系統測試 (test_system.py)
===========================================

端到端整合測試
測試整個分析流程的正確性
"""

import pytest
import numpy as np
import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import BallDetector
from filter import TrajectoryFilter, smooth_trajectory
from tracker import BallTracker
from landing_detector import LandingDetector, LandingPoint
from perspective import PerspectiveTransformer, TableCorners
from visualizer import TrajectoryVisualizer, BirdEyeVisualizer, create_side_by_side_view


class TestIntegratedPipeline:
    """測試整合流程"""

    def test_detector_with_synthetic_image(self):
        """測試 1：偵測器處理合成影像"""
        detector = BallDetector(confidence=0.1)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result is None or isinstance(result, tuple)

    def test_filter_and_tracker_integration(self):
        """測試 2：過濾器與追蹤器整合"""
        filter_obj = TrajectoryFilter(max_jump=50)
        tracker = BallTracker(max_missing_frames=2)

        detections = [(100, 100), (105, 105), (110, 110), None,
                      (115, 115), (120, 120), (500, 500), (125, 125)]

        for i, detection in enumerate(detections):
            tracker.update(detection, i)

        trajectory = tracker.get_trajectory()
        filtered = filter_obj.filter_trajectory(trajectory)
        assert len(filtered) < len(detections)
        assert (500, 500) not in filtered

    def test_landing_detection_integration(self):
        """測試 3：落點偵測整合"""
        detector = LandingDetector(y_reversal_threshold=5, min_fall_distance=20)

        trajectory = [(100, 50), (110, 100), (115, 150), (120, 200), (125, 180), (130, 150)]
        landings = detector.detect(trajectory)
        assert len(landings) >= 1

    def test_perspective_transform_integration(self):
        """測試 4：透視變換整合"""
        transformer = PerspectiveTransformer()
        corners = TableCorners(top_left=(100, 100), top_right=(500, 100),
                            bottom_left=(100, 400), bottom_right=(500, 400))
        transformer.set_corners(corners)

        original = (300, 250)
        transformed = transformer.transform_point(*original)
        assert 0 <= transformed[0] < transformer.output_width
        assert 0 <= transformed[1] < transformer.output_height

    def test_full_analysis_flow(self):
        """測試 5：完整分析流程"""
        detector = BallDetector(confidence=0.1)
        filter_obj = TrajectoryFilter(max_jump=100)
        tracker = BallTracker(max_missing_frames=2)
        landing_detector = LandingDetector(y_reversal_threshold=5, min_fall_distance=20)

        transformer = PerspectiveTransformer()
        corners = TableCorners(top_left=(0, 0), top_right=(640, 0),
                            bottom_left=(0, 480), bottom_right=(640, 480))
        transformer.set_corners(corners)

        viz = TrajectoryVisualizer()
        bird_eye_viz = BirdEyeVisualizer(transformer=transformer)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection = detector.detect(frame)
        tracker.update(detection, 0)
        trajectory = tracker.get_trajectory()
        filtered = filter_obj.filter_trajectory(trajectory)

        if len(filtered) >= 3:
            landings = landing_detector.detect(filtered)
        else:
            landings = []

        output = viz.draw_trajectory(frame, filtered)
        output = viz.draw_landing_points(output, landings)

        bird_eye = bird_eye_viz.create_bird_eye_view()
        bird_eye = bird_eye_viz.draw_landing_points(bird_eye, landings)

        assert output.shape == frame.shape
        assert bird_eye.shape[0] == 600
        assert bird_eye.shape[1] == 800


class TestEndToEndScenarios:
    """測試端到端情境"""

    def test_scenario_ball_bouncing(self):
        """測試 6：球反呑情境"""
        trajectory = [(320, 50), (315, 100), (310, 150), (305, 200), (300, 250),
                     (295, 220), (290, 180), (285, 140), (280, 100), (275, 50)]

        detector = LandingDetector(y_reversal_threshold=5, min_fall_distance=50)
        landings = detector.detect(trajectory)

        assert len(landings) >= 1
        if landings:
            landing = landings[0]
            assert landing.y >= 200

    def test_scenario_multiple_bounces(self):
        """測試 7：多次反呑情境"""
        trajectory = [(320, 50), (310, 150), (300, 250), (290, 200), (280, 150),
                     (270, 250), (260, 200), (250, 150), (240, 250), (230, 200)]

        detector = LandingDetector(y_reversal_threshold=5, min_fall_distance=50)
        landings = detector.detect(trajectory)
        assert len(landings) >= 2

    def test_scenario_with_dropped_frames(self):
        """測試 8：帶掉幀的情境"""
        tracker = BallTracker(max_missing_frames=2)
        filter_obj = TrajectoryFilter(max_jump=50)

        detections = [(100, 100), (110, 110), None, (120, 120), (130, 130)]
        for i, det in enumerate(detections):
            tracker.update(det, i)

        trajectory = tracker.get_trajectory()
        filtered = filter_obj.filter_trajectory(trajectory)
        assert len(trajectory) >= 3

    def test_scenario_trajectory_smoothing(self):
        """測試 9：軌跡平滑化情境"""
        noisy_trajectory = [(100, 100), (105, 105), (150, 110), (120, 120), (125, 125)]
        smoothed = smooth_trajectory(noisy_trajectory, window_size=3)
        assert len(smoothed) == len(noisy_trajectory)

    def test_scenario_perspective_transform(self):
        """測試 10：透視變換情境"""
        transformer = PerspectiveTransformer()
        corners = TableCorners(top_left=(100, 100), top_right=(540, 100),
                            bottom_left=(100, 380), bottom_right=(540, 380))
        transformer.set_corners(corners)

        transformed = transformer.transform_point(320, 240)
        assert 0 <= transformed[0] < transformer.output_width
        assert 0 <= transformed[1] < transformer.output_height


class TestVisualizationIntegration:
    """測試視覺化整合"""

    def test_combined_view(self):
        """測試 11：合併視圖"""
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        bird_eye = np.zeros((600, 800, 3), dtype=np.uint8)
        combined = create_side_by_side_view(original, bird_eye)
        assert combined.shape[1] == 640 + 800

    def test_trajectory_visualization(self):
        """測試 12：軌跡視覺化"""
        viz = TrajectoryVisualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(100, 100), (110, 110), (120, 120)]
        output = viz.draw_trajectory(frame, trajectory)
        assert output.shape == frame.shape

    def test_landing_visualization(self):
        """測試 13：落點視覺化"""
        viz = TrajectoryVisualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landings = [LandingPoint(x=320, y=400, frame_index=10, lowest_y=400, trajectory_segment=[]),
                    LandingPoint(x=400, y=350, frame_index=30, lowest_y=350, trajectory_segment=[])]
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
        result = transformer.transform_point(100, 100)
        assert result == (100, 100)


class TestPerformance:
    """測試效能相關"""

    def test_large_trajectory_processing(self):
        """測試 17：大量軌跡點處理"""
        trajectory = [(i * 10 % 640, i * 10 % 480) for i in range(1000)]
        filter_obj = TrajectoryFilter(max_jump=100)
        filtered = filter_obj.filter_trajectory(trajectory)
        detector = LandingDetector()
        landings = detector.detect(filtered)
        assert isinstance(landings, list)

    def test_rapid_frame_processing(self):
        """測試 18：快速幀處理"""
        tracker = BallTracker(max_missing_frames=2)
        for i in range(100):
            if i % 10 == 0:
                tracker.update(None, i)
            else:
                tracker.update((i * 6 % 640, i * 5 % 480), i)
        trajectory = tracker.get_trajectory()
        assert len(trajectory) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
