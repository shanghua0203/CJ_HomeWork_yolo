"""
===========================================
軌跡追蹤模組單元測試 (test_tracker.py)
===========================================

測試項目：
1. 追蹤器初始化
2. 基本追蹤功能
3. 掉幀處理與插值
4. 軌跡繪製
5. 統計資訊計算

作者：Python 影像辨識工程師
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tracker import (
    BallTracker,
    TrackingState,
    connect_trajectory_gaps,
    calculate_trajectory_speed
)


class TestBallTrackerInit:
    """測試追蹤器初始化"""

    def test_init_default(self):
        """測試 1：使用預設參數建立追蹤器"""
        tracker = BallTracker()

        assert tracker.max_missing_frames == 2
        assert tracker.min_trajectory_length == 2
        assert tracker.is_tracking() is False

    def test_init_custom(self):
        """測試 2：使用自訂參數建立追蹤器"""
        tracker = BallTracker(
            max_missing_frames=5,
            min_trajectory_length=3,
            trajectory_color=(255, 0, 0),
            trajectory_thickness=3
        )

        assert tracker.max_missing_frames == 5
        assert tracker.min_trajectory_length == 3
        assert tracker.trajectory_color == (255, 0, 0)


class TestTrackingBasic:
    """測試基本追蹤功能"""

    def test_update_with_detection(self):
        """測試 3：更新並傳入有效偵測"""
        tracker = BallTracker()

        result = tracker.update((100, 100), 0)

        assert result is True
        assert tracker.is_tracking() is True
        assert len(tracker.get_trajectory()) == 1
        assert tracker.get_trajectory()[0] == (100, 100)

    def test_update_without_detection(self):
        """測試 4：更新並傳入無偵測"""
        tracker = BallTracker()

        tracker.update((100, 100), 0)
        result = tracker.update(None, 1)

        assert result is True
        assert tracker.is_tracking() is True
        assert tracker.state.missing_frames == 1
        assert len(tracker.state.trajectory) == 2

    def test_update_multiple_frames(self):
        """測試 5：多幀追蹤"""
        tracker = BallTracker()

        tracker.update((100, 100), 0)
        tracker.update((110, 110), 1)
        tracker.update((120, 120), 2)

        trajectory = tracker.get_trajectory()

        assert len(trajectory) == 3
        assert trajectory == [(100, 100), (110, 110), (120, 120)]

    def test_reset_tracker(self):
        """測試 6：重置追蹤器"""
        tracker = BallTracker()

        tracker.update((100, 100), 0)
        tracker.update((110, 110), 1)
        tracker.reset()

        assert len(tracker.get_trajectory()) == 0
        assert tracker.is_tracking() is False


class TestFrameIndices:
    """測試幀編號記錄"""

    def test_frame_indices_tracking(self):
        """測試 7：追蹤時記錄幀編號"""
        tracker = BallTracker()

        tracker.update((100, 100), 5)
        tracker.update((110, 110), 10)
        tracker.update((120, 120), 15)

        indices = tracker.get_frame_indices()

        assert indices == [5, 10, 15]


class TestDroppedFrames:
    """測試掉幀處理"""

    def test_dropped_frames_within_limit(self):
        """測試 8：掉幀在容忍範圍內"""
        tracker = BallTracker(max_missing_frames=2)

        tracker.update((100, 100), 0)
        tracker.update(None, 1)
        tracker.update(None, 2)

        assert tracker.is_tracking() is True
        assert tracker.state.missing_frames == 2

    def test_dropped_frames_exceeds_limit(self):
        """測試 9：掉幀超過容忍範圍"""
        tracker = BallTracker(max_missing_frames=2)

        tracker.update((100, 100), 0)
        tracker.update(None, 1)
        tracker.update(None, 2)
        tracker.update(None, 3)

        assert tracker.is_tracking() is False

    def test_recovery_after_drop(self):
        """測試 10：掉幀後恢復追蹤"""
        tracker = BallTracker(max_missing_frames=2)

        tracker.update((100, 100), 0)
        tracker.update(None, 1)
        tracker.update(None, 2)
        result = tracker.update((110, 110), 3)

        assert result is True
        assert tracker.is_tracking() is True
        assert tracker.state.missing_frames == 0

    def test_interpolate_dropped_frames(self):
        """測試 11：掉幀插值"""
        tracker = BallTracker(max_missing_frames=2)

        tracker.update((100, 100), 0)
        tracker.update(None, 1)

        predictions = tracker.interpolate_dropped_frames(None, 1)

        assert len(predictions) == 1
        assert predictions[0] == (100, 100)


class TestStatistics:
    """測試統計資訊"""

    def test_statistics_empty(self):
        """測試 12：空軌跡統計"""
        tracker = BallTracker()

        stats = tracker.get_statistics()

        assert stats["total_points"] == 0
        assert stats["is_tracking"] is False

    def test_statistics_single_point(self):
        """測試 13：單點軌跡統計"""
        tracker = BallTracker()

        tracker.update((100, 100), 0)

        stats = tracker.get_statistics()

        assert stats["total_points"] == 1
        assert stats["min_y"] == 100
        assert stats["max_y"] == 100

    def test_statistics_multiple_points(self):
        """測試 14：多點軌跡統計"""
        tracker = BallTracker()

        tracker.update((100, 100), 0)
        tracker.update((110, 120), 1)
        tracker.update((120, 140), 2)

        stats = tracker.get_statistics()

        assert stats["total_points"] == 3
        assert stats["min_y"] == 100
        assert stats["max_y"] == 140
        assert stats["average_speed"] > 0


class TestDrawing:
    """測試繪製功能"""

    def test_draw_trajectory_empty(self):
        """測試 15：空軌跡繪製"""
        tracker = BallTracker()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = tracker.draw_trajectory(frame)

        assert output.shape == frame.shape
        assert np.array_equal(output, frame)

    def test_draw_trajectory_single_point(self):
        """測試 16：單點軌跡繪製"""
        tracker = BallTracker()

        tracker.update((100, 100), 0)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = tracker.draw_trajectory(frame)

        assert output.shape == frame.shape

    def test_draw_trajectory_multiple_points(self):
        """測試 17：多點軌跡繪製"""
        tracker = BallTracker()

        tracker.update((100, 100), 0)
        tracker.update((110, 110), 1)
        tracker.update((120, 120), 2)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = tracker.draw_trajectory(frame)

        assert output.shape == frame.shape


class TestConnectGaps:
    """測試連接斷點功能"""

    def test_connect_no_gap(self):
        """測試 18：無需連接的軌跡"""
        trajectory = [(100, 100), (110, 110), (120, 120)]
        indices = [0, 1, 2]

        result_traj, result_idx = connect_trajectory_gaps(trajectory, indices)

        assert result_traj == trajectory
        assert result_idx == indices

    def test_connect_with_small_gap(self):
        """測試 19：連接小間隔"""
        trajectory = [(100, 100), (120, 120)]
        indices = [0, 2]

        result_traj, result_idx = connect_trajectory_gaps(trajectory, indices, max_gap=2)

        # 會插值幀 1
        assert len(result_traj) == 3
        assert len(result_idx) == 3

    def test_connect_with_large_gap(self):
        """測試 20：大間隔不連接"""
        trajectory = [(100, 100), (120, 120)]
        indices = [0, 5]

        result_traj, result_idx = connect_trajectory_gaps(trajectory, indices, max_gap=2)

        # 不會插值
        assert result_traj == trajectory
        assert result_idx == indices


class TestSpeedCalculation:
    """測試速度計算"""

    def test_speed_empty_trajectory(self):
        """測試 21：空軌跡"""
        speeds = calculate_trajectory_speed([], [], fps=30.0)

        assert speeds == []

    def test_speed_single_point(self):
        """測試 22：單點軌跡"""
        speeds = calculate_trajectory_speed([(100, 100)], [0], fps=30.0)

        assert speeds == [0.0]

    def test_speed_multiple_points(self):
        """測試 23：多點軌跡"""
        trajectory = [(0, 0), (30, 0), (60, 0)]
        indices = [0, 1, 2]

        speeds = calculate_trajectory_speed(trajectory, indices, fps=30.0)

        assert len(speeds) == 3
        assert speeds[0] == 0.0  # 第一點沒有前一個點
        assert speeds[1] > 0  # 有速度
        assert speeds[2] > 0


class TestEdgeCases:
    """測試邊界情況"""

    def test_update_negative_frame(self):
        """測試 24：負幀編號"""
        tracker = BallTracker()

        tracker.update((100, 100), -1)

        assert len(tracker.get_trajectory()) == 1
        assert tracker.get_frame_indices()[0] == -1

    def test_large_trajectory(self):
        """測試 25：長軌跡"""
        tracker = BallTracker()

        for i in range(100):
            tracker.update((i * 10, i * 10), i)

        assert len(tracker.get_trajectory()) == 100

    def test_alternating_detection(self):
        """測試 26：交替出現/消失"""
        tracker = BallTracker(max_missing_frames=2)

        for i in range(20):
            if i % 3 == 0:
                tracker.update(None, i)
            else:
                tracker.update((i * 10, i * 10), i)

        stats = tracker.get_statistics()
        assert stats["total_points"] > 0


# ===========================================
# 主程式入口
# ===========================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])