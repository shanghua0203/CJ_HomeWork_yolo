"""
===========================================
雜訊過濾模組單元測試 (test_filter.py)
===========================================

測試項目：
1. 座標有效性檢查
2. 跳動過大檢測
3. 軌跡過濾
4. 掉幀插值
5. 平滑與離群值移除

作者：Python 影像辨識工程師
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from filter import (
    TrajectoryFilter,
    smooth_trajectory,
    remove_outliers
)


class TestTrajectoryFilter:
    """測試 TrajectoryFilter 類別"""

    def test_filter_init_default(self):
        """測試 1：使用預設參數建立過濾器"""
        filter_obj = TrajectoryFilter()

        assert filter_obj.frame_width == 640
        assert filter_obj.frame_height == 480
        assert filter_obj.max_jump == 100

    def test_filter_init_custom(self):
        """測試 2：使用自訂參數建立過濾器"""
        filter_obj = TrajectoryFilter(
            frame_width=1920,
            frame_height=1080,
            max_jump=50,
            max_missing_frames=3
        )

        assert filter_obj.frame_width == 1920
        assert filter_obj.frame_height == 1080
        assert filter_obj.max_jump == 50
        assert filter_obj.max_missing_frames == 3


class TestValidCoordinate:
    """測試座標有效性"""

    def test_valid_coordinate_inside(self):
        """測試 3：有效座標（在畫面內）"""
        filter_obj = TrajectoryFilter(frame_width=640, frame_height=480)

        assert filter_obj.is_valid_coordinate(100, 200) is True
        assert filter_obj.is_valid_coordinate(0, 0) is True
        assert filter_obj.is_valid_coordinate(639, 479) is True

    def test_valid_coordinate_outside(self):
        """測試 4：無效座標（超出畫面）"""
        filter_obj = TrajectoryFilter(frame_width=640, frame_height=480)

        assert filter_obj.is_valid_coordinate(-1, 100) is False
        assert filter_obj.is_valid_coordinate(640, 100) is False
        assert filter_obj.is_valid_coordinate(100, 480) is False

    def test_valid_coordinate_boundary(self):
        """測試 5：邊界情況"""
        filter_obj = TrajectoryFilter(frame_width=640, frame_height=480)

        assert filter_obj.is_valid_coordinate(0, 0) is True
        assert filter_obj.is_valid_coordinate(639, 479) is True


class TestJumpDetection:
    """測試跳動檢測"""

    def test_jump_small(self):
        """測試 6：小跳動（正常）"""
        filter_obj = TrajectoryFilter(max_jump=100)

        assert filter_obj.is_jump_too_large(100, 100, 110, 110) == False
        assert filter_obj.is_jump_too_large(100, 100, 150, 100) == False

    def test_jump_large(self):
        """測試 7：大跳動（異常）"""
        filter_obj = TrajectoryFilter(max_jump=100)

        assert filter_obj.is_jump_too_large(100, 100, 300, 300) == True
        assert filter_obj.is_jump_too_large(100, 100, 250, 100) == True

    def test_jump_no_previous(self):
        """測試 8：沒有上一個座標"""
        filter_obj = TrajectoryFilter(max_jump=100)

        assert filter_obj.is_jump_too_large(None, None, 100, 100) is False


class TestSingleCoordinate:
    """測試單座標過濾"""

    def test_filter_single_valid(self):
        """測試 9：過濾有效的單一座標"""
        filter_obj = TrajectoryFilter()

        result = filter_obj.filter_single_coordinate(100, 200)

        assert result == (100, 200)

    def test_filter_single_outside(self):
        """測試 10：過濾超出範圍的座標"""
        filter_obj = TrajectoryFilter(frame_width=640, frame_height=480)

        result = filter_obj.filter_single_coordinate(700, 200)

        assert result is None

    def test_filter_single_with_previous(self):
        """測試 11：過濾時考慮上一個座標"""
        filter_obj = TrajectoryFilter(max_jump=50)

        # 跳動太大，應該被過濾
        result = filter_obj.filter_single_coordinate(200, 200, 100, 100)

        assert result is None

        # 跳動正常，應該保留
        result = filter_obj.filter_single_coordinate(110, 110, 100, 100)

        assert result == (110, 110)


class TestTrajectoryFiltering:
    """測試軌跡過濾"""

    def test_filter_empty_trajectory(self):
        """測試 12：空軌跡"""
        filter_obj = TrajectoryFilter()

        result = filter_obj.filter_trajectory([])

        assert result == []

    def test_filter_normal_trajectory(self):
        """測試 13：正常軌跡（不需過濾）"""
        filter_obj = TrajectoryFilter(max_jump=100)

        trajectory = [(100, 100), (110, 110), (120, 120)]
        result = filter_obj.filter_trajectory(trajectory)

        assert result == [(100, 100), (110, 110), (120, 120)]

    def test_filter_trajectory_with_outliers(self):
        """測試 14：包含異常點的軌跡"""
        filter_obj = TrajectoryFilter(max_jump=50)

        trajectory = [
            (100, 100),
            (110, 110),
            (500, 500),  # 跳動太大，應該被濾除
            (120, 120)
        ]
        result = filter_obj.filter_trajectory(trajectory)

        assert result == [(100, 100), (110, 110), (120, 120)]
        assert len(result) == 3
        assert (500, 500) not in result

    def test_filter_trajectory_outside_bounds(self):
        """測試 15：超出邊界的座標"""
        filter_obj = TrajectoryFilter(frame_width=640, frame_height=480)

        trajectory = [
            (100, 100),
            (700, 100),  # 超出範圍
            (120, 120)
        ]
        result = filter_obj.filter_trajectory(trajectory)

        assert result == [(100, 100), (120, 120)]

    def test_filter_all_invalid(self):
        """測試 16：全部是無效座標"""
        filter_obj = TrajectoryFilter(frame_width=640, frame_height=480)

        trajectory = [(700, 100), (800, 200), (-10, 50)]
        result = filter_obj.filter_trajectory(trajectory)

        assert result == []


class TestInterpolation:
    """測試插值功能"""

    def test_interpolate_simple(self):
        """測試 17：簡單插值"""
        filter_obj = TrajectoryFilter()

        trajectory = [(100, 100), (None, None), (200, 200)]
        frame_indices = [0, 1, 2]

        result = filter_obj.interpolate_missing(trajectory, frame_indices)

        assert len(result) == 3
        assert result[0] == (100, 100)
        assert result[1] == (150, 150)
        assert result[2] == (200, 200)

    def test_interpolate_no_gap(self):
        """測試 18：無需插值（連續座標）"""
        filter_obj = TrajectoryFilter()

        trajectory = [(100, 100), (110, 110), (120, 120)]
        frame_indices = [0, 1, 2]

        result = filter_obj.interpolate_missing(trajectory, frame_indices)

        assert result == [(100, 100), (110, 110), (120, 120)]

    def test_interpolate_empty(self):
        """測試 19：空軌跡"""
        filter_obj = TrajectoryFilter()

        result = filter_obj.interpolate_missing([], [])

        assert result == []


class TestSmoothing:
    """測試平滑功能"""

    def test_smooth_normal(self):
        """測試 20：正常平滑"""
        trajectory = [(100, 100), (110, 110), (120, 120)]
        result = smooth_trajectory(trajectory, window_size=3)

        assert len(result) == 3

    def test_smooth_short_trajectory(self):
        """測試 21：短軌跡（少於視窗大小）"""
        trajectory = [(100, 100)]
        result = smooth_trajectory(trajectory, window_size=5)

        assert result == [(100, 100)]

    def test_smooth_odd_window(self):
        """測試 22：視窗大小自動調整為奇數"""
        trajectory = [(100, 100), (110, 110), (120, 120)]
        result = smooth_trajectory(trajectory, window_size=4)  # 會變成 5

        assert len(result) == 3


class TestOutlierRemoval:
    """測試離群值移除"""

    def test_remove_outliers_normal(self):
        """測試 23：正常情況"""
        trajectory = [(100, 100), (110, 110), (120, 120)]
        result = remove_outliers(trajectory, threshold=2.0)

        assert len(result) == 3

    def test_remove_outliers_with_gap(self):
        """測試 24：有離群值的情況"""
        trajectory = [
            (100, 100),
            (110, 110),
            (500, 500),  # 離群值
            (120, 120)
        ]
        result = remove_outliers(trajectory, threshold=1.5)

        assert len(result) < len(trajectory)

    def test_remove_outliers_too_short(self):
        """測試 25：軌跡太短不處理"""
        trajectory = [(100, 100), (110, 110)]
        result = remove_outliers(trajectory, threshold=2.0)

        assert result == [(100, 100), (110, 110)]


class TestEdgeCases:
    """測試邊界情況"""

    def test_filter_negative_coordinates(self):
        """測試 26：負座標"""
        filter_obj = TrajectoryFilter()

        result = filter_obj.filter_single_coordinate(-10, 50)

        assert result is None

    def test_filter_zero_coordinates(self):
        """測試 27：零座標（應該有效）"""
        filter_obj = TrajectoryFilter()

        result = filter_obj.filter_single_coordinate(0, 0)

        assert result == (0, 0)

    def test_large_trajectory(self):
        """測試 28：大量座標點"""
        filter_obj = TrajectoryFilter()

        # 生成 1000 個點
        trajectory = [(i, i) for i in range(1000)]
        result = filter_obj.filter_trajectory(trajectory)

        assert len(result) > 0


# ===========================================
# 主程式入口
# ===========================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])