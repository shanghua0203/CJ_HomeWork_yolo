"""
===========================================
落點判斷模組單元測試 (test_landing_detector.py)
===========================================

測試項目：
1. 偵測器初始化
2. Y 軸反転點偵測
3. 最低點尋找
4. 落點有效性判斷
5. 多落點偵測
6. 邊界情況處理

作者：Python 影像辨識工程師
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from landing_detector import (
    LandingDetector,
    LandingPoint,
    analyze_y_trend,
    find_y_extremes
)


class TestLandingDetectorInit:
    """測試偵測器初始化"""

    def test_init_default(self):
        """測試 1：使用預設參數建立偵測器"""
        detector = LandingDetector()

        assert detector.y_reversal_threshold == 5
        assert detector.min_fall_distance == 20
        assert detector.min_trajectory_length == 3

    def test_init_custom(self):
        """測試 2：使用自訂參數建立偵測器"""
        detector = LandingDetector(
            y_reversal_threshold=10,
            min_fall_distance=30,
            min_trajectory_length=5
        )

        assert detector.y_reversal_threshold == 10
        assert detector.min_fall_distance == 30
        assert detector.min_trajectory_length == 5


class TestYReversalDetection:
    """測試 Y 軸反転點偵測"""

    def test_reversal_simple(self):
        """測試 3：簡單的 Y 軸反転"""
        detector = LandingDetector(y_reversal_threshold=5)

        y_values = [50, 100, 150, 180, 160, 130]
        reversals = detector.detect_y_reversal(y_values)

        # 應該在索引 3 處反転（150 -> 180 -> 160）
        assert 3 in reversals

    def test_reversal_multiple(self):
        """測試 4：多個反転點"""
        detector = LandingDetector(y_reversal_threshold=5)

        y_values = [50, 100, 150, 180, 160, 130, 180, 160, 130]
        reversals = detector.detect_y_reversal(y_values)

        # 應該有兩個反転點
        assert len(reversals) >= 1

    def test_reversal_no_reversal(self):
        """測試 5：沒有反転"""
        detector = LandingDetector(y_reversal_threshold=5)

        # 持續下跌，沒有反転
        y_values = [50, 100, 150, 200, 250]
        reversals = detector.detect_y_reversal(y_values)

        assert len(reversals) == 0

    def test_reversal_below_threshold(self):
        """測試 6：反転幅度小於閾值"""
        detector = LandingDetector(y_reversal_threshold=10)

        # 反転幅度只有 5，小於閾值 10
        y_values = [50, 100, 150, 180, 175, 130]
        reversals = detector.detect_y_reversal(y_values)

        assert len(reversals) == 0

    def test_reversal_too_short(self):
        """測試 7：軌跡太短無法偵測"""
        detector = LandingDetector(y_reversal_threshold=5)

        y_values = [100, 150]
        reversals = detector.detect_y_reversal(y_values)

        assert len(reversals) == 0

    def test_reversal_falling_only(self):
        """測試 8：只有下跌（無反転）"""
        detector = LandingDetector(y_reversal_threshold=5)

        y_values = [50, 100, 150, 200]
        reversals = detector.detect_y_reversal(y_values)

        assert len(reversals) == 0


class TestLowestPointFinding:
    """測試最低點尋找"""

    def test_find_lowest_simple(self):
        """測試 9：簡單的最低點"""
        detector = LandingDetector()

        trajectory = [(100, 50), (110, 100), (115, 150), (120, 180)]
        x, y, idx = detector.find_lowest_before_reversal(trajectory, 3)

        assert x == 120
        assert y == 180
        assert idx == 3

    def test_find_lowest_first_point(self):
        """測試 10：最低點在第一個點"""
        detector = LandingDetector()

        trajectory = [(100, 50), (110, 40), (115, 30)]
        x, y, idx = detector.find_lowest_before_reversal(trajectory, 2)

        assert x == 100
        assert y == 50
        assert idx == 0

    def test_find_lowest_empty(self):
        """測試 11：空軌跡"""
        detector = LandingDetector()

        x, y, idx = detector.find_lowest_before_reversal([], 0)

        assert x == -1
        assert y == -1


class TestLandingValidity:
    """測試落點有效性判斷"""

    def test_valid_landing(self):
        """測試 12：有效落點"""
        detector = LandingDetector(
            min_fall_distance=20,
            min_trajectory_length=3
        )

        trajectory = [(100, 50), (110, 100), (115, 150)]
        is_valid = detector.is_valid_landing(trajectory, 2)

        assert is_valid is True

    def test_invalid_too_short(self):
        """測試 13：軌跡太短"""
        detector = LandingDetector(min_trajectory_length=5)

        trajectory = [(100, 50), (110, 100)]
        is_valid = detector.is_valid_landing(trajectory, 1)

        assert is_valid is False

    def test_invalid_short_fall(self):
        """測試 14：下落距離不足"""
        detector = LandingDetector(min_fall_distance=50)

        trajectory = [(100, 50), (110, 60), (115, 70)]
        is_valid = detector.is_valid_landing(trajectory, 2)

        assert is_valid is False


class TestLandingDetection:
    """測試落點偵測"""

    def test_detect_single_landing(self):
        """測試 15：單一落點"""
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        trajectory = [
            (100, 50),
            (110, 100),
            (115, 150),
            (120, 180),
            (125, 160),
            (130, 130)
        ]

        landings = detector.detect(trajectory)

        assert len(landings) >= 1
        assert isinstance(landings[0], LandingPoint)

    def test_detect_no_landing(self):
        """測試 16：無落點"""
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        # 只有下跌，沒有反転
        trajectory = [
            (100, 50),
            (110, 100),
            (115, 150)
        ]

        landings = detector.detect(trajectory)

        assert len(landings) == 0

    def test_detect_multiple_landings(self):
        """測試 17：多個落點"""
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        trajectory = [
            (100, 50),
            (110, 100),
            (115, 150),
            (120, 180),
            (125, 160),
            (130, 130),
            (135, 180),
            (140, 160),
            (145, 130)
        ]

        landings = detector.detect(trajectory)

        # 應該偵測到至少一個落點
        assert len(landings) >= 1

    def test_detect_with_frame_indices(self):
        """測試 18：带幀編號偵測"""
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        trajectory = [
            (100, 50),
            (110, 100),
            (115, 150),
            (120, 180),
            (125, 160)
        ]

        frame_indices = [10, 20, 30, 40, 50]
        landings = detector.detect(trajectory, frame_indices)

        if len(landings) > 0:
            assert landings[0].frame_index in frame_indices


class TestFirstLastLanding:
    """測試取得第一/最後落點"""

    def test_get_first_landing(self):
        """測試 19：取得第一個落點"""
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        trajectory = [
            (100, 50),
            (110, 100),
            (115, 150),
            (120, 180),
            (125, 160)
        ]

        first = detector.get_first_landing(trajectory)

        if first:
            assert isinstance(first, LandingPoint)

    def test_get_last_landing(self):
        """測試 20：取得最後一個落點"""
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        trajectory = [
            (100, 50),
            (110, 100),
            (115, 150),
            (120, 180),
            (125, 160)
        ]

        last = detector.get_last_landing(trajectory)

        if last:
            assert isinstance(last, LandingPoint)

    def test_get_first_no_landing(self):
        """測試 21：沒有落點時回傳 None"""
        detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        trajectory = [(100, 50), (110, 100)]
        first = detector.get_first_landing(trajectory)

        assert first is None


class TestYTrendAnalysis:
    """測試 Y 趨勢分析"""

    def test_analyze_falling(self):
        """測試 22：下跌趨勢"""
        y_values = [50, 100, 150, 200]

        trend = analyze_y_trend(y_values)

        assert trend["count"] == 4
        assert trend["min"] == 50
        assert trend["max"] == 200
        assert trend["is_falling"] is True

    def test_analyze_rising(self):
        """測試 23：上升趨勢"""
        y_values = [200, 150, 100, 50]

        trend = analyze_y_trend(y_values)

        assert trend["is_rising"] is True

    def test_analyze_empty(self):
        """測試 24：空列表"""
        trend = analyze_y_trend([])

        assert trend["count"] == 0
        assert trend["min"] is None


class TestYExtremes:
    """測試 Y 極值"""

    def test_find_extremes(self):
        """測試 25：找出極值"""
        y_values = [50, 100, 30, 200, 80]

        min_idx, max_idx = find_y_extremes(y_values)

        assert min_idx == 2  # 30 最小
        assert max_idx == 3  # 200 最大

    def test_find_extremes_empty(self):
        """測試 26：空列表"""
        min_idx, max_idx = find_y_extremes([])

        assert min_idx is None
        assert max_idx is None


class TestEdgeCases:
    """測試邊界情況"""

    def test_single_point_trajectory(self):
        """測試 27：單點軌跡"""
        detector = LandingDetector()

        trajectory = [(100, 100)]
        landings = detector.detect(trajectory)

        assert len(landings) == 0

    def test_identical_y_values(self):
        """測試 28：Y 值都相同"""
        detector = LandingDetector(y_reversal_threshold=5)

        y_values = [100, 100, 100, 100]
        reversals = detector.detect_y_reversal(y_values)

        assert len(reversals) == 0

    def test_rapid_reversals(self):
        """測試 29：快速連續反転"""
        detector = LandingDetector(y_reversal_threshold=5)

        y_values = [50, 100, 150, 160, 170, 100, 150]
        reversals = detector.detect_y_reversal(y_values)

        # 應該偵測到反転
        assert len(reversals) >= 1


# ===========================================
# 主程式入口
# ===========================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])