"""
===========================================
視角轉換模組單元測試 (test_perspective.py)
===========================================

測試項目：
1. 角落類別建立與轉換
2. 透視變換器初始化
3. 座標轉換
4. 軌跡轉換
5. 自動偵測角落
6. 點排序

作者：Python 影像辨識工程師
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from perspective import (
    TableCorners,
    PerspectiveTransformer,
    MouseCornerSelector,
    auto_detect_table_corners,
    sort_four_points
)


class TestTableCorners:
    """測試 TableCorners 類別"""

    def test_create_corners(self):
        """測試 1：建立角落物件"""
        corners = TableCorners(
            top_left=(100, 100),
            top_right=(500, 100),
            bottom_left=(100, 400),
            bottom_right=(500, 400)
        )

        assert corners.top_left == (100, 100)
        assert corners.top_right == (500, 100)
        assert corners.bottom_left == (100, 400)
        assert corners.bottom_right == (500, 400)

    def test_to_array(self):
        """測試 2：轉換為陣列"""
        corners = TableCorners(
            top_left=(100, 100),
            top_right=(500, 100),
            bottom_left=(100, 400),
            bottom_right=(500, 400)
        )

        array = corners.to_array()

        assert array.shape == (4, 2)
        assert np.array_equal(array[0], [100, 100])
        assert np.array_equal(array[1], [500, 100])

    def test_from_array(self):
        """測試 3：從陣列建立（順時針順序）"""
        array = np.array([
            [100, 100],   # top_left
            [500, 100],   # top_right
            [500, 400],   # bottom_right
            [100, 400]    # bottom_left
        ], dtype=np.float32)

        corners = TableCorners.from_array(array)

        assert corners.top_left == (100, 100)
        assert corners.top_right == (500, 100)
        assert corners.bottom_right == (500, 400)
        assert corners.bottom_left == (100, 400)


class TestPerspectiveTransformerInit:
    """測試透視變換器初始化"""

    def test_init_default(self):
        """測試 4：使用預設參數"""
        transformer = PerspectiveTransformer()

        assert transformer.output_width == 800
        assert transformer.output_height == 600
        assert transformer.transform_matrix is None

    def test_init_custom(self):
        """測試 5：使用自訂參數"""
        transformer = PerspectiveTransformer(
            output_width=1000,
            output_height=700
        )

        assert transformer.output_width == 1000
        assert transformer.output_height == 700


class TestSetCorners:
    """測試設定角落"""

    def test_set_corners(self):
        """測試 6：設定角落並計算矩陣"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(100, 100),
            top_right=(500, 100),
            bottom_left=(100, 400),
            bottom_right=(500, 400)
        )

        transformer.set_corners(corners)

        assert transformer.current_corners is not None
        assert transformer.transform_matrix is not None
        assert transformer.inverse_matrix is not None


class TestPointTransform:
    """測試座標轉換"""

    def test_transform_point_no_matrix(self):
        """測試 7：沒有矩陣時回傳原座標"""
        transformer = PerspectiveTransformer()

        result = transformer.transform_point(100, 100)

        assert result == (100, 100)

    def test_transform_point_with_matrix(self):
        """測試 8：使用矩陣轉換"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer.set_corners(corners)

        result = transformer.transform_point(320, 240)

        # 應該在合理範圍內
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_inverse_transform(self):
        """測試 9：反向轉換"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(100, 100),
            top_right=(500, 100),
            bottom_left=(100, 400),
            bottom_right=(500, 400)
        )
        transformer.set_corners(corners)

        test_points = [(300, 250), (200, 200), (400, 300)]

        for original in test_points:
            transformed = transformer.transform_point(*original)
            # 檢查 transformed 在合理範圍內
            assert 0 <= transformed[0] < transformer.output_width
            assert 0 <= transformed[1] < transformer.output_height


class TestTrajectoryTransform:
    """測試軌跡轉換"""

    def test_transform_trajectory_empty(self):
        """測試 10：空軌跡"""
        transformer = PerspectiveTransformer()

        result = transformer.transform_trajectory([])

        assert result == []

    def test_transform_trajectory_single(self):
        """測試 11：單點軌跡"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer.set_corners(corners)

        trajectory = [(320, 240)]
        result = transformer.transform_trajectory(trajectory)

        assert len(result) == 1

    def test_transform_trajectory_multiple(self):
        """測試 12：多點軌跡"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer.set_corners(corners)

        trajectory = [(100, 100), (200, 200), (300, 300)]
        result = transformer.transform_trajectory(trajectory)

        assert len(result) == 3


class TestImageTransform:
    """測試影像轉換"""

    def test_transform_image_no_matrix(self):
        """測試 13：沒有矩陣時回傳原影像"""
        import cv2

        transformer = PerspectiveTransformer()

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = transformer.transform_image(image)

        assert result.shape == image.shape

    def test_transform_image_with_matrix(self):
        """測試 14：使用矩陣轉換影像"""
        transformer = PerspectiveTransformer(
            output_width=800,
            output_height=600
        )

        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer.set_corners(corners)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = transformer.transform_image(image)

        # 輸出尺寸應該是設定的值
        assert result.shape == (600, 800, 3)


class TestSortFourPoints:
    """測試四點排序"""

    def test_sort_normal(self):
        """測試 15：正常情況"""
        points = np.array([
            [500, 100],  # 右上
            [100, 100],  # 左上
            [500, 400],  # 右下
            [100, 400]   # 左下
        ])

        sorted_pts = sort_four_points(points)

        # 檢查排序結果
        assert tuple(sorted_pts[0]) == (100, 100)  # 左上
        assert tuple(sorted_pts[1]) == (500, 100)  # 右上
        assert tuple(sorted_pts[2]) == (100, 400)  # 左下
        assert tuple(sorted_pts[3]) == (500, 400)  # 右下

    def test_sort_already_sorted(self):
        """測試 16：已經排序"""
        points = np.array([
            [100, 100],
            [500, 100],
            [100, 400],
            [500, 400]
        ])

        sorted_pts = sort_four_points(points)

        assert np.array_equal(sorted_pts, points)


class TestDrawCorners:
    """測試角落繪製"""

    def test_draw_corners_no_corners(self):
        """測試 17：沒有角落時回傳原影像"""
        transformer = PerspectiveTransformer()

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = transformer.draw_corners(image)

        assert result.shape == image.shape

    def test_draw_corners_with_corners(self):
        """測試 18：有角落時繪製"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(100, 100),
            top_right=(500, 100),
            bottom_left=(100, 400),
            bottom_right=(500, 400)
        )
        transformer.set_corners(corners)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = transformer.draw_corners(image)

        assert result.shape == image.shape


class TestAutoDetect:
    """測試自動偵測"""

    def test_auto_detect_no_image(self):
        """測試 19：空影像"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = auto_detect_table_corners(image)

        # 可能偵測不到任何東西
        # 這裡只檢查函數可以執行
        assert result is None or isinstance(result, TableCorners)

    def test_auto_detect_invalid_params(self):
        """測試 20：無效參數"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = auto_detect_table_corners(image, min_area=1000000)

        assert result is None


class TestEdgeCases:
    """測試邊界情況"""

    def test_transform_zero_coordinates(self):
        """測試 21：零座標"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer.set_corners(corners)

        result = transformer.transform_point(0, 0)

        assert isinstance(result, tuple)

    def test_transform_negative_coordinates(self):
        """測試 22：負座標"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer.set_corners(corners)

        result = transformer.transform_point(-10, -10)

        # 負座標轉換後可能超出範圍
        assert isinstance(result, tuple)

    def test_transform_outside_bounds(self):
        """測試 23：超出邊界"""
        transformer = PerspectiveTransformer()

        corners = TableCorners(
            top_left=(0, 0),
            top_right=(640, 0),
            bottom_left=(0, 480),
            bottom_right=(640, 480)
        )
        transformer.set_corners(corners)

        result = transformer.transform_point(1000, 1000)

        assert isinstance(result, tuple)


# ===========================================
# 主程式入口
# ===========================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])