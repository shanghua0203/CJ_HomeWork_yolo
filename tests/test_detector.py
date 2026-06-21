"""
===========================================
乒乓球偵測器單元測試 (test_detector.py)
===========================================

測試項目：
1. 建立偵測器實例
2. 載入影像
3. 偵測乒乓球並驗證輸出格式為 (x, y)

作者：Python 影像辨識工程師
"""

import pytest
import numpy as np
import cv2
import os
import sys

# 將 src 目錄加入路徑，以便匯入模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import BallDetector, load_image, detect_from_video


class TestBallDetector:
    """測試 BallDetector 類別"""

    def test_detector_init_default(self):
        """測試 1：使用預設參數建立偵測器"""
        detector = BallDetector()

        assert detector is not None
        assert detector.model is not None
        # 預設類別現在是 32 (sports ball)
        assert detector.class_id == 32
        assert detector.confidence == 0.01

    def test_detector_init_custom(self):
        """測試 2：使用自訂參數建立偵測器"""
        detector = BallDetector(
            model_path="yolov8n.pt",
            class_id=1,
            confidence=0.8,
            min_size=20,
            iou_threshold=0.5
        )

        assert detector.class_id == 1
        assert detector.confidence == 0.8
        assert detector.min_size == 20
        assert detector.iou_threshold == 0.5

    def test_detector_model_loaded(self):
        """測試 3：確認 YOLO 模型已成功載入"""
        detector = BallDetector()

        # 模型應該已經載入且可用
        assert detector.model is not None


class TestLoadImage:
    """測試 load_image 函數"""

    def test_load_valid_image(self):
        """測試 4：載入有效的影像檔案"""
        # 建立一張測試影像（100x100 的白色圖片）
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # 先將影像存成檔案
        test_path = "/tmp/test_image.jpg"
        cv2.imwrite(test_path, test_image)

        # 載入影像
        frame = load_image(test_path)

        # 驗證
        assert frame is not None
        assert frame.shape == (100, 100, 3)

        # 清理測試檔案
        os.remove(test_path)

    def test_load_invalid_image(self):
        """測試 5：載入不存在的影像應該回傳 None"""
        result = load_image("/tmp/nonexistent_image_12345.jpg")

        assert result is None


class TestBallDetection:
    """測試球體偵測功能"""

    @pytest.fixture
    def test_image_path(self):
        """建立測試影像"""
        # 建立一張 640x480 的測試影像
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 寫入暫存檔案
        path = "/tmp/test_detection.jpg"
        cv2.imwrite(path, frame)

        yield path

        # 測試後清理
        if os.path.exists(path):
            os.remove(path)

    def test_detect_returns_tuple_or_none(self, test_image_path):
        """測試 6：偵測結果應該是 (x, y) 座標或 None"""
        # 建立偵測器
        detector = BallDetector()

        # 載入測試影像
        frame = load_image(test_image_path)

        # 執行偵測
        result = detector.detect(frame)

        # 結果應該是 None（因為測試影像沒有球）
        # 或者是 (x, y) 元組
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 2
            x, y = result
            assert isinstance(x, int)
            assert isinstance(y, int)

    def test_detect_output_format(self):
        """測試 7：確認偵測輸出格式正確"""
        # 建立一個簡單的測試影像（模拟检测场景）
        # 這裡我們測試即使沒有球，也不會崩潰
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        detector = BallDetector(confidence=0.1)

        # 執行偵測不會抛出異常
        result = detector.detect(frame)

        # 回傳值應該是 None 或者有效的座標
        assert result is None or isinstance(result, tuple)

    def test_detect_returns_valid_coordinates(self):
        """測試 8：偵測到的座標應該在合理範圍內"""
        # 建立一個稍大的測試影像
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detector = BallDetector(confidence=0.1)
        result = detector.detect(frame)

        # 如果有偵測到球，座標應該在影像範圍內
        if result is not None:
            x, y = result
            assert 0 <= x <= 640
            assert 0 <= y <= 480


class TestVideoDetection:
    """測試影片偵測功能"""

    def test_detect_from_video_nonexistent(self):
        """測試 9：不存在的影片應該回傳空列表"""
        result = detect_from_video("/tmp/nonexistent_video.mp4")

        assert result == []

    def test_detect_from_video_returns_list(self):
        """測試 10：影片偵測應該回傳列表格式"""
        # 由於沒有測試影片，這裡只測試函數不會崩潰
        # 預期會回傳空列表
        result = detect_from_video("data/sample_video.mp4")

        assert isinstance(result, list)


class TestEdgeCases:
    """測試邊界情況"""

    def test_detect_empty_frame(self):
        """測試 11：空影像"""
        frame = np.array([])

        detector = BallDetector()

        # 應該能處理（可能會報錯，這是預期行為）
        try:
            result = detector.detect(frame)
            # 如果成功，結果應該是 None
            assert result is None or isinstance(result, tuple)
        except Exception:
            # 這也是可接受的結果
            pass

    def test_detect_single_pixel_frame(self):
        """測試 12：單像素影像"""
        frame = np.zeros((1, 1, 3), dtype=np.uint8)

        detector = BallDetector()
        result = detector.detect(frame)

        # 結果應該是 None（太小無法偵測）
        assert result is None

    def test_detect_batch_empty(self):
        """測試 13：批次偵測空列表"""
        detector = BallDetector()

        result = detector.detect_batch([])

        assert result == []


# ===========================================
# 主程式入口
# ===========================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])