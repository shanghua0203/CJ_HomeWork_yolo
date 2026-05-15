"""
===========================================
乒乓球偵測模組 (detector.py)
===========================================
使用 YOLO (ultralytics) 偵測乒乓球並輸出中心座標 (x, y)

主要功能：
1. 載入 YOLO 模型
2. 對輸入影像進行偵測
3. 輸出乒乓球中心座標

作者：Python 影像辨識工程師
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from ultralytics import YOLO


class BallDetector:
    """
    乒乓球偵測器
    使用 YOLO 模型偵測乒乓球並返回中心座標
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        class_id: int = 32,
        confidence: float = 0.5,
        min_size: int = 10,
        iou_threshold: float = 0.4
    ):
        """
        初始化偵測器

        參數說明：
        - model_path: YOLO 模型路徑（預設為 yolov8n.pt，會自動從 ultralytics 下載）
        - class_id: 要偵測的類別編號（預設為 0）
        - confidence: 信心閾值，偵測結果必須超過此信心值才採用
        - min_size: 偵測框最小邊長，低於此值忽略
        - iou_threshold: 非極大值抑制 (NMS) 的 IOU 閾值
        """
        self.model_path = model_path
        self.class_id = class_id
        self.confidence = confidence
        self.min_size = min_size
        self.iou_threshold = iou_threshold

        # 載入 YOLO 模型
        # 使用 verbose=False 減少終端機輸出
        self.model = YOLO(model_path)
        
        # 過濾已知的誤判位置（穩定的錯誤偵測）
        # 格式：(center_x, center_y, tolerance)
        self.filter_false_positives = [
            (1750, 581, 15),  # 影片右上角LOGO/計時器區域
        ]

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        偵測乒乓球並返回中心座標

        參數：
        - frame: OpenCV 讀取的影像 (numpy array，格式為 BGR)

        回傳：
        - (x, y) 中心座標（像素），如果沒偵測到球則回傳 None
        """
        # 執行 YOLO 偵測
        # conf=信心閾值, iou=NMS閾值, verbose=False=安靜模式
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )

        # 解析偵測結果
        if results is None or len(results) == 0:
            return None

        # 取得第一個結果（因為我們只輸入一張圖）
        result = results[0]

        # 取得所有偵測到的物件
        boxes = result.boxes

        # 如果沒有偵測到任何物件
        if boxes is None or len(boxes) == 0:
            return None

        # 遍歷所有偵測到的物件，找乒乓球
        best_ball = None
        best_conf = 0.0

        for box in boxes:
            # 取得類別編號
            cls_id = int(box.cls[0])

            # 檢查是否符合目標類別
            if cls_id != self.class_id:
                continue

            # 取得信心值
            conf = float(box.conf[0])

            # 檢查信心值是否符合閾值
            if conf < self.confidence:
                continue

            # 取得偵測框座標 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # 檢查框的大小是否符合最小尺寸要求
            width = x2 - x1
            height = y2 - y1

            if width < self.min_size or height < self.min_size:
                continue

            # 選擇信心值最高的球
            if conf > best_conf:
                best_conf = conf
                best_ball = (x1, y1, x2, y2)

        # 如果沒有找到符合條件的球
        if best_ball is None:
            return None

        # 計算中心座標
        x1, y1, x2, y2 = best_ball
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return (center_x, center_y)

    def detect_batch(self, frames: List[np.ndarray]) -> List[Optional[Tuple[int, int]]]:
        """
        批次偵測多張影像中的乒乓球

        參數：
        - frames: 多張影像的列表

        回傳：
        - 每張影像對應的中心座標列表，沒偵測到則為 None
        """
        results = []

        for frame in frames:
            result = self.detect(frame)
            results.append(result)

        return results

    def detect_with_box(
        self,
        frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int, int, int, float]]:
        """
        偵測乒乓球並返回中心座標與偵測框資訊

        參數：
        - frame: OpenCV 讀取的影像 (numpy array，格式為 BGR)

        回傳：
        - (center_x, center_y, x1, y1, x2, y2, confidence)
          如果沒偵測到球則回傳 None
        """
        # 執行 YOLO 偵測
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )

        # 解析偵測結果
        if results is None or len(results) == 0:
            return None

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return None

        # 遍歷所有偵測到的物件，找乒乓球
        best_ball = None
        best_conf = 0.0

        for box in boxes:
            cls_id = int(box.cls[0])

            if cls_id != self.class_id:
                continue

            conf = float(box.conf[0])

            if conf < self.confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            width = x2 - x1
            height = y2 - y1

            if width < self.min_size or height < self.min_size:
                continue

            if conf > best_conf:
                best_conf = conf
                best_ball = (x1, y1, x2, y2)

        if best_ball is None:
            return None

        x1, y1, x2, y2 = best_ball
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if self._is_false_positive(center_x, center_y):
            return None
        
        return (int(center_x), int(center_y), int(x1), int(y1), int(x2), int(y2), float(best_conf))

    def _is_false_positive(self, center_x: float, center_y: float) -> bool:
        """檢查是否是已知的誤判位置"""
        for fx, fy, tolerance in self.filter_false_positives:
            if abs(center_x - fx) <= tolerance and abs(center_y - fy) <= tolerance:
                return True
        return False

    def detect_all_objects(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25
    ) -> List[Tuple[int, int, int, int, int, float, str]]:
        """
        偵測畫面中所有物件（用於偵測球桌）

        參數：
        - frame: 原始影像
        - conf_threshold: 信心閾值

        回傳：
        - [(center_x, center_y, x1, y1, x2, y2, confidence, class_name), ...]
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)

        if not results or not results[0].boxes:
            return []

        detections = []
        # COCO 類別名稱（只用常見的）
        coco_names = {
            0: 'person', 32: 'sports ball', 13: 'bench',
            38: 'teddy bear'
        }

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            class_name = coco_names.get(cls_id, f'class_{cls_id}')

            detections.append((
                center_x, center_y,
                int(x1), int(y1), int(x2), int(y2),
                float(conf),
                class_name
            ))

        return detections


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    載入影像檔案

    參數：
    - image_path: 影像檔案路徑

    回傳：
    - OpenCV 格式的影像，載入失敗回傳 None
    """
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"錯誤：無法讀取影像 {image_path}")
        return None

    return frame


def detect_from_video(
    video_path: str,
    model_path: str = "yolov8n.pt",
    class_id: int = 32,
    confidence: float = 0.5
) -> List[Tuple[int, int, int]]:
    """
    從影片中偵測乒乓球並返回所有座標

    參數：
    - video_path: 影片檔案路徑
    - model_path: YOLO 模型路徑
    - class_id: 類別編號
    - confidence: 信心閾值

    回傳：
    - [(frame_number, x, y), ...] 所有偵測到的球座標
    """
    # 建立偵測器
    detector = BallDetector(
        model_path=model_path,
        class_id=class_id,
        confidence=confidence
    )

    # 開啟影片
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 {video_path}")
        return []

    # 存放所有偵測結果
    detections = []
    frame_number = 0

    # 逐幀偵測
    while True:
        # 讀取下一幀
        ret, frame = cap.read()

        # 如果讀取失敗（影片結束），跳出迴圈
        if not ret:
            break

        # 偵測乒乓球
        center = detector.detect(frame)

        # 如果有偵測到球，記錄座標
        if center is not None:
            x, y = center
            detections.append((frame_number, x, y))

        frame_number += 1

    # 釋放影片資源
    cap.release()

    return detections


# ===========================================
# 如果直接執行此檔案，進行測試
# ===========================================
if __name__ == "__main__":
    import sys

    # 檢查是否有提供測試影像
    if len(sys.argv) < 2:
        print("用法：python detector.py <影像路徑>")
        print("範例：python detector.py test.jpg")
        sys.exit(1)

    # 取得影像路徑
    image_path = sys.argv[1]

    # 載入影像
    frame = load_image(image_path)

    if frame is not None:
        # 建立偵測器（使用預設模型）
        detector = BallDetector()

        # 偵測乒乓球
        result = detector.detect(frame)

        if result is not None:
            x, y = result
            print(f"偵測到乒乓球，中心座標：({x}, {y})")
        else:
            print("未偵測到乒乓球")