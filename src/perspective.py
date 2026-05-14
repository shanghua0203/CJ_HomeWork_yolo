"""
===========================================
視角轉換模組 (perspective.py)
===========================================

使用 OpenCV 的透視變換 (Perspective Transform) 將斜角畫面拉正為 2D 鳥瞰圖

主要功能：
1. 計算透視變換矩陣
2. 支援滑鼠點選四個角落手動設定轉換
3. 自動偵測桌面角落（預設功能）
4. 將落點座標轉換到鳥瞰圖

作者：Python 影像辨識工程師
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TableCorners:
    """
    桌面四個角落的座標
    順序：左上、右上、左下、右下
    """
    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_left: Tuple[int, int]
    bottom_right: Tuple[int, int]

    def to_array(self) -> np.ndarray:
        """
        轉換為 numpy 陣列格式

        回傳：
        - 4x2 的 numpy 陣列
        """
        return np.array([
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_right
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "TableCorners":
        """
        從 numpy 陣列建立

        參數：
        - array: 4x2 的 numpy 陣列

        回傳：
        - TableCorners 物件
        """
        return cls(
            top_left=tuple(map(int, array[0])),
            top_right=tuple(map(int, array[1])),
            bottom_left=tuple(map(int, array[2])),
            bottom_right=tuple(map(int, array[3]))
        )


class PerspectiveTransformer:
    """
    透視變換器
    將斜角的桌面視角轉換為 2D 鳥瞰圖
    """

    def __init__(
        self,
        output_width: int = 800,
        output_height: int = 600,
        default_corners: Optional[TableCorners] = None
    ):
        """
        初始化透視變換器

        參數說明：
        - output_width: 輸出鳥瞰圖的寬度
        - output_height: 輸出鳥瞰圖的高度
        - default_corners: 預設的四個角落座標（可選）
        """
        self.output_width = output_width
        self.output_height = output_height
        self.default_corners = default_corners
        self.current_corners: Optional[TableCorners] = default_corners
        self.transform_matrix: Optional[np.ndarray] = None
        self.inverse_matrix: Optional[np.ndarray] = None

    def set_corners(self, corners: TableCorners):
        """
        設定四個角落座標並計算轉換矩陣

        參數：
        - corners: 桌面四個角落座標
        """
        self.current_corners = corners
        self._calculate_transform_matrix()

    def _calculate_transform_matrix(self):
        """
        計算透視變換矩陣

        使用 OpenCV 的 getPerspectiveTransform 計算
        將斜角視角的四個點轉換為矩形鳥瞰圖
        """
        if self.current_corners is None:
            return

        # 取得原始座標點
        src_points = self.current_corners.to_array()

        # 定義輸出矩形的四個角
        dst_points = np.array([
            [0, 0],
            [self.output_width, 0],
            [0, self.output_height],
            [self.output_width, self.output_height]
        ], dtype=np.float32)

        # 計算透視變換矩陣
        self.transform_matrix = cv2.getPerspectiveTransform(
            src_points, dst_points
        )

        # 計算反向變換矩陣（用於從鳥瞰圖轉換回原視角）
        self.inverse_matrix = cv2.getPerspectiveTransform(
            dst_points, src_points
        )

    def transform_point(self, x: int, y: int) -> Tuple[int, int]:
        """
        將原視角座標轉換為鳥瞰圖座標

        參數：
        - x, y: 原視角的座標

        回傳：
        - (x, y) 鳥瞰圖中的座標
        """
        if self.transform_matrix is None:
            return (x, y)

        # 使用透視變換矩陣轉換座標
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.transform_matrix)

        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    def inverse_transform_point(self, x: int, y: int) -> Tuple[int, int]:
        """
        將鳥瞰圖座標轉換回原視角座標

        參數：
        - x, y: 鳥瞰圖中的座標

        回傳：
        - (x, y) 原視角的座標
        """
        if self.inverse_matrix is None:
            return (x, y)

        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.inverse_matrix)

        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    def transform_trajectory(
        self,
        trajectory: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        轉換整條軌跡

        參數：
        - trajectory: 原始軌跡座標

        回傳：
        - 轉換後的軌跡
        """
        if self.transform_matrix is None:
            return trajectory

        return [self.transform_point(x, y) for x, y in trajectory]

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        對整張影像進行透視變換

        參數：
        - image: 原始影像

        回傳：
        - 變換後的鳥瞰圖
        """
        if self.transform_matrix is None:
            return image

        return cv2.warpPerspective(
            image,
            self.transform_matrix,
            (self.output_width, self.output_height)
        )

    def draw_corners(
        self,
        image: np.ndarray,
        line_color: Tuple[int, int, int] = (0, 255, 0),
        line_thickness: int = 2,
        point_radius: int = 5
    ) -> np.ndarray:
        """
        在影像上繪製四個角落

        參數：
        - image: 原始影像
        - line_color: 線條顏色
        - line_thickness: 線條粗細
        - point_radius: 點的半徑

        回傳：
        - 繪製後的影像
        """
        if self.current_corners is None:
            return image

        output = image.copy()

        # 取得四個角落座標
        tl = self.current_corners.top_left
        tr = self.current_corners.top_right
        bl = self.current_corners.bottom_left
        br = self.current_corners.bottom_right

        # 繪製四邊形
        pts = np.array([tl, tr, br, bl], dtype=np.int32)
        cv2.polylines(output, [pts], True, line_color, line_thickness)

        # 繪製四個角落點
        cv2.circle(output, tl, point_radius, (255, 0, 0), -1)
        cv2.circle(output, tr, point_radius, (0, 255, 0), -1)
        cv2.circle(output, bl, point_radius, (0, 0, 255), -1)
        cv2.circle(output, br, point_radius, (0, 255, 255), -1)

        return output


class MouseCornerSelector:
    """
    滑鼠點選角落工具
    讓使用者用滑鼠在影像上點擊四個角落
    """

    def __init__(
        self,
        window_name: str = "Select Corners",
        output_width: int = 800,
        output_height: int = 600
    ):
        """
        初始化角落選擇器

        參數：
        - window_name: 視窗名稱
        - output_width: 輸出鳥瞰圖寬度
        - output_height: 輸出鳥瞰圖高度
        """
        self.window_name = window_name
        self.output_width = output_width
        self.output_height = output_height

        self.corners: List[Tuple[int, int]] = []
        self.image: Optional[np.ndarray] = None
        self.is_selecting = False

        # 預設順序：左上、右上、左下、右下
        self.labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]

    def mouse_callback(self, event, x, y, flags, param):
        """
        滑鼠事件回調函數
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # 按下左鍵，記錄點
            self.corners.append((x, y))

            # 超過四個點，清除重新選
            if len(self.corners) > 4:
                self.corners = [(x, y)]
            elif len(self.corners) == 4:
                self.is_selecting = True

    def select_corners(self, image: np.ndarray) -> Optional[TableCorners]:
        """
        讓使用者點選四個角落

        參數：
        - image: 要顯示的影像

        回傳：
        - TableCorners 物件，選取完成後回傳；取消則回傳 None
        """
        self.image = image.copy()
        self.corners = []
        self.is_selecting = False

        # 建立視窗
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print(f"\n=== 角落點選模式 ===")
        print("請依序點擊四個角落：")
        print("1. 左上角 (Top-Left)")
        print("2. 右上角 (Top-Right)")
        print("3. 左下角 (Bottom-Left)")
        print("4. 右下角 (Bottom-Right)")
        print("按 'r' 重新選取")
        print("按 'q' 取消")

        while True:
            # 繪製目前選取的情況
            display = self.image.copy()

            # 繪製已選取的點
            for i, (px, py) in enumerate(self.corners):
                color = self._get_color(i)
                cv2.circle(display, (px, py), 10, color, -1)
                cv2.putText(
                    display,
                    f"{i + 1}",
                    (px + 15, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            # 如果選了多個點，繪製連線
            if len(self.corners) > 1:
                for i in range(len(self.corners) - 1):
                    cv2.line(
                        display,
                        self.corners[i],
                        self.corners[i + 1],
                        (0, 255, 0),
                        2
                    )

            # 顯示提示文字
            remaining = 4 - len(self.corners)
            if remaining > 0:
                msg = f"Please click corner {len(self.corners) + 1}/4"
            else:
                msg = "Done! Press 'Enter' to confirm or 'q' to cancel"

            cv2.putText(
                display,
                msg,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(self.window_name, display)

            # 等待按鍵
            key = cv2.waitKey(1) & 0xFF

            # 按 'q' 取消
            if key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return None

            # 按 'r' 重選
            if key == ord('r'):
                self.corners = []
                self.is_selecting = False

            # 按 Enter 確認
            if key == 13 and len(self.corners) == 4:
                break

        cv2.destroyWindow(self.window_name)

        # 建立 TableCorners 物件
        return TableCorners(
            top_left=self.corners[0],
            top_right=self.corners[1],
            bottom_left=self.corners[2],
            bottom_right=self.corners[3]
        )

    def _get_color(self, index: int) -> Tuple[int, int, int]:
        """
        根據索引取得顏色
        """
        colors = [
            (255, 0, 0),    # 藍
            (0, 255, 0),    # 綠
            (0, 0, 255),    # 紅
            (0, 255, 255)   # 黃
        ]
        return colors[index % len(colors)]


def auto_detect_table_corners(
    image: np.ndarray,
    min_area: int = 1000
) -> Optional[TableCorners]:
    """
    自動偵測桌面角落

    使用邊緣偵測和輪廓尋找來偵測桌面

    參數：
    - image: 輸入影像
    - min_area: 最小輪廓面積

    回傳：
    - TableCorners 物件，偵測失敗回傳 None
    """
    # 轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 邊緣偵測
    edges = cv2.Canny(blurred, 50, 150)

    # 找輪廓
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # 找最大的四邊形輪廓
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # 逼近輪廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果是四邊形
        if len(approx) == 4:
            # 找出四個角並排序
            points = approx.reshape(4, 2)
            corners = sort_four_points(points)

            return TableCorners(
                top_left=tuple(corners[0]),
                top_right=tuple(corners[1]),
                bottom_left=tuple(corners[2]),
                bottom_right=tuple(corners[3])
            )

    return None


def sort_four_points(points: np.ndarray) -> np.ndarray:
    """
    將四個點排序為：左上、右上、左下、右下

    參數：
    - points: 4x2 的點陣列

    回傳：
    - 排序後的點陣列
    """
    # 根據 Y 座標分組（上面和下面）
    sorted_points = points[np.argsort(points[:, 1])]

    # 上面兩個點
    top_points = sorted_points[:2]
    # 下面兩個點
    bottom_points = sorted_points[2:]

    # 根據 X 座標排序
    top_left = top_points[np.argmin(top_points[:, 0])]
    top_right = top_points[np.argmax(top_points[:, 0])]
    bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
    bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]

    return np.array([top_left, top_right, bottom_left, bottom_right])


# ===========================================
# 如果直接執行此檔案，進行測試
# ===========================================
if __name__ == "__main__":
    print("透視變換模組測試")
    print("=" * 40)

    # 建立角落選擇器
    selector = MouseCornerSelector()

    # 建立透視變換器
    transformer = PerspectiveTransformer(
        output_width=800,
        output_height=600
    )

    # 測試點轉換
    result = transformer.transform_point(100, 100)
    print(f"測試點 (100, 100) 轉換後：{result}")

    print("\n提示：實際使用時請在有影像的情況下執行")