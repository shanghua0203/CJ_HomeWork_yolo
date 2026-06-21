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
        轉換為 numpy 陣列格式（順時針順序：左上 -> 右上 -> 右下 -> 左下）

        回傳：
        - 4x2 的 numpy 陣列
        """
        return np.array([
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "TableCorners":
        """
        從 numpy 陣列建立（預期順時針順序）

        參數：
        - array: 4x2 的 numpy 陣列 [左上, 右上, 右下, 左下]

        回傳：
        - TableCorners 物件
        """
        return cls(
            top_left=tuple(int(x) for x in array[0]),
            top_right=tuple(int(x) for x in array[1]),
            bottom_right=tuple(int(x) for x in array[2]),
            bottom_left=tuple(int(x) for x in array[3])
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

        # 定義輸出矩形的四個角（順序須匹配 src_points：TL, TR, BR, BL）
        dst_points = np.array([
            [0, 0],
            [self.output_width, 0],
            [self.output_width, self.output_height],
            [0, self.output_height]
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
        將原視角座標轉換為鳥瞰圖座標（含邊界檢查）

        參數：
        - x, y: 原視角的座標

        回傳：
        - (x, y) 鳥瞰圖中的座標（已 clamp 到合理範圍）
        """
        if self.transform_matrix is None:
            return (x, y)

        # 使用透視變換矩陣轉換座標
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.transform_matrix)

        tx = transformed[0][0][0]
        ty = transformed[0][0][1]

        # clamp 到鳥瞰圖範圍內
        tx = max(0, min(int(tx), self.output_width - 1))
        ty = max(0, min(int(ty), self.output_height - 1))

        return (tx, ty)

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

        # 排序四個點確保順序正確（左上→右上→左下→右下）
        points_array = np.array(self.corners, dtype=np.float32)
        sorted_pts = sort_four_points(points_array)

        return TableCorners(
            top_left=tuple(int(x) for x in sorted_pts[0]),
            top_right=tuple(int(x) for x in sorted_pts[1]),
            bottom_left=tuple(int(x) for x in sorted_pts[2]),
            bottom_right=tuple(int(x) for x in sorted_pts[3])
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

    使用邊緣偵測和輪廓尋找來偵測桌面。
    嘗試多組 Canny 閾值，回傳面積最大的四邊形。

    參數：
    - image: 輸入影像
    - min_area: 最小輪廓面積

    回傳：
    - TableCorners 物件，偵測失敗回傳 None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = image.shape[:2]
    max_img_area = h * w

    # 嘗試多組 Canny 閾值，適應不同亮度/對比
    canny_params = [
        (50, 150),
        (30, 100),
        (100, 200),
        (20, 80),
    ]

    best_corners = None
    best_area = 0

    for low, high in canny_params:
        edges = cv2.Canny(blurred, low, high)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_img_area * 0.95:
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) != 4:
                continue

            points = approx.reshape(4, 2)
            corners = sort_four_points(points)

            if area > best_area:
                best_area = area
                best_corners = TableCorners(
                    top_left=tuple(corners[0]),
                    top_right=tuple(corners[1]),
                    bottom_left=tuple(corners[2]),
                    bottom_right=tuple(corners[3])
                )

    return best_corners


def validate_corners(
    corners: TableCorners,
    frame_width: int,
    frame_height: int,
    min_side_ratio: float = 0.05
) -> bool:
    """
    驗證桌面角落是否合理。

    檢查條件：
    - 所有點在畫面內
    - 四邊形非退化（有實際面積）
    - 長邊 / 短邊比例不超過 5:1

    回傳：
    - True 如果角落合理
    """
    pts = np.array([
        corners.top_left, corners.top_right,
        corners.bottom_left, corners.bottom_right
    ], dtype=np.float32)

    # 所有點必須在畫面範圍內（允許小幅邊界溢位）
    margin = 50
    if np.any(pts[:, 0] < -margin) or np.any(pts[:, 0] > frame_width + margin):
        return False
    if np.any(pts[:, 1] < -margin) or np.any(pts[:, 1] > frame_height + margin):
        return False

    # 四邊形必須有正面積
    area = cv2.contourArea(pts.astype(np.int32).reshape(4, 1, 2))
    if area < frame_width * frame_height * min_side_ratio:
        return False

    # 長寬比不能過於極端（桌面約 2:1，允許到 5:1）
    side_lengths = [
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[1] - pts[3]),
        np.linalg.norm(pts[3] - pts[2]),
        np.linalg.norm(pts[2] - pts[0]),
    ]
    long_side = max(side_lengths)
    short_side = min(side_lengths)
    if short_side == 0:
        return False
    if long_side / short_side > 5.0:
        return False

    return True


def corners_center_distance(
    a: TableCorners,
    b: TableCorners
) -> float:
    """計算兩組角落中心點的歐氏距離"""
    def center(c: TableCorners):
        pts = np.array([c.top_left, c.top_right, c.bottom_left, c.bottom_right])
        return np.mean(pts[:, 0]), np.mean(pts[:, 1])
    ca = center(a)
    cb = center(b)
    return float(np.sqrt((ca[0] - cb[0])**2 + (ca[1] - cb[1])**2))


def sort_four_points(points: np.ndarray) -> np.ndarray:
    """
    將四個點排序為：左上、右上、左下、右下

    使用 (x+y) 與 (x-y) 組合判斷，比單純 Y 軸分組或角度排序更穩健。

    參數：
    - points: 4x2 的點陣列

    回傳：
    - 排序後的點陣列
    """
    s = points[:, 0] + points[:, 1]   # x+y：左上最小，右下最大
    d = points[:, 0] - points[:, 1]   # x-y：右上→左下遞增

    tl_idx = int(np.argmin(s))
    br_idx = int(np.argmax(s))

    # 從剩餘兩點中區分 TR（x-y 較大，右上）與 BL（x-y 較小，左下）
    remaining = [i for i in range(4) if i != tl_idx and i != br_idx]
    if d[remaining[0]] > d[remaining[1]]:
        tr_idx, bl_idx = remaining[0], remaining[1]
    else:
        tr_idx, bl_idx = remaining[1], remaining[0]

    return np.array([
        points[tl_idx],
        points[tr_idx],
        points[bl_idx],
        points[br_idx],
    ])


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