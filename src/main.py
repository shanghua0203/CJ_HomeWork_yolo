"""
===========================================
主程式 (main.py)
===========================================

乒乓球落點分析系統主程式
整合所有模組：偵測、過濾、追蹤、落點判斷、視角轉換、視覺化

作者：Python 影像辨識工程師
"""

import cv2
import numpy as np
import sys
import os
import argparse
import yaml
from typing import List, Tuple, Optional, Dict, Any

from detector import BallDetector, load_image
from filter import TrajectoryFilter
from tracker import BallTracker
from landing_detector import LandingDetector, LandingPoint
from perspective import (
    PerspectiveTransformer,
    TableCorners,
    MouseCornerSelector,
    auto_detect_table_corners,
    validate_corners,
    corners_center_distance,
)
from visualizer import (
    TrajectoryVisualizer,
    BirdEyeVisualizer,
    VisualizationConfig,
    create_side_by_side_view
)


class PingPongAnalyzer:
    """
    乒乓球分析器
    整合所有分析功能
    """

    def __init__(
        self,
        video_path: str,
        model_path: str = "yolov8n.pt",
        output_dir: str = "output",
        use_mouse_selection: bool = False,
        auto_detect_table: bool = True,
        confidence: float = 0.01,
        frame_width: int = 1920,
        frame_height: int = 1080,
        max_jump: int = 100,
        max_missing_frames: int = 2,
        y_reversal_threshold: int = 5,
        min_fall_distance: int = 20,
        persp_output_width: int = 800,
        persp_output_height: int = 600,
        class_id: int = 32,
        min_size: int = 5,
        iou_threshold: float = 0.4,
        default_corners: Optional[TableCorners] = None,
    ):
        """
        初始化分析器
        """
        self.video_path = video_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_jump = max_jump
        self.output_dir = output_dir
        self.use_mouse_selection = use_mouse_selection
        self.auto_detect_table = auto_detect_table
        self.confidence = confidence

        # 建立輸出資料夾
        os.makedirs(output_dir, exist_ok=True)

        # 初始化各模組
        self.detector = BallDetector(
            model_path=model_path,
            confidence=confidence,
            class_id=class_id,
            min_size=min_size,
            iou_threshold=iou_threshold,
        )

        self.filter = TrajectoryFilter(
            frame_width=frame_width,
            frame_height=frame_height,
            max_jump=max_jump,
            max_missing_frames=max_missing_frames
        )
        self.tracker = BallTracker(max_missing_frames=max_missing_frames)
        self.landing_detector = LandingDetector(
            y_reversal_threshold=y_reversal_threshold,
            min_fall_distance=min_fall_distance,
            frame_height=frame_height
        )

        self.transformer = PerspectiveTransformer(
            output_width=persp_output_width,
            output_height=persp_output_height,
            default_corners=default_corners,
        )

        # 視覺化器
        self.viz_config = VisualizationConfig()
        self.trajectory_viz = TrajectoryVisualizer(config=self.viz_config)
        self.bird_eye_viz: Optional[BirdEyeVisualizer] = None

        # 影片擷取物件
        self.cap = None

        # 分析結果
        self.all_landings: List[LandingPoint] = []

        # 角落重新偵測狀態
        self._consecutive_corner_failures = 0

    def setup_perspective(self, frame: np.ndarray):
        """
        設定透視變換

        參數：
        - frame: 包含桌子的參考影像
        """
        h, w = frame.shape[:2]
        corners = None

        # 嘗試自動偵測
        if self.auto_detect_table:
            print("嘗試自動偵測桌面角落...")
            corners = auto_detect_table_corners(frame)
            if corners and not validate_corners(corners, w, h):
                print("  自動偵測結果驗證失敗，嘗試其他方式")
                corners = None

        # 如果自動偵測失敗或需要手動點選
        if corners is None:
            if self.use_mouse_selection:
                print("請手動點選桌面四個角落...")
                selector = MouseCornerSelector()
                corners = selector.select_corners(frame)

                if corners is None:
                    print("取消角落點選，使用預設角落")
                    corners = TableCorners(
                        top_left=(100, 100),
                        top_right=(540, 100),
                        bottom_left=(100, 380),
                        bottom_right=(540, 380)
                    )
            else:
                print("使用預設角落座標")
                corners = TableCorners(
                    top_left=(100, 100),
                    top_right=(540, 100),
                    bottom_left=(100, 380),
                    bottom_right=(540, 380)
                )

        # 設定角落並建立鳥瞰圖視覺化器
        self.transformer.set_corners(corners)
        self.bird_eye_viz = BirdEyeVisualizer(transformer=self.transformer)

        print(f"角落設定完成：")
        print(f"  左上：{corners.top_left}")
        print(f"  右上：{corners.top_right}")
        print(f"  左下：{corners.bottom_left}")
        print(f"  右下：{corners.bottom_right}")

    def process_video(
        self,
        show_preview: bool = True,
        save_output: bool = True,
        max_frames: Optional[int] = None
    ) -> List[LandingPoint]:
        """
        處理影片

        參數：
        - show_preview: 是否顯示預覽視窗
        - save_output: 是否儲存輸出影片
        - max_frames: 最大處理幀數（None 表示處理全部）

        回傳：
        - 所有偵測到的落點列表
        """
        # 開啟影片
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise ValueError(f"無法開啟影片：{self.video_path}")

        # 取得影片屬性
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 以實際影片尺寸覆寫各模組設定
        self.frame_width = width
        self.frame_height = height
        self.filter.frame_width = width
        self.filter.frame_height = height
        self.landing_detector.frame_height = height

        print(f"\n影片屬性：")
        print(f"  尺寸：{width} x {height}")
        print(f"  幀率：{fps} fps")
        print(f"  總幀數：{total_frames}")

        # 讀取第一幀設定透視變換
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("無法讀取影片第一幀")

        self.setup_perspective(first_frame)

        # 重置影片位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 確保輸出尺寸是偶數（FFmpeg 要求）
        combined_w = width + self.transformer.output_width
        combined_h = max(height, self.transformer.output_height)
        output_w = (combined_w + 1) // 2 * 2
        output_h = (combined_h + 1) // 2 * 2

        # 建立輸出影片編寫器（僅在需要儲存時）
        out = None
        if save_output:
            output_path = os.path.join(self.output_dir, "result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))

        # 重置追蹤器
        self.tracker.reset()
        self.all_landings = []

        # 重置落點偵測器
        self.landing_detector.reset()

        frame_number = 0
        processed_frames = 0

        # 每 N 幀偵測一次桌面角落（攝影機移動時需要重新偵測）
        table_detect_interval = 150
        last_table_detection = 0

        # 追蹤軌跡段，用於分段偵測落點
        last_frame_idx = -9999

        print("\n開始處理影片...")

        while True:
            # 讀取下一幀
            ret, frame = self.cap.read()

            if not ret:
                break

            # 定期偵測桌面角落（附多重 sanity check）
            if frame_number - last_table_detection >= table_detect_interval:
                new_corners = auto_detect_table_corners(frame)
                can_update = False
                skip_reason = ""

                if new_corners and self.transformer.current_corners:
                    # 驗證 1：基本合理性檢查（在畫面內、有面積、長寬比正常）
                    if not validate_corners(new_corners, width, height):
                        skip_reason = f"基本驗證失敗"
                    else:
                        # 驗證 2：面積比（新舊角落 bounding box 面積比較）
                        old_pts = self.transformer.current_corners.to_array()
                        new_pts = new_corners.to_array()
                        old_area = (np.max(old_pts[:, 0]) - np.min(old_pts[:, 0])) * \
                                   (np.max(old_pts[:, 1]) - np.min(old_pts[:, 1]))
                        new_area = (np.max(new_pts[:, 0]) - np.min(new_pts[:, 0])) * \
                                   (np.max(new_pts[:, 1]) - np.min(new_pts[:, 1]))
                        area_ratio = new_area / old_area if old_area > 0 else 1.0

                        if area_ratio < 0.2 or area_ratio > 5.0:
                            skip_reason = f"面積比 {area_ratio:.2f} 超出範圍 [0.2, 5.0]"
                        else:
                            # 驗證 3：中心點位移（避免跳到畫面另一側）
                            dist = corners_center_distance(
                                self.transformer.current_corners, new_corners
                            )
                            if dist > width * 0.5:
                                skip_reason = f"中心位移 {dist:.0f}px 超過畫面寬度 50%"
                            else:
                                can_update = True

                elif new_corners and not self.transformer.current_corners:
                    # 首次設定或之前無有效角落：只做基本驗證
                    if validate_corners(new_corners, width, height):
                        can_update = True
                    else:
                        skip_reason = "基本驗證失敗"

                if can_update:
                    self.transformer.set_corners(new_corners)
                    self.bird_eye_viz = BirdEyeVisualizer(transformer=self.transformer)
                    last_table_detection = frame_number
                    self._consecutive_corner_failures = 0
                    print(f"  [幀 {frame_number}] 更新桌面角落")
                else:
                    self._consecutive_corner_failures += 1
                    if self._consecutive_corner_failures <= 3:
                        # 前 3 次只記錄，不跳過間隔（容錯）
                        print(f"  [幀 {frame_number}] 角落驗證略過（{skip_reason}）")
                    else:
                        # 連續失敗 3 次以上才跳過間隔，避免反覆失敗浪費效能
                        last_table_detection = frame_number
                        print(f"  [幀 {frame_number}] 連續 {self._consecutive_corner_failures} 次失敗，跳過此週期")

            # 偵測乒乓球（使用新方法取得偵測框）
            detection_result = self.detector.detect_with_box(frame)
            detection = None

            if detection_result:
                center_x, center_y, x1, y1, x2, y2, conf = detection_result
                detection = (center_x, center_y)
                current_detection = (x1, y1, x2, y2, conf, 'ball')
            else:
                current_detection = None

            # 更新追蹤器
            self.tracker.update(detection, frame_number)

            # 追蹤當前球段
            trajectory = self.tracker.get_trajectory()
            frame_indices = self.tracker.get_frame_indices()
            
            # 找當前球開始的索引（大間隙後開始）
            current_ball_start = 0
            if len(frame_indices) >= 2:
                for j in range(len(frame_indices) - 1, 0, -1):
                    if frame_indices[j] - frame_indices[j-1] > 50:
                        current_ball_start = j + 1
                        break
            
            # 只取當前球的軌跡
            current_trajectory = trajectory[current_ball_start:]
            current_frame_indices = frame_indices[current_ball_start:]
            
            # 掉幀補間（填補小的幀間隔）
            interpolated = self.filter.interpolate_missing(
                current_trajectory, current_frame_indices
            )
            valid_trajectory = [p for p in interpolated if p is not None]

            # 過濾雜訊
            filtered_trajectory = self.filter.filter_trajectory(valid_trajectory)
            
            # 偵測落點（不傳 frame_indices，因插值/過濾後長度已不一致）
            current_landings = []
            if len(filtered_trajectory) >= 3:
                current_landings = self.landing_detector.detect(
                    filtered_trajectory
                )
            
            # 添加新落點到總列表（避免重複）
            for lp in current_landings:
                is_dup = any(
                    abs(lp.x - l.x) < 10 and abs(lp.y - l.y) < 10 
                    for l in self.all_landings
                )
                if not is_dup:
                    self.all_landings.append(lp)

            # 繪製軌跡（先建立 output_frame）
            output_frame = self.trajectory_viz.draw_trajectory(
                frame, filtered_trajectory
            )

            # 繪製偵測框（疊在軌跡上方）
            if current_detection:
                x1, y1, x2, y2, conf, label = current_detection
                output_frame = self.trajectory_viz.draw_detection_box(
                    output_frame, x1, y1, x2, y2, conf, label
                )

            # 繪製落點
            output_frame = self.trajectory_viz.draw_landing_points(
                output_frame, self.all_landings
            )

            # 繪製資訊面板
            output_frame = self.trajectory_viz.draw_info_panel(
                output_frame,
                filtered_trajectory,
                self.all_landings,
                frame_number
            )

            # 建立鳥瞰圖
            bird_eye = self.bird_eye_viz.create_bird_eye_view()
            bird_eye = self.bird_eye_viz.draw_grid(bird_eye)
            bird_eye = self.bird_eye_viz.draw_landing_points(
                bird_eye, self.all_landings
            )

            # 左右併排
            combined = create_side_by_side_view(output_frame, bird_eye)

            # 顯示預覽
            if show_preview:
                cv2.imshow("Ping Pong Analysis", combined)

                # 按 'q' 退出，按 'p' 暫停
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)

            # 儲存輸出（若尺寸不符則填補至偶數）
            if save_output:
                if combined.shape[1] != output_w or combined.shape[0] != output_h:
                    padded = np.zeros((output_h, output_w, 3), dtype=np.uint8)
                    ch, cw = combined.shape[:2]
                    padded[:ch, :cw] = combined
                    out.write(padded)
                else:
                    out.write(combined)

            frame_number += 1
            processed_frames += 1

            # 進度顯示
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"\r處理進度：{progress:.1f}% ({frame_number}/{total_frames})", end="")

            # 最大幀數限制
            if max_frames and processed_frames >= max_frames:
                break

        print(f"\n\n處理完成！")
        print(f"總共處理：{processed_frames} 幀")
        print(f"偵測到落點：{len(self.all_landings)} 個")

        # 釋放資源
        self.cap.release()
        if save_output:
            out.release()

        if show_preview:
            cv2.destroyAllWindows()

        # 儲存落點報告
        self.save_landing_report()

        return self.all_landings

    def save_landing_report(self):
        """
        儲存落點報告
        """
        report_path = os.path.join(self.output_dir, "landing_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("乒乓球落點分析報告\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"輸入影片：{self.video_path}\n")
            f.write(f"偵測到落點數量：{len(self.all_landings)}\n\n")

            for i, landing in enumerate(self.all_landings):
                f.write(f"\n落點 {i + 1}:\n")
                f.write(f"  座標：(x={landing.x}, y={landing.y})\n")
                f.write(f"  幀編號：{landing.frame_index}\n")
                f.write(f"  最低 Y 值：{landing.lowest_y}\n")

                # 轉換為鳥瞰圖座標
                bx, by = self.transformer.transform_point(landing.x, landing.y)
                f.write(f"  鳥瞰圖座標：(x={bx}, y={by})\n")

        print(f"\n落點報告已儲存至：{report_path}")


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    讀取設定檔，回傳設定字典（不存在時回傳空字典）
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """
    主程式入口
    """
    parser = argparse.ArgumentParser(
        description="乒乓球落點分析系統"
    )

    parser.add_argument(
        "video",
        nargs="?",
        default="data/sample_video.mp4",
        help="輸入影片路徑（預設：data/sample_video.mp4）"
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO 模型路徑（預設：yolov8n.pt）"
    )

    parser.add_argument(
        "--output",
        default="output",
        help="輸出資料夾（預設：output）"
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="不顯示預覽視窗"
    )

    parser.add_argument(
        "--mouse-select",
        action="store_true",
        help="使用滑鼠點選桌子角落"
    )

    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="不使用自動偵測桌子角落"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="偵測信心閾值（預設由 config.yaml 決定，可在此覆蓋）"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大處理幀數"
    )

    args = parser.parse_args()

    # 載入設定檔
    config = load_config("config/config.yaml")

    # 設定值優先序：命令列 > config.yaml > 程式碼預設值
    cfg_paths = config.get("paths", {})
    if args.model == "yolov8n.pt":
        args.model = cfg_paths.get("model", "yolov8n.pt")
    if args.output == "output":
        args.output = cfg_paths.get("output", "output")

    cfg_detector = config.get("detector", {})
    if args.confidence is None:
        args.confidence = cfg_detector.get("confidence", 0.01)
    detector_class_id = cfg_detector.get("class_id", 32)
    detector_min_size = int(cfg_detector.get("min_size", 5))
    detector_iou = cfg_detector.get("iou_threshold", 0.4)

    cfg_filter = config.get("filter", {})
    cfg_landing = config.get("landing", {})
    cfg_persp = config.get("perspective", {})
    persp_default_corners = None
    if "default_corners" in cfg_persp:
        dc = cfg_persp["default_corners"]
        try:
            persp_default_corners = TableCorners(
                top_left=tuple(dc["top_left"]),
                top_right=tuple(dc["top_right"]),
                bottom_left=tuple(dc["bottom_left"]),
                bottom_right=tuple(dc["bottom_right"]),
            )
        except Exception as e:
            print(f"警告：預設角落格式無效，跳過 ({e})")

    print("=" * 50)
    print("乒乓球落點分析系統")
    print("=" * 50)
    print(f"設定：模型={args.model} confidence={args.confidence}")
    print(f"      config 載入={'是' if config else '否'}")

    # 建立分析器（CLI > config.yaml > 預設值）
    analyzer = PingPongAnalyzer(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output,
        use_mouse_selection=args.mouse_select,
        auto_detect_table=not args.no_auto_detect,
        confidence=args.confidence,
        class_id=detector_class_id,
        min_size=detector_min_size,
        iou_threshold=detector_iou,
        max_jump=cfg_filter.get("max_jump", 100),
        max_missing_frames=cfg_filter.get("max_missing_frames", 2),
        y_reversal_threshold=cfg_landing.get("y_reversal_threshold", 5),
        min_fall_distance=cfg_landing.get("min_fall_distance", 20),
        persp_output_width=cfg_persp.get("output_width", 800),
        persp_output_height=cfg_persp.get("output_height", 600),
        default_corners=persp_default_corners,
    )

    # 處理影片
    try:
        landings = analyzer.process_video(
            show_preview=not args.no_preview,
            save_output=True,
            max_frames=args.max_frames
        )

        print("\n分析完成！")

    except Exception as e:
        print(f"\n錯誤：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()