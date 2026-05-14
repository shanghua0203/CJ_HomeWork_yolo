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
from typing import List, Tuple, Optional

from detector import BallDetector, load_image
from filter import TrajectoryFilter, smooth_trajectory
from tracker import BallTracker, connect_trajectory_gaps
from landing_detector import LandingDetector, LandingPoint
from perspective import (
    PerspectiveTransformer,
    TableCorners,
    MouseCornerSelector,
    auto_detect_table_corners
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
        confidence: float = 0.5
    ):
        """
        初始化分析器

        參數：
        - video_path: 輸入影片路徑
        - model_path: YOLO 模型路徑
        - output_dir: 輸出資料夾
        - use_mouse_selection: 是否使用滑鼠點選桌子角落
        - auto_detect_table: 是否自動偵測桌子角落
        - confidence: 偵測信心閾值
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.use_mouse_selection = use_mouse_selection
        self.auto_detect_table = auto_detect_table
        self.confidence = confidence

        # 建立輸出資料夾
        os.makedirs(output_dir, exist_ok=True)

        # 初始化各模組
        self.detector = BallDetector(
            model_path=model_path,
            confidence=confidence
        )

        self.filter = TrajectoryFilter(max_jump=100)
        self.tracker = BallTracker(max_missing_frames=2)
        self.landing_detector = LandingDetector(
            y_reversal_threshold=5,
            min_fall_distance=20
        )

        # 透視變換器（稍後設定角落）
        self.transformer = PerspectiveTransformer()

        # 視覺化器
        self.viz_config = VisualizationConfig()
        self.trajectory_viz = TrajectoryVisualizer(config=self.viz_config)
        self.bird_eye_viz: Optional[BirdEyeVisualizer] = None

        # 影片擷取物件
        self.cap = None

        # 分析結果
        self.all_landings: List[LandingPoint] = []

    def setup_perspective(self, frame: np.ndarray):
        """
        設定透視變換

        參數：
        - frame: 包含桌子的參考影像
        """
        corners = None

        # 嘗試自動偵測
        if self.auto_detect_table:
            print("嘗試自動偵測桌面角落...")
            corners = auto_detect_table_corners(frame)

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

        # 建立輸出影片編寫器
        output_path = os.path.join(self.output_dir, "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        # 重置追蹤器
        self.tracker.reset()
        self.all_landings = []

        frame_number = 0
        processed_frames = 0

        print("\n開始處理影片...")

        while True:
            # 讀取下一幀
            ret, frame = self.cap.read()

            if not ret:
                break

            # 偵測乒乓球
            detection = self.detector.detect(frame)

            # 更新追蹤器
            self.tracker.update(detection, frame_number)

            # 取得當前軌跡
            trajectory = self.tracker.get_trajectory()
            frame_indices = self.tracker.get_frame_indices()

            # 過濾軌跡
            filtered_trajectory = self.filter.filter_trajectory(trajectory)

            # 偵測落點
            if len(filtered_trajectory) >= 3:
                landings = self.landing_detector.detect(
                    filtered_trajectory,
                    frame_indices
                )
                self.all_landings.extend(landings)

            # 繪製軌跡
            output_frame = self.trajectory_viz.draw_trajectory(
                frame, filtered_trajectory
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

            # 儲存輸出
            if save_output:
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
        default=0.5,
        help="偵測信心閾值（預設：0.5）"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大處理幀數"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("乒乓球落點分析系統")
    print("=" * 50)

    # 建立分析器
    analyzer = PingPongAnalyzer(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output,
        use_mouse_selection=args.mouse_select,
        auto_detect_table=not args.no_auto_detect,
        confidence=args.confidence
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