# Bug 修復記錄

## Phase 1 - 崩潰性 Bug

### 1. `main.py` — output_frame 未初始化（2026-06-21）
- **症狀**：若第一幀就偵測到球，`draw_detection_box` 引用未定義的 `output_frame`，拋出 `NameError`
- **原因**：繪製偵測框的程式碼在 `draw_trajectory` 之前，而前者需要後者產生的 `output_frame`
- **修復**：交換繪製順序，先畫軌跡再疊偵測框

### 2. `perspective.py` — dst_points 順序錯亂（2026-06-21）
- **症狀**：鳥瞰圖扭曲或水平翻轉（即「飛出宇宙的鳥瞰圖座標」的根本原因）
- **原因**：`TableCorners.to_array()` 回傳 `[TL, TR, BR, BL]`，但 `_calculate_transform_matrix` 的 `dst_points` 為 `[TL, TR, BL, BR]`，映射順序不匹配
- **修復**：將 `dst_points` 改為 `[TL, TR, BR, BL]` 以匹配 src 順序

### 3. `main.py` — 輸出高度未確保偶數（2026-06-21）
- **症狀**：FFmpeg 寫入失敗 `Failed to write frame`
- **原因**：`output_w` 有 `// 2 * 2` 但 `output_h` 沒有，奇數高度觸發 FFmpeg 限制
- **修復**：兩者皆用 `(x + 1) // 2 * 2` 確保偶數

### 4. `main.py` — Filter 使用寫死解析度（2026-06-21）
- **症狀**：若影片非 1920x1080，filter 的邊界檢查完全失靈
- **原因**：`TrajectoryFilter` 初始化時使用 `__init__` 傳入的寫死值，而非實際影片尺寸
- **修復**：讀取影片後覆寫 `filter.frame_width` 與 `filter.frame_height`

---

## Phase 2 - 邏輯錯誤

### 5. `perspective.py` — 滑鼠點選未排序角落（2026-06-21）
- **症狀**：使用者若未嚴格依序點選角落，透視變換結果完全錯誤
- **原因**：`MouseCornerSelector.select_corners()` 直接按點擊順序填入，未呼叫 `sort_four_points`
- **修復**：建立 `TableCorners` 前先排序

### 6. `landing_detector.py` — _processed_index 機制缺陷（2026-06-21）
- **症狀**：後續落點被跳過未偵測
- **原因**：`_processed_index` 基於每次傳入的切片軌跡長度，但主迴圈每幀傳入不同切片，索引無法對齊
- **修復**：移除此機制，由 `main.py` 的空間去重 (`is_dup`) 統一處理

### 7. `requirements.txt` — opencv 重複依賴（2026-06-21）
- **症狀**：`opencv-python` 與 `opencv-python-headless` 衝突
- **修復**：移除 `opencv-python-headless`

---

## Phase 3 - 程式碼品質

### 8. `filter.py` — interpolate_missing 型別標示不準確
- 回傳值含 `None` 但標示為 `List[Tuple[int,int]]`
- 修正為 `List[Optional[Tuple[int,int]]]`

### 9. `main.py` — 未使用的 import
- `smooth_trajectory`、`connect_trajectory_gaps` 已 import 但從未使用
- 移除

### 10. `main.py` — interpolate_missing 未被呼叫
- 函數已實作但主流程跳過此步驟
- 加入軌跡補間流程

### 11. 全域 — config.yaml 未被載入
- 程式完全忽略 `config/config.yaml`
- 新增 `load_config()` 函數並整合至 `main()`

---

## Phase 4 - 架構設計債

### 12. `tracker.py` — 軌跡無上限增長
- 新增 `max_trajectory_length`（預設 5000）自動截斷舊資料

### 13. `main.py` — 角落重新偵測無驗證
- 加入面積比 sanity check（0.3~3.0），防止異常輪廓覆蓋正確角落

### 14. `perspective.py` — sort_four_points Y 軸分組不穩健
- 改用 `(x+y)/(x-y)` 排除法，從全部點中分別選出極值，再從剩餘兩點區分 TR/BL

### 15. `landing_detector.py` — Y 軸閾值固定
- 改為自適應閾值 `max(5, frame_height // 200)`，根據畫面高度動態調整
