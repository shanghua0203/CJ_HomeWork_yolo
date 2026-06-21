# 乒乓球落點分析系統

使用 YOLO 影像辨識技術，自動偵測並分析乒乓球軌跡與落點。

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Tests](https://img.shields.io/badge/Tests-160%20passed-green)
![License](https://img.shields.io/badge/License-Academic-red)

---

## 功能特色

| 功能 | 說明 |
|------|------|
| YOLO 物件偵測 | 使用 Ultralytics YOLOv8 偵測乒乓球 |
| 軌跡追蹤 | 持續追蹤球的位置，含掉幀處理 |
| 雜訊過濾 | 濾除異常座標，保持軌跡平滑 |
| 落點判斷 | 分析 Y 軸變化，找出反彈點 |
| 透視變換 | 將斜角畫面轉換為 2D 鳥瞰圖 |
| 視覺化輸出 | 支援原視角與鳥瞰圖雙視圖顯示 |

---

## 專案結構

```
yolo/
├── _doc/                      # 開發文件
│   ├── v0.1.md               # 開發計畫
│   └── v0.2.md               # 版本記錄
│
├── src/                       # 原始碼 (~3100 行)
│   ├── detector.py           # YOLO 偵測模組 (471 行)
│   ├── filter.py             # 雜訊過濾模組 (329 行)
│   ├── tracker.py            # 軌跡追蹤模組 (478 行)
│   ├── landing_detector.py   # 落點判斷模組 (221 行)
│   ├── perspective.py        # 透視變換模組 (530 行)
│   ├── visualizer.py         # 視覺化輸出模組 (620 行)
│   └── main.py               # 主程式入口 (536 行)
│
├── tests/                     # 測試程式碼 (160 個測試)
│   ├── test_detector.py      # 13 個測試
│   ├── test_filter.py        # 28 個測試
│   ├── test_tracker.py       # 26 個測試
│   ├── test_landing_detector.py  # 29 個測試
│   ├── test_perspective.py   # 23 個測試
│   ├── test_visualizer.py    # 23 個測試
│   └── test_system.py        # 18 個測試
│
├── config/
│   └── config.yaml           # 主設定檔（程式啟動時自動載入）
│
├── fixBug.md                  # 已知錯誤修復記錄
├── requirements.txt           # 依賴套件
├── README.md                  # 本文件
├── task.md                    # 開發任務指示書
└── .venv/                     # 虛擬環境
```

---

## 快速開始

### 1. 安裝依賴

```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境 (Linux/Mac)
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt
```

### 2. 執行測試

```bash
# 執行所有測試
pytest tests/ -v

# 執行特定模組測試
pytest tests/test_detector.py -v
pytest tests/test_landing_detector.py -v
```

### 3. 執行分析

```bash
# 基本用法
python src/main.py data/sample_video.mp4

# 使用滑鼠點選桌面角落
python src/main.py data/sample_video.mp4 --mouse-select

# 不顯示預覽（離線模式）
python src/main.py data/sample_video.mp4 --no-preview

# 自訂參數
python src/main.py data/sample_video.mp4 \
  --confidence 0.6 \
  --output my_results \
  --max-frames 100
```

---

## 命令列參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `video` | 輸入影片路徑 | `data/sample_video.mp4` |
| `--model` | YOLO 模型路徑 | `yolov8n.pt` |
| `--output` | 輸出資料夾 | `output` |
| `--confidence` | 偵測信心閾值 (0.0~1.0) | `0.01` |
| `--max-frames` | 最大處理幀數 | `None` (全部) |
| `--mouse-select` | 使用滑鼠點選角落 | `False` |
| `--no-auto-detect` | 停用自動偵測角落 | `False` |
| `--no-preview` | 不顯示預覽視窗 | `False` |

> 註：預設信心值 `0.01` 是因為 YOLOv8n 對快速移動小球的信心值通常極低，`config.yaml` 可覆蓋此值。

---

## 設定檔

程式啟動時自動讀取 `config/config.yaml`，覆蓋命令列未指定的參數：

```yaml
# === YOLO 偵測設定 ===
detector:
  confidence: 0.5       # 信心閾值
  min_size: 10          # 最小偵測框
  class_id: 32          # 類別編號 (COCO: 32 = sports ball)

# === 雜訊過濾設定 ===
filter:
  max_jump: 100         # 最大位移閾值 (pixels)
  max_missing_frames: 2 # 掉幀容忍度

# === 落點判斷設定 ===
landing:
  y_reversal_threshold: 5   # 反彈偵測閾值
  min_fall_distance: 20     # 最小下落距離

# === 視角轉換設定 ===
perspective:
  output_width: 800     # 鳥瞰圖寬度
  output_height: 600    # 鳥瞰圖高度
```

---

## 模組架構

### detector.py - 乒乓球偵測
使用 Ultralytics YOLO 模型偵測乒乓球，輸出中心座標 (x, y)。
```
輸入: frame (影像)
輸出: (x, y) 或 None (含完整偵測框與信心值的版本)
```

### filter.py - 雜訊過濾
- 濾除超出畫面範圍的座標（使用實際影片解析度）
- 濾除瞬間跳動過大的點（預設 >100 pixels）
- 線性插值補間掉幀
- Z-score 離群值移除
- 移動平均平滑軌跡

### tracker.py - 軌跡追蹤
- 持續追蹤球的位置
- 掉幀處理（容忍 1-2 幀消失）
- 軌跡點上限自動截斷（預設 5000 點，防止記憶體無限制增長）
- 統計資訊計算（速度、Y 軸範圍）

### landing_detector.py - 落點判斷
- 分析 Y 軸變化趨勢
- 自適應反轉閾值（根據畫面高度動態調整）
- 偵測反彈點（Y 軸變大後突然變小）
- 找出反彈前的最低點（落點位置）
- 驗證落點有效性
```
Y軸變大（下落）→ Y軸變小（反彈）→ 最低點 = 落點
```

### perspective.py - 透視變換
- Canny 邊緣偵測自動尋找桌面四邊形
- 滑鼠手動點選角落（附自動排序校正）
- 每 150 幀重新偵測角落，附面積比驗證（防止異常值覆蓋）
- 使用 (x+y)/(x-y) 組合的排序演算法（比 Y 軸分組更穩健）
- 斜角 → 2D 鳥瞰圖轉換，反向轉換支援

### visualizer.py - 視覺化輸出
- 原視角：軌跡線 + 偵測框 + 落點標記（含編號）+ 資訊面板
- 鳥瞰圖：桌面底圖 + 網格 + 落點
- 多視圖組合（左右併排 / 上下堆疊）

### main.py - 主程式入口
- 解析命令列參數 + 自動載入 `config.yaml`
- 串接完整分析流程：讀取影片 → 偵測 → 補間 → 過濾 → 追蹤 → 落點判斷 → 視角轉換 → 視覺化輸出
- 輸出 `result.mp4` + `landing_report.txt`

---

## 操作說明

### 預覽視窗控制
| 按鍵 | 功能 |
|------|------|
| `q` | 退出分析 |
| `p` | 暫停/繼續 |

### 滑鼠點選角落順序
點選時不需嚴格依序，系統會自動排序校正。
```
1. 依序點擊桌面四個角落（順序不拘）
2. 按 Enter 確認
3. 按 r 重新選取
4. 按 q 取消
```

---

## 測試覆蓋

| 模組 | 測試數 | 狀態 |
|------|--------|------|
| detector | 13 | ✅ |
| filter | 28 | ✅ |
| tracker | 26 | ✅ |
| landing_detector | 29 | ✅ |
| perspective | 23 | ✅ |
| visualizer | 23 | ✅ |
| system | 18 | ✅ |
| **總計** | **160** | **✅ 100%** |

---

## 技術規格

| 項目 | 規格 |
|------|------|
| Python 版本 | 3.8+ |
| YOLO 模型 | yolov8n.pt (Ultralytics) |
| OpenCV | 4.8.0+ |
| 測試框架 | pytest 7.4.0+ |
| 原始碼行數 | ~3100 行 |
| 測試總行數 | ~2300 行 |

---

## Bug 修復記錄

詳細修復歷程記錄於 `fixBug.md`，包含以下關鍵修復：

| 問題 | 狀態 | 說明 |
|------|------|------|
| 跳針的落點偵測 | 已修復 | 移除 `_processed_index` 機制，由主迴圈的空間去重統一處理 |
| 鳥瞰圖座標異常 | 已修復 | 修正 `dst_points` 順序匹配 `to_array()`、改善 `sort_four_points` 演算法 |
| FFmpeg 寫入失敗 | 已修復 | `output_w` 與 `output_h` 皆確保偶數 |
| `output_frame` 未初始化 | 已修復 | 交換繪製順序，先畫軌跡再疊偵測框 |
| 角落無驗證機制 | 已修復 | 每 150 幀重新偵測時加入面積比範圍檢查 (0.3~3.0) |

---

## 輸出範例

分析完成後會產生：

### 結果影片
`output/result.mp4` - 左右併排視圖（原視角 + 鳥瞰圖）

### 落點報告
`output/landing_report.txt`

```
乒乓球落點分析報告
========================================

輸入影片：data/sample_video.mp4
偵測到落點數量：3 個

落點 1:
  座標：(x=320, y=400)
  幀編號：45
  最低 Y 值：400
  鳥瞰圖座標：(x=400, y=300)

落點 2:
  座標：(x=280, y=380)
  幀編號：120
  最低 Y 值：380
  鳥瞰圖座標：(x=350, y=280)
...
```

---

## 資料夾用途

| 資料夾 | 用途 |
|--------|------|
| `_doc/` | 開發文件、版本記錄 |
| `src/` | 所有模組原始碼 |
| `tests/` | pytest 單元測試 |
| `data/` | 測試用影片 |
| `config/` | 參數設定檔 |
| `output/` | 分析結果輸出 |

---

## 授權

本專案僅供學術研究使用。

---

*最後更新：v0.2 - 2026-05-14*
