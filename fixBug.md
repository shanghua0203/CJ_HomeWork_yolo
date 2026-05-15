# BUG

## 跳針的落點偵測
-  請修改 landing_detector.py。當我們確認並記錄了一個落點後，必須將該點標記為『已處理』，或清空分析暫存區，絕對不能在後續的 Frame 重複記錄同一個落點。

## 飛出宇宙的鳥瞰圖座標
- 透視變換矩陣的基準點順序似乎錯亂，導致輸出極端大數與負數。請在程式碼中強制確保四個點是嚴格的『順時針』排序（左上 -> 右上 -> 右下 -> 左下），再進行轉換計算。

## 修復尺寸問題
```
[ WARN:0@88.579] global cap_ffmpeg.cpp:218 write FFmpeg: Failed to write frame
[ WARN:0@88.636] global cap_ffmpeg.cpp:218 write FFmpeg: Failed to write frame
```
