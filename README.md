## 虛擬簡報筆: 基於指向與動作辨識於人機互動 (2023國科會大專生計畫)

使用YOLOv8模型結合MediaPipe Hand進行手部動作和關節點的即時辨識，將辨識的手勢轉換為具體的電腦操作，允許使用者通過手勢控制PowerPoint簡報，進而提升簡報的互動性和使用者體驗。

YOLO進行初步動作辨識，根據MediaPipe關節點做二次檢查來確認動作，同時讓Blender根據MediaPipe關節點去計算指向。

#### 使用套件&軟體:
* Python 3.10
* Blender 3.4
* YOLOv8
* Mediapipe (Hand)
