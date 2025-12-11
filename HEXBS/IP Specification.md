# **Hexagon-Based Search (HEXBS) Motion Estimation Accelerator Specification**

**Version:** 1.0

**Author:** HSUN

**Date:** 2025-12-11

## **1\. 概述 (General Description)**

本設計為一款針對視訊編碼 (Video Coding) 應用所開發的運動估計 (Motion Estimation) 硬體加速器。採用 **六角形搜尋演算法 (Hexagon-Based Search, HEXBS)**，相較於全搜尋演算法 (Full Search)，在保持影像品質的同時，顯著降低了運算複雜度與搜尋時間。

本 IP Core 採用全數位化邏輯設計，具備獨立的控制單元 (Control Unit) 與資料路徑 (Datapath)，並支援標準 SRAM 介面存取影像數據。

## **2\. 主要特點 (Key Features)**

* **演算法優化**：實現標準 HEXBS 演算法，包含大六角 (Large Hexagon) 粗搜尋與小六角 (Small Hexagon) 細搜尋機制。  
* **區塊大小 (Block Size)**：支援 16x16 巨集區塊 (Macroblock) 運算。  
* **搜尋範圍 (Search Range)**：由外部記憶體配置決定，內部資料路徑支援 \-32 \~ \+31 像素偏移。  
* **單週期處理**：SAD 計算單元採用管線化設計，每個時脈週期處理一個像素差值。  
* **介面標準**：簡單的握手協定 (Start/Done)，易於整合至 SoC 系統。

## **3\. I/O 腳位定義 (Pin Description)**

### **3.1 系統訊號 (System Signals)**

| 訊號名稱 (Signal Name) | 方向 (Direction) | 寬度 (Width) | 描述 (Description) |
| :---- | :---- | :---- | :---- |
| clk | Input | 1 | 系統時脈 (System Clock)，上升緣觸發。 |
| rst\_n | Input | 1 | 非同步重置訊號 (Active Low Reset)。 |
| i\_start | Input | 1 | 模組啟動訊號，高電位有效。 |
| o\_done | Output | 1 | 運算完成訊號，當此訊號為 High 時，輸出結果有效。 |

### **3.2 記憶體介面 (Memory Interface)**

| 訊號名稱 (Signal Name) | 方向 (Direction) | 寬度 (Width) | 描述 (Description) |
| :---- | :---- | :---- | :---- |
| o\_cur\_x | Output | 6 | 目前區塊 (Current Block) 讀取 X 座標 (0\~15)。 |
| o\_cur\_y | Output | 6 | 目前區塊 (Current Block) 讀取 Y 座標 (0\~15)。 |
| i\_cur\_pixel | Input | 8 | 從記憶體讀回的目前區塊像素值 (Luma 8-bit)。 |
| o\_ref\_x | Output | 12 | 參考畫面 (Reference Frame) 讀取 X 座標。 |
| o\_ref\_y | Output | 12 | 參考畫面 (Reference Frame) 讀取 Y 座標。 |
| i\_ref\_pixel | Input | 8 | 從記憶體讀回的參考畫面像素值 (Luma 8-bit)。 |

### **3.3 運算結果 (Result Outputs)**

| 訊號名稱 (Signal Name) | 方向 (Direction) | 寬度 (Width) | 描述 (Description) |
| :---- | :---- | :---- | :---- |
| o\_mv\_x | Output | 6 | 最終運動向量 X 分量 (Signed, 2's complement)。 |
| o\_mv\_y | Output | 6 | 最終運動向量 Y 分量 (Signed, 2's complement)。 |
| o\_min\_sad | Output | 16 | 最佳匹配區塊的絕對誤差總和 (Minimum SAD)。 |

## **4\. 功能描述 (Functional Description)**

本設計之操作流程由有限狀態機 (FSM) 控制，分為以下階段：

1. **初始化 (Initialization)**：  
   * 系統重置後進入 IDLE 狀態。  
   * 接收到 i\_start 後，首先計算原點 (0,0) 之 SAD。  
2. **大六角搜尋 (LHP Search)**：  
   * 依序計算周圍 6 個特定搜尋點的 SAD。  
   * 搜尋點樣式：(0,-2), (1,-2), (2,0), (1,2), (-1,2), (-2,0) (示意)。  
   * **決策**：  
     * 若最小 SAD 發生在中心點，則進入細搜尋階段 (SHP)。  
     * 若最小 SAD 發生在周圍點，則將中心移動至該點，並重複大六角搜尋。  
3. **小六角搜尋 (SHP Search)**：  
   * 針對最終中心點周圍的 4 個鄰近點進行精細比對。  
   * 搜尋點樣式：(0,-1), (1,0), (0,1), (-1,0)。  
4. **結果輸出 (Output)**：  
   * 鎖存最佳 Motion Vector 與 Min SAD。  
   * 拉高 o\_done 訊號通知外部主機讀取結果。

## **5\. 效能估算 (Performance Estimation)**

* **運算週期 (Cycle Count)**：  
  * 單一搜尋點計算成本：256 Cycles (16x16 pixels) \+ Pipeline Latency。  
  * 總週期數 \= (搜尋點總數) × 256。  
  * HEXBS 平均搜尋點數約為 10\~15 點，遠低於 Full Search 的 (2R+1)² 點。  
* **邏輯閘計數 (Gate Count)**：  
  * (預估值) 約 3k \~ 5k Gates (不含記憶體)，屬於極輕量級硬體設計。