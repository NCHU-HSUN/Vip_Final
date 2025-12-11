# **單元庫設計流程 (Cell-Based Design Flow) 戰鬥指南**

## **1\. 核心概念：什麼是 Cell-Based Design?**

想像你要蓋一棟房子：

* **Full Custom (全客製化):** 你親自燒磚頭、鋸木頭，每一塊磚形狀都不一樣。效能最好，但最累。  
* **Cell-Based (單元庫設計):** 你去 IKEA 買現成的標準家具（Standard Cells，如 AND, OR, D-FlipFlop），然後照著設計圖把家具擺好。這就是現在業界的主流做法。

## **2\. 流程總覽 (Front-end vs. Back-end)**

整個流程可以切成兩半，中間那條虛線是關鍵分水嶺。

* **前端 (Front-end):** 重點在「邏輯對不對」。(程式碼能不能跑？)  
* **後端 (Back-end):** 重點在「物理行不行」。(電路擺得進去嗎？線接得起來嗎？)

## **3\. 詳細步驟拆解**

### **階段一：演算法與架構 (Algorithm Development)**

* **做什麼：** 這是大腦。先不管硬體，先確認數學是對的。  
* **工具：** **Python** / C++  
* **你的任務：** 寫出 Motion Estimation 的 Python code，產出 input.txt 和 golden\_output.txt。

### **階段二：RTL 編碼 (RTL Coding)**

* **做什麼：** 這是翻譯。把 Python 邏輯翻譯成硬體語言 (Verilog)。  
* **工具：** 任何文字編輯器 (VS Code, Vim)。  
* **輸入：** 你的腦袋 \+ Python code。  
* **輸出：** Verilog 原始碼 (.v 檔)。

### **階段三：功能驗證 (Functional Verification)**

* **做什麼：** 這是除錯。跑模擬確認 Verilog 算出來的跟 Python 一樣。  
* **工具：** **VCS** (跑模擬), **Verdi / nWave** (看波形)。  
* **關鍵動作：** 如果結果不對 (Fail)，就要回去改 Verilog (Loop back to RTL Coding)。

### **階段四：邏輯合成 (Logic Synthesis)**

* **做什麼：** 這是轉譯。把你看得懂的 Verilog (if-else)，轉成電腦看得懂的邏輯閘清單 (AND, OR, MUX)，這個清單叫 **Netlist**。  
* **工具：** **Design Compiler (DC)** (Synopsys 公司的招牌工具)。  
* **軟體工程師類比：** 這就像是把 C 語言 Compile 成 Assembly。  
* **輸出：** Gate-level Netlist (.v 檔), 面積報告, 速度報告。

### **階段五：合成後驗證 (Gate-level Verification)**

* **做什麼：** 再次確認。確保 DC 轉譯的過程中沒有把邏輯搞壞。  
* **工具：** 同樣使用 **VCS** \+ **Verdi**。

## **(由此跨過虛線，進入後端 Back-end)**

### **階段六：自動佈局與繞線 (Auto Place & Route, APR)**

* **做什麼：** 這是裝潢。把剛剛合成出來的幾萬個邏輯閘 (Gate)，實際擺放到晶片的正方形空間裡 (Place)，並把電線連起來 (Route)。  
* **工具：** **Innovus** (Cadence 公司) 或 **ICC2** (Synopsys 公司)。  
* **關鍵動作：**  
  * 如果擺不下？回去重寫 Code 或放寬限制。  
  * 如果線太長跑不快？工具會自動優化。

### **階段七：時序與功耗分析 (Timing/Power Analysis)**

* **做什麼：** 這是健檢。  
  * **Timing:** 訊號能不能在時脈 (Clock) 到達前跑完？(Setup Time Check)。  
  * **Power:** 這顆晶片會不會太熱？  
* **工具：** **PrimeTime (PT)**, **PTPX**。

### **階段八：實體驗證 (Physical Verification)**

* **做什麼：** 這是安檢。檢查有沒有違反物理規則 (例如兩條電線靠太近會短路)。  
* **專有名詞：**  
  * **DRC:** Design Rule Check (設計規則檢查)。  
  * **LVS:** Layout Vs Schematic (電路圖與佈局圖比對)。  
* **狀態：** 如果 Pass，就可以送去台積電生產了 (Tape-out)。

## **4\. 軟體 vs. 硬體名詞對照表 (給 HSUN 專用)**

| 軟體開發 (Software) | 硬體設計 (ASIC Flow) | 備註 |
| :---- | :---- | :---- |
| **Source Code** (Python/C) | **RTL** (Verilog) | 都是原始碼 |
| **Compilation** (gcc) | **Synthesis** (Design Compiler) | 編譯/合成 |
| **Assembly / Binary** | **Netlist** (Gate-level) | 機器才看得懂的底層 |
| **Unit Test** | **Simulation** (VCS/Testbench) | 驗證功能 |
| **Memory Allocation** | **Place & Route** (Innovus) | 決定東西放哪裡 |
| **Performance Profiling** | **Timing Analysis** (PrimeTime) | 檢查跑得快不快 |
| **Release / Deploy** | **Tape-out** | 下線生產 |

