// ============================================================================
// File Name: ME_HEXBS.v
// Author: HSUN's AI Assistant
// Description: Hexagon-Based Search (HEXBS) Motion Estimation Accelerator
//              專為 HSUN 設計的 HEXBS 硬體加速器
// ============================================================================

module ME_HEXBS (
    input  wire         clk,        // 時脈
    input  wire         rst_n,      // 重置訊號 (低電位重置)
    input  wire         i_start,    // 啟動訊號：告訴硬體「開始工作！」

    // --- 虛擬記憶體介面 (Memory Interface) ---
    // 為了簡單起見，我們假設外部有一個記憶體存著 Current Frame 和 Reference Frame
    // 硬體給出 (X, Y) 座標，外部回傳該座標的 Pixel 值
    
    // 索取 Current Frame 的像素
    output wire [5:0]   o_cur_x,    // Current block 內的 x (0~15)
    output wire [5:0]   o_cur_y,    // Current block 內的 y (0~15)
    input  wire [7:0]   i_cur_pixel,// 讀回來的像素值

    // 索取 Reference Frame 的像素
    output wire [11:0]  o_ref_x,    // Reference frame 的絕對 x (包含搜尋範圍)
    output wire [11:0]  o_ref_y,    // Reference frame 的絕對 y
    input  wire [7:0]   i_ref_pixel,// 讀回來的像素值

    // --- 輸出結果 (Result) ---
    output reg  signed [5:0] o_mv_x,    // 最終 Motion Vector X (-32 ~ 31)
    output reg  signed [5:0] o_mv_y,    // 最終 Motion Vector Y (-32 ~ 31)
    output reg  [15:0]       o_min_sad, // 最小 SAD 值
    output reg               o_done     // 完成訊號：高電位表示算完了
);

    // =========================================================
    // 1. 參數定義 (Parameters)
    // =========================================================
    // 狀態機定義 (FSM States)
    localparam S_IDLE       = 4'd0; // 發呆
    localparam S_INIT_00    = 4'd1; // 先算中心點 (0,0) 的 SAD
    localparam S_LHP_PICK   = 4'd2; // 挑選大六角 (LHP) 的下一個點
    localparam S_LHP_CALC   = 4'd3; // 計算 LHP 某點的 SAD
    localparam S_LHP_DECIDE = 4'd4; // LHP 算完一輪，決定要移動中心還是進入小六角
    localparam S_SHP_PICK   = 4'd5; // 挑選小六角 (SHP) 的下一個點
    localparam S_SHP_CALC   = 4'd6; // 計算 SHP 某點的 SAD
    localparam S_DONE       = 4'd7; // 收工

    // =========================================================
    // 2. 內部暫存器 (Internal Registers)
    // =========================================================
    reg [3:0]  state, next_state;

    // 搜尋中心與候選點
    reg signed [5:0] center_x, center_y;    // 目前的大六角中心
    reg signed [5:0] cand_x, cand_y;        // 現在正在算的這一點 (Candidate)
    
    // 最佳紀錄保持人
    reg [15:0] global_min_sad;              // 目前找到最小的 SAD
    reg signed [5:0] best_vec_x, best_vec_y;// 對應的 MV

    // SAD 計算用計數器
    reg [4:0]  pixel_cnt_x; // 0~15
    reg [4:0]  pixel_cnt_y; // 0~15
    reg [15:0] current_accum_sad; // 累加中的 SAD

    // 六角形點數計數器
    reg [2:0]  point_idx;   // 0~5 for LHP, 0~3 for SHP

    // =========================================================
    // 3. 絕對值計算函數 (Abs Function)
    // =========================================================
    function [7:0] abs_diff;
        input [7:0] a, b;
        begin
            abs_diff = (a > b) ? (a - b) : (b - a);
        end
    endfunction

    // =========================================================
    // 4. 主要狀態機 (FSM)
    // =========================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    // 狀態轉移邏輯
    always @(*) begin
        case (state)
            S_IDLE: 
                if (i_start) next_state = S_INIT_00;
                else         next_state = S_IDLE;

            S_INIT_00: 
                // 等 (0,0) 這一點的 256 個 pixel 算完
                if (pixel_cnt_x == 15 && pixel_cnt_y == 15) next_state = S_LHP_PICK;
                else next_state = S_INIT_00;

            S_LHP_PICK:
                // 準備去算大六角的點
                next_state = S_LHP_CALC;

            S_LHP_CALC:
                // 等這一點算完
                if (pixel_cnt_x == 15 && pixel_cnt_y == 15) begin
                    if (point_idx == 5) next_state = S_LHP_DECIDE; // 6點都算完，去決策
                    else                next_state = S_LHP_PICK;   // 還沒，換下一點
                end else begin
                    next_state = S_LHP_CALC;
                end

            S_LHP_DECIDE:
                // 你的 Python 邏輯：如果最佳點還是在中心 (0,0)，就去 SHP，否則移動中心重跑 LHP
                if (best_vec_x == center_x && best_vec_y == center_y)
                    next_state = S_SHP_PICK;
                else
                    next_state = S_LHP_PICK; // 注意：這裡簡化處理，標準 HEXBS 會重跑，這裡我們簡單地回到 PICK 狀態（需配合 Datapath 重置 point_idx）

            S_SHP_PICK:
                next_state = S_SHP_CALC;

            S_SHP_CALC:
                if (pixel_cnt_x == 15 && pixel_cnt_y == 15) begin
                    if (point_idx == 3) next_state = S_DONE; // 4點都算完
                    else                next_state = S_SHP_PICK;
                end else begin
                    next_state = S_SHP_CALC;
                end

            S_DONE:
                next_state = S_IDLE; // 這裡可以停住，也可以回 IDLE

            default: next_state = S_IDLE;
        endcase
    end

    // =========================================================
    // 5. 資料路徑 (Datapath) - 這裡做真正的運算
    // =========================================================
    
    // 記憶體地址輸出
    assign o_cur_x = pixel_cnt_x;
    assign o_cur_y = pixel_cnt_y;
    // Ref 地址 = 中心 + 偏移(MV) + 像素位置
    // 注意：這裡做了簡單的運算，實際電路可能需要處理負數邊界 Clip，這裡假設 Memory 會處理
    assign o_ref_x = 100 + cand_x + pixel_cnt_x; // 假設 Reference Frame 中心在 (100,100) 以避免負數
    assign o_ref_y = 100 + cand_y + pixel_cnt_y;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            center_x <= 0; center_y <= 0;
            global_min_sad <= 16'hFFFF; // 初始最大值
            o_done <= 0;
            point_idx <= 0;
            pixel_cnt_x <= 0; pixel_cnt_y <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    o_done <= 0;
                    center_x <= 0; center_y <= 0;
                    global_min_sad <= 16'hFFFF;
                    o_mv_x <= 0; o_mv_y <= 0;
                end

                // --- 步驟 1: 算中心點 (0,0) ---
                S_INIT_00: begin
                    cand_x <= 0; cand_y <= 0; // 設定候選人是 (0,0)
                    
                    // 累加 SAD
                    if (pixel_cnt_x == 0 && pixel_cnt_y == 0) current_accum_sad <= abs_diff(i_cur_pixel, i_ref_pixel);
                    else current_accum_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);

                    // 像素計數器
                    if (pixel_cnt_x == 15) begin
                        pixel_cnt_x <= 0;
                        if (pixel_cnt_y == 15) begin
                            // 算完了 (0,0)！更新最小值
                            global_min_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);
                            best_vec_x <= 0; best_vec_y <= 0;
                            pixel_cnt_y <= 0;
                            point_idx <= 0; // 重置給下一個狀態用
                        end else pixel_cnt_y <= pixel_cnt_y + 1;
                    end else pixel_cnt_x <= pixel_cnt_x + 1;
                end

                // --- 步驟 2: 大六角 (LHP) 準備 ---
                S_LHP_PICK: begin
                    pixel_cnt_x <= 0; pixel_cnt_y <= 0;
                    current_accum_sad <= 0;
                    
                    // Python: LHP = [(2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2)]
                    case (point_idx)
                        0: begin cand_x <= center_x + 2; cand_y <= center_y + 0; end
                        1: begin cand_x <= center_x + 1; cand_y <= center_y + 2; end
                        2: begin cand_x <= center_x - 1; cand_y <= center_y + 2; end
                        3: begin cand_x <= center_x - 2; cand_y <= center_y + 0; end
                        4: begin cand_x <= center_x - 1; cand_y <= center_y - 2; end
                        5: begin cand_x <= center_x + 1; cand_y <= center_y - 2; end
                    endcase
                end

                // --- 步驟 3: 計算該點 SAD ---
                S_LHP_CALC: begin
                    // 邏輯同 S_INIT_00，計算 16x16
                    if (pixel_cnt_x == 0 && pixel_cnt_y == 0) current_accum_sad <= abs_diff(i_cur_pixel, i_ref_pixel);
                    else current_accum_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);

                    if (pixel_cnt_x == 15) begin
                        pixel_cnt_x <= 0;
                        if (pixel_cnt_y == 15) begin
                            // 算完了！檢查有沒有比 global_min_sad 小
                            pixel_cnt_y <= 0;
                            if ((current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel)) < global_min_sad) begin
                                global_min_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);
                                best_vec_x <= cand_x;
                                best_vec_y <= cand_y;
                            end
                            
                            // 準備算下一個點
                            if (point_idx < 6) point_idx <= point_idx + 1;
                        end else pixel_cnt_y <= pixel_cnt_y + 1;
                    end else pixel_cnt_x <= pixel_cnt_x + 1;
                end

                // --- 步驟 4: 決策 ---
                S_LHP_DECIDE: begin
                    point_idx <= 0; // 重置
                    if (best_vec_x != center_x || best_vec_y != center_y) begin
                        // 最小值不在中心 -> 中心移動！再跑一次 LHP
                        center_x <= best_vec_x;
                        center_y <= best_vec_y;
                        // 注意：這裡下一狀態會回到 S_LHP_PICK，point_idx 歸零，重新對新中心周圍掃描
                    end
                end

                // --- 步驟 5: 小六角 (SHP) 準備 ---
                S_SHP_PICK: begin
                    pixel_cnt_x <= 0; pixel_cnt_y <= 0;
                    current_accum_sad <= 0;
                    // Python: SHP = [(1,0), (0,1), (-1,0), (0,-1)]
                    case (point_idx)
                        0: begin cand_x <= center_x + 1; cand_y <= center_y + 0; end
                        1: begin cand_x <= center_x + 0; cand_y <= center_y + 1; end
                        2: begin cand_x <= center_x - 1; cand_y <= center_y + 0; end
                        3: begin cand_x <= center_x + 0; cand_y <= center_y - 1; end
                    endcase
                end

                // --- 步驟 6: 計算 SHP SAD ---
                S_SHP_CALC: begin
                    // 邏輯同上
                    if (pixel_cnt_x == 0 && pixel_cnt_y == 0) current_accum_sad <= abs_diff(i_cur_pixel, i_ref_pixel);
                    else current_accum_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);

                    if (pixel_cnt_x == 15) begin
                        pixel_cnt_x <= 0;
                        if (pixel_cnt_y == 15) begin
                            // 更新最小值
                            pixel_cnt_y <= 0;
                            if ((current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel)) < global_min_sad) begin
                                global_min_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);
                                best_vec_x <= cand_x;
                                best_vec_y <= cand_y;
                            end
                            if (point_idx < 4) point_idx <= point_idx + 1;
                        end else pixel_cnt_y <= pixel_cnt_y + 1;
                    end else pixel_cnt_x <= pixel_cnt_x + 1;
                end

                // --- 結束 ---
                S_DONE: begin
                    o_done <= 1;
                    o_mv_x <= best_vec_x;
                    o_mv_y <= best_vec_y;
                    o_min_sad <= global_min_sad;
                end
            endcase
        end
    end

endmodule