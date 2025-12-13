`timescale 1ns/1ps

module hexbs_top (
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // 記憶體介面
    output reg [31:0] mem_addr,
    input wire [7:0]  mem_rdata,
    
    // 參數輸入
    input wire [31:0] frame_start_addr, 
    input wire [31:0] ref_start_addr,
    input wire [31:0] mb_x_pos,         
    input wire [31:0] mb_y_pos,         
    
    // 結果輸出
    output reg signed [5:0] mv_x, 
    output reg signed [5:0] mv_y, 
    output reg [15:0] sad,        
    output reg done               
);
    // 參數設定 (可從外部覆蓋)
    parameter WIDTH = 352;
    parameter HEIGHT = 240;
    parameter SEARCH_RANGE = 32;

    // 搜尋範圍限制
    localparam signed [6:0] SEARCH_MIN = -SEARCH_RANGE;
    localparam signed [6:0] SEARCH_MAX = SEARCH_RANGE;
    
    // 狀態機定義
    localparam S_IDLE        = 0;
    localparam S_LOAD_CUR    = 1;  // 載入當前 Block 到暫存區
    
    // Large Hexagon States
    localparam S_LH_START    = 2;  // 大六角搜尋開始 (設定 Center)
    localparam S_LH_POINT    = 3;  // 設定下一個測試點座標
    localparam S_LH_CALC     = 4;  // 計算該點 SAD
    localparam S_LH_CHECK    = 5;  // 檢查是否更新最佳解
    
    // Small Hexagon States
    localparam S_SH_START    = 6;  // 小六角搜尋開始
    localparam S_SH_POINT    = 7;
    localparam S_SH_CALC     = 8;
    localparam S_SH_CHECK    = 9;
    
    localparam S_DONE        = 10;

    reg [3:0] state;
    reg [7:0] cur_mb_mem [0:255]; // 當前 Macroblock 暫存 (16x16)
    
    // 座標暫存器
    reg signed [6:0] center_x, center_y;       // 目前搜尋中心
    reg signed [6:0] cand_x, cand_y;           // 當前測試點
    reg signed [6:0] best_cand_x, best_cand_y; // 目前已知最佳點
    
    // 計數器
    reg [8:0] pixel_cnt; // 0~255
    reg [3:0] point_idx; // 搜尋點索引 (0~6 or 0~3)
    
    // SAD 計算相關
    reg [15:0] current_accum_sad;
    reg [15:0] min_sad_reg;       
    reg best_updated; // 標記是否在這一輪搜尋中找到了更好的點

    // Pattern Tables (Offsets)
    reg signed [2:0] l_hex_off_x [0:5];
    reg signed [2:0] l_hex_off_y [0:5];
    reg signed [1:0] s_hex_off_x [0:3];
    reg signed [1:0] s_hex_off_y [0:3];
    
    // 計算用的中間變數 (用於計算記憶體地址)
    reg signed [31:0] ref_x_calc, ref_y_calc;

    // =========================================================
    // 初始化搜尋圖形
    // =========================================================
    initial begin
        // Large Hexagon Pattern (LHP)
        l_hex_off_y[0]=2;  l_hex_off_x[0]=0;
        l_hex_off_y[1]=1;  l_hex_off_x[1]=2;
        l_hex_off_y[2]=-1; l_hex_off_x[2]=2;
        l_hex_off_y[3]=-2; l_hex_off_x[3]=0;
        l_hex_off_y[4]=-1; l_hex_off_x[4]=-2;
        l_hex_off_y[5]=1;  l_hex_off_x[5]=-2;
        
        // Small Hexagon Pattern (SHP)
        s_hex_off_y[0]=1;  s_hex_off_x[0]=0;
        s_hex_off_y[1]=0;  s_hex_off_x[1]=1;
        s_hex_off_y[2]=-1; s_hex_off_x[2]=0;
        s_hex_off_y[3]=0;  s_hex_off_x[3]=-1;
    end

    // 絕對值差函數
    function [7:0] abs_diff;
        input [7:0] a, b;
        begin
            abs_diff = (a > b) ? (a - b) : (b - a);
        end
    endfunction
    
    // 邊界檢查 flag
    wire bound_ok;
    assign bound_ok = (cand_x >= SEARCH_MIN && cand_x <= SEARCH_MAX &&
                       cand_y >= SEARCH_MIN && cand_y <= SEARCH_MAX);

    // =========================================================
    // 主狀態機
    // =========================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 0;
            pixel_cnt <= 0;
            point_idx <= 0;
            center_x <= 0; center_y <= 0;
            cand_x <= 0; cand_y <= 0;
            min_sad_reg <= 16'hFFFF;
            mv_x <= 0; mv_y <= 0; sad <= 0;
            best_cand_x <= 0; best_cand_y <= 0; 
            best_updated <= 0;
        end
        else begin
            case (state)
                // --- 閒置狀態 ---
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= S_LOAD_CUR;
                        pixel_cnt <= 0;
                    end
                end

                // --- 載入 Current Block (256 bytes) ---
                S_LOAD_CUR: begin
                    cur_mb_mem[pixel_cnt] <= mem_rdata;
                    
                    // 【核彈級修正】在此狀態的 256 個 Cycle 內，持續強制重置關鍵變數
                    // 這保證了當我們進入 S_LH_START 時，這些數值絕對是乾淨的 0 和 FFFF
                    min_sad_reg <= 16'hFFFF;
                    best_cand_x <= 0; best_cand_y <= 0;
                    center_x <= 0; center_y <= 0;
                    cand_x <= 0; cand_y <= 0;
                    best_updated <= 0;

                    if (pixel_cnt == 255) begin
                        state <= S_LH_START;
                    end else begin
                        pixel_cnt <= pixel_cnt + 1;
                    end
                end

                // --- 大六角搜尋: 開始 ---
                S_LH_START: begin
                    point_idx <= 7; // 特殊標記: 表示正在算 Center
                    cand_x <= center_x;
                    cand_y <= center_y;
                    
                    state <= S_LH_CALC;
                    pixel_cnt <= 0;
                    current_accum_sad <= 0;
                    best_updated <= 0; 
                end

                // --- 大六角搜尋: 設定測試點 ---
                S_LH_POINT: begin
                    cand_x <= center_x + l_hex_off_x[point_idx];
                    cand_y <= center_y + l_hex_off_y[point_idx];
                    
                    state <= S_LH_CALC;
                    pixel_cnt <= 0;
                    current_accum_sad <= 0;
                end

                // --- 通用 SAD 計算狀態 (LH) ---
                S_LH_CALC: begin
                    if (bound_ok) begin
                        current_accum_sad <= current_accum_sad + abs_diff(cur_mb_mem[pixel_cnt], mem_rdata);
                    end else begin
                        current_accum_sad <= 16'hFFFF; // 出界設為最大值
                    end

                    if (pixel_cnt == 255) begin
                        state <= S_LH_CHECK;
                    end else begin
                        pixel_cnt <= pixel_cnt + 1;
                    end
                end

                // --- 大六角搜尋: 檢查結果 ---
                S_LH_CHECK: begin
                    if (current_accum_sad < min_sad_reg) begin
                        min_sad_reg <= current_accum_sad;
                        best_cand_x <= cand_x;
                        best_cand_y <= cand_y;
                        best_updated <= 1; 
                    end
                    
                    if (point_idx == 7) begin
                        point_idx <= 0;
                        state <= S_LH_POINT;
                    end else if (point_idx < 5) begin
                        point_idx <= point_idx + 1;
                        state <= S_LH_POINT;
                    end else begin
                        if (best_updated) begin
                            center_x <= best_cand_x;
                            center_y <= best_cand_y;
                            state <= S_LH_START; // Loop back
                        end else begin
                            state <= S_SH_START;
                        end
                    end
                end

                // --- 小六角搜尋: 開始 ---
                S_SH_START: begin
                    point_idx <= 0;
                    state <= S_SH_POINT;
                end

                S_SH_POINT: begin
                    cand_x <= best_cand_x + s_hex_off_x[point_idx]; 
                    cand_y <= best_cand_y + s_hex_off_y[point_idx];
                    
                    state <= S_SH_CALC;
                    pixel_cnt <= 0;
                    current_accum_sad <= 0;
                end

                S_SH_CALC: begin
                    if (bound_ok) 
                        current_accum_sad <= current_accum_sad + abs_diff(cur_mb_mem[pixel_cnt], mem_rdata);
                    else 
                        current_accum_sad <= 16'hFFFF;

                    if (pixel_cnt == 255) begin
                        state <= S_SH_CHECK;
                    end else begin
                        pixel_cnt <= pixel_cnt + 1;
                    end
                end

                S_SH_CHECK: begin
                    if (current_accum_sad < min_sad_reg) begin
                        min_sad_reg <= current_accum_sad;
                        best_cand_x <= cand_x;
                        best_cand_y <= cand_y;
                    end
                    
                    if (point_idx < 3) begin
                        point_idx <= point_idx + 1;
                        state <= S_SH_POINT;
                    end else begin
                        state <= S_DONE;
                    end
                end

                // --- 完成 ---
                S_DONE: begin
                    mv_x <= best_cand_x[5:0];
                    mv_y <= best_cand_y[5:0];
                    sad  <= min_sad_reg;
                    done <= 1;
                    if (!start) state <= S_IDLE;
                end
            endcase
        end
    end

    // ==========================================
    // 記憶體地址計算組合邏輯 (保持不變)
    // ==========================================
    always @(*) begin
        mem_addr = 0;
        ref_x_calc = 0;
        ref_y_calc = 0;

        case (state)
            S_LOAD_CUR: begin
                mem_addr = frame_start_addr + 
                           (mb_y_pos * WIDTH + mb_x_pos) + 
                           ((pixel_cnt[7:4]) * WIDTH) + pixel_cnt[3:0];
            end
            
            S_LH_CALC, S_SH_CALC: begin
                ref_x_calc = $signed(mb_x_pos) + cand_x;
                ref_y_calc = $signed(mb_y_pos) + cand_y;

                if (ref_x_calc < 0) ref_x_calc = 0;
                else if (ref_x_calc > WIDTH - 16) ref_x_calc = WIDTH - 16;

                if (ref_y_calc < 0) ref_y_calc = 0;
                else if (ref_y_calc > HEIGHT - 16) ref_y_calc = HEIGHT - 16;

                mem_addr = ref_start_addr + 
                           (ref_y_calc * WIDTH + ref_x_calc) + 
                           ((pixel_cnt[7:4]) * WIDTH) + pixel_cnt[3:0];
            end
            
            default: mem_addr = 0;
        endcase
    end

endmodule