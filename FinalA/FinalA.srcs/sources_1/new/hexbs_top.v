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
    input wire [31:0] ref_start_addr,   // [NEW]
    input wire [31:0] mb_x_pos,         
    input wire [31:0] mb_y_pos,         
    
    // 結果輸出
    output reg signed [5:0] mv_x, 
    output reg signed [5:0] mv_y, 
    output reg [15:0] sad,        
    output reg done               
);
    parameter WIDTH = 352;
    parameter HEIGHT = 240;
    parameter SEARCH_RANGE = 32;
    localparam signed [6:0] SEARCH_MIN = -SEARCH_RANGE;
    localparam signed [6:0] SEARCH_MAX = SEARCH_RANGE;
    
    // 狀態機定義
    localparam S_IDLE        = 0;
    localparam S_LOAD_CUR    = 1;
    localparam S_L_HEX_INIT  = 2;
    localparam S_L_HEX_CALC  = 3;
    localparam S_L_HEX_DECIDE= 4;
    localparam S_S_HEX_INIT  = 5;
    localparam S_S_HEX_CALC  = 6;
    localparam S_DONE        = 7;

    reg [3:0] state;
    reg [7:0] cur_mb_mem [0:255];
    
    // 座標相關宣告
    reg signed [6:0] center_x, center_y;
    reg signed [6:0] shex_center_x, shex_center_y;
    reg signed [6:0] cand_x, cand_y;     
    reg [8:0] pixel_cnt; 
    reg [8:0] pixel_cnt_d1;
    reg [8:0] pixel_cnt_d2;
    reg [3:0] point_cnt; 
    reg shex_load;
    reg [1:0]  mem_pipe_cnt;
    
    // SAD 計算相關
    reg [15:0] current_accum_sad;
    reg [15:0] min_sad_reg;       
    reg [3:0]  best_point_idx;    
    reg signed [6:0] best_cand_x, best_cand_y;

    // Pattern Tables
    reg signed [2:0] l_hex_off_x [0:6];
    reg signed [2:0] l_hex_off_y [0:6];
    reg signed [1:0] s_hex_off_x [0:3];
    reg signed [1:0] s_hex_off_y [0:3];
    
    // === 修正變數 ===
    reg signed [31:0] ref_x_calc, ref_y_calc;

    initial begin
        // Large Hexagon Offsets
        // (dr, dc) -> (cand_y, cand_x)
        l_hex_off_x[0]=0;  l_hex_off_y[0]=0;
        l_hex_off_x[1]=0;  l_hex_off_y[1]=2;   // (dr=2,  dc=0)
        l_hex_off_x[2]=2;  l_hex_off_y[2]=1;   // (dr=1,  dc=2)
        l_hex_off_x[3]=2;  l_hex_off_y[3]=-1;  // (dr=-1, dc=2)
        l_hex_off_x[4]=0;  l_hex_off_y[4]=-2;  // (dr=-2, dc=0)
        l_hex_off_x[5]=-2; l_hex_off_y[5]=-1;  // (dr=-1, dc=-2)
        l_hex_off_x[6]=-2; l_hex_off_y[6]=1;   // (dr=1,  dc=-2)
        
        // Small Hexagon offsets (Python: SHP = [(1,0), (0,1), (-1,0), (0,-1)])
        s_hex_off_x[0]=0;   s_hex_off_y[0]=1;   // (dr=1,  dc=0)
        s_hex_off_x[1]=1;   s_hex_off_y[1]=0;   // (dr=0,  dc=1)
        s_hex_off_x[2]=0;   s_hex_off_y[2]=-1;  // (dr=-1, dc=0)
        s_hex_off_x[3]=-1;  s_hex_off_y[3]=0;   // (dr=0,  dc=-1)
    end

    function [7:0] abs_diff;
        input [7:0] a, b;
        begin
            abs_diff = (a > b) ? (a - b) : (b - a);
        end
    endfunction
    
    // === 組合邏輯：即時計算當前 SAD (包含最後一個像素) ===
    wire [7:0] current_diff;
    wire [15:0] sad_accum_next;
    wire bound_ok;
    wire [15:0] sad_final_check;

    wire        mem_sample_ready = (mem_pipe_cnt == 2'd2);
    wire [7:0]  cur_mb_sample = cur_mb_mem[pixel_cnt_d2[7:0]];

    assign current_diff = abs_diff(cur_mb_sample, mem_rdata);
    assign sad_accum_next = current_accum_sad + current_diff;
    
    // 邊界檢查：確保目前的 cand_x/cand_y 合法
    assign bound_ok = (cand_x >= SEARCH_MIN && cand_x <= SEARCH_MAX &&
                       cand_y >= SEARCH_MIN && cand_y <= SEARCH_MAX);
    // 如果出界，將 SAD 視為最大值 (FFFF)，否則使用計算出的 SAD
    assign sad_final_check = bound_ok ? sad_accum_next : 16'hFFFF;


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 0;
            pixel_cnt <= 0;
            pixel_cnt_d1 <= 0;
            pixel_cnt_d2 <= 0;
            point_cnt <= 0;
            center_x <= 0; center_y <= 0;
            shex_center_x <= 0; shex_center_y <= 0;
            shex_load <= 0;
            min_sad_reg <= 16'hFFFF;
            mv_x <= 0; mv_y <= 0; sad <= 0;
            cand_x <= 0; cand_y <= 0;
            mem_pipe_cnt <= 0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= S_LOAD_CUR;
                        pixel_cnt <= 0;
                        pixel_cnt_d1 <= 0;
                        pixel_cnt_d2 <= 0;
                        mem_pipe_cnt <= 0;
                    end
                end

                S_LOAD_CUR: begin
                    pixel_cnt_d1 <= pixel_cnt;
                    pixel_cnt_d2 <= pixel_cnt_d1;
                    if (!mem_sample_ready) begin
                        mem_pipe_cnt <= mem_pipe_cnt + 1'b1;
                    end else begin
                        cur_mb_mem[pixel_cnt_d2[7:0]] <= mem_rdata;
                    end

                    if (mem_sample_ready && pixel_cnt_d2 == 9'd255) begin
                        state <= S_L_HEX_INIT;
                        center_x <= 0; center_y <= 0;
                        min_sad_reg <= 16'hFFFF;
                        pixel_cnt <= 0;
                        pixel_cnt_d1 <= 0;
                        pixel_cnt_d2 <= 0;
                        mem_pipe_cnt <= 0;
                    end else if (pixel_cnt != 9'd255) begin
                        pixel_cnt <= pixel_cnt + 1;
                    end
                end

                S_L_HEX_INIT: begin
                    point_cnt <= 0;
                    state <= S_L_HEX_CALC;
                    pixel_cnt <= 0;
                    pixel_cnt_d1 <= 0;
                    pixel_cnt_d2 <= 0;
                    mem_pipe_cnt <= 0;
                    current_accum_sad <= 0;
                    best_point_idx <= 0;
                    // === 預先載入第 0 點 ===
                    cand_x <= center_x + l_hex_off_x[0];
                    cand_y <= center_y + l_hex_off_y[0];
                end

                S_L_HEX_CALC: begin
                    pixel_cnt_d1 <= pixel_cnt;
                    pixel_cnt_d2 <= pixel_cnt_d1;
                    if (!mem_sample_ready) begin
                        mem_pipe_cnt <= mem_pipe_cnt + 1'b1;
                    end else begin
                        if (bound_ok)
                            current_accum_sad <= sad_accum_next;
                        else
                            current_accum_sad <= 16'hFFFF;
                    end

                    if (mem_sample_ready && pixel_cnt_d2 == 9'd255) begin
`ifdef HEXBS_DEBUG
                        if ((mb_x_pos == 0 && (mb_y_pos == 0 || mb_y_pos == 16)) ||
                            (mb_x_pos == 64 && mb_y_pos == 16))
                            $display("[DBGALL][LHEX] center=(%0d,%0d) cand=(%0d,%0d) ref=(%0d,%0d) sad=%0d point=%0d", 
                                     center_x, center_y, cand_x, cand_y, ref_x_calc, ref_y_calc, sad_final_check, point_cnt);
`endif
                        if (sad_final_check < min_sad_reg) begin
                            min_sad_reg <= sad_final_check;
                            best_point_idx <= point_cnt;
                            best_cand_x <= cand_x;
                            best_cand_y <= cand_y;
`ifdef HEXBS_DEBUG
                            if ((mb_x_pos == 0 && mb_y_pos == 0) || (mb_x_pos == 0 && mb_y_pos == 16) ||
                                (mb_x_pos == 64 && mb_y_pos == 16))
                                $display("[DBG][LHEX] center=(%0d,%0d) cand=(%0d,%0d) ref=(%0d,%0d) sad=%0d point=%0d", 
                                         center_x, center_y, cand_x, cand_y, ref_x_calc, ref_y_calc, sad_final_check, point_cnt);
`endif
                        end

                        if (point_cnt == 6) begin
                            state <= S_L_HEX_DECIDE;
                        end else begin
                            point_cnt <= point_cnt + 1;
                            pixel_cnt <= 0;
                            pixel_cnt_d1 <= 0;
                            pixel_cnt_d2 <= 0;
                            mem_pipe_cnt <= 0;
                            current_accum_sad <= 0;
                            // === 預先計算下一個點 ===
                            cand_x <= center_x + l_hex_off_x[point_cnt + 1];
                            cand_y <= center_y + l_hex_off_y[point_cnt + 1];
                        end
                    end else if (pixel_cnt != 9'd255) begin
                        pixel_cnt <= pixel_cnt + 1;
                    end
                end

                S_L_HEX_DECIDE: begin
                    if (best_point_idx == 0) begin
                        state <= S_S_HEX_INIT;
                    end else begin
                        center_x <= best_cand_x;
                        center_y <= best_cand_y;
                        state <= S_L_HEX_INIT; 
                    end
                end

                S_S_HEX_INIT: begin
                    point_cnt <= 0;
                    state <= S_S_HEX_CALC;
                    pixel_cnt <= 0;
                    pixel_cnt_d1 <= 0;
                    pixel_cnt_d2 <= 0;
                    mem_pipe_cnt <= 0;
                    current_accum_sad <= 0;
                    shex_center_x <= center_x;
                    shex_center_y <= center_y;
                    shex_load <= 1;
                end

                S_S_HEX_CALC: begin
                    if (shex_load) begin
                        cand_x <= shex_center_x + s_hex_off_x[point_cnt];
                        cand_y <= shex_center_y + s_hex_off_y[point_cnt];
                        pixel_cnt <= 0;
                        pixel_cnt_d1 <= 0;
                        pixel_cnt_d2 <= 0;
                        current_accum_sad <= 0;
                        mem_pipe_cnt <= 0;
                        shex_load <= 0;
                    end else begin
                        pixel_cnt_d1 <= pixel_cnt;
                        pixel_cnt_d2 <= pixel_cnt_d1;
                        if (!mem_sample_ready) begin
                            mem_pipe_cnt <= mem_pipe_cnt + 1'b1;
                        end else begin
                            if (bound_ok)
                                current_accum_sad <= sad_accum_next;
                            else
                                current_accum_sad <= 16'hFFFF;
                        end

                        if (mem_sample_ready && pixel_cnt_d2 == 9'd255) begin
`ifdef HEXBS_DEBUG
                            if ((mb_x_pos == 0 && mb_y_pos == 0) || (mb_x_pos == 0 && mb_y_pos == 16) ||
                                (mb_x_pos == 64 && mb_y_pos == 16))
                                $display("[DBG][SHEX] center=(%0d,%0d) cand=(%0d,%0d) ref=(%0d,%0d) sad=%0d point=%0d", 
                                         center_x, center_y, cand_x, cand_y, ref_x_calc, ref_y_calc, sad_final_check, point_cnt);
`endif
                            if (sad_final_check < min_sad_reg) begin
                                min_sad_reg <= sad_final_check;
                                best_cand_x <= cand_x;
                                best_cand_y <= cand_y;
                                shex_center_x <= cand_x;
                                shex_center_y <= cand_y;
                            end

                            if (point_cnt == 3) begin
                                state <= S_DONE;
                            end else begin
                                point_cnt <= point_cnt + 1;
                                pixel_cnt <= 0;
                                pixel_cnt_d1 <= 0;
                                pixel_cnt_d2 <= 0;
                                mem_pipe_cnt <= 0;
                                shex_load <= 1;
                            end
                        end else if (pixel_cnt != 9'd255) begin
                            pixel_cnt <= pixel_cnt + 1;
                        end
                    end
                end

                S_DONE: begin
                    mv_x <= best_cand_x;
                    mv_y <= best_cand_y;
                    sad  <= min_sad_reg;
                    done <= 1;
                    if (!start) state <= S_IDLE;
                end
            endcase
        end
    end

    // ==========================================
    // 記憶體地址計算組合邏輯
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
            
            S_L_HEX_CALC, S_S_HEX_CALC: begin
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
        endcase
    end
endmodule
