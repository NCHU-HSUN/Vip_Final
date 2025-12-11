// ============================================================================
// File Name: ME_HEXBS.v
// Author: HSUN's AI Assistant
// Description: Hexagon-Based Search (HEXBS) Motion Estimation Accelerator
//              Fixed: Added reset logic for cand_x/cand_y to prevent 'X' propagation
// ============================================================================

module ME_HEXBS (
    input  wire         clk,        // 時脈
    input  wire         rst_n,      // 重置訊號 (低電位重置)
    input  wire         i_start,    // 啟動訊號

    // --- 虛擬記憶體介面 ---
    output wire [5:0]   o_cur_x,    
    output wire [5:0]   o_cur_y,    
    input  wire [7:0]   i_cur_pixel,

    output wire [11:0]  o_ref_x,    
    output wire [11:0]  o_ref_y,    
    input  wire [7:0]   i_ref_pixel,

    // --- 輸出結果 ---
    output reg  signed [5:0] o_mv_x,    
    output reg  signed [5:0] o_mv_y,    
    output reg  [15:0]       o_min_sad, 
    output reg               o_done     
);

    // =========================================================
    // 1. 參數與變數
    // =========================================================
    localparam S_IDLE       = 4'd0; 
    localparam S_INIT_00    = 4'd1; 
    localparam S_LHP_PICK   = 4'd2; 
    localparam S_LHP_CALC   = 4'd3; 
    localparam S_LHP_DECIDE = 4'd4; 
    localparam S_SHP_PICK   = 4'd5; 
    localparam S_SHP_CALC   = 4'd6; 
    localparam S_DONE       = 4'd7; 

    reg [3:0]  state, next_state;

    // 搜尋中心與候選點
    reg signed [5:0] center_x, center_y;    
    reg signed [5:0] cand_x, cand_y;        // [Bug Fix] 這裡原本沒 Reset
    
    reg [15:0] global_min_sad;              
    reg signed [5:0] best_vec_x, best_vec_y;

    reg [4:0]  pixel_cnt_x; 
    reg [4:0]  pixel_cnt_y; 
    reg [15:0] current_accum_sad; 

    reg [2:0]  point_idx;   

    // =========================================================
    // 2. 函數與 FSM
    // =========================================================
    function [7:0] abs_diff;
        input [7:0] a, b;
        begin
            abs_diff = (a > b) ? (a - b) : (b - a);
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    always @(*) begin
        case (state)
            S_IDLE:       next_state = i_start ? S_INIT_00 : S_IDLE;
            S_INIT_00:    next_state = (pixel_cnt_x == 15 && pixel_cnt_y == 15) ? S_LHP_PICK : S_INIT_00;
            S_LHP_PICK:   next_state = S_LHP_CALC;
            S_LHP_CALC:   begin
                if (pixel_cnt_x == 15 && pixel_cnt_y == 15) 
                    next_state = (point_idx == 5) ? S_LHP_DECIDE : S_LHP_PICK;
                else 
                    next_state = S_LHP_CALC;
            end
            S_LHP_DECIDE: next_state = (best_vec_x == center_x && best_vec_y == center_y) ? S_SHP_PICK : S_LHP_PICK;
            S_SHP_PICK:   next_state = S_SHP_CALC;
            S_SHP_CALC:   begin
                if (pixel_cnt_x == 15 && pixel_cnt_y == 15) 
                    next_state = (point_idx == 3) ? S_DONE : S_SHP_PICK;
                else 
                    next_state = S_SHP_CALC;
            end
            S_DONE:       next_state = S_IDLE;
            default:      next_state = S_IDLE;
        endcase
    end

    // =========================================================
    // 3. 資料路徑 (Datapath)
    // =========================================================
    assign o_cur_x = pixel_cnt_x;
    assign o_cur_y = pixel_cnt_y;
    assign o_ref_x = 100 + cand_x + pixel_cnt_x; 
    assign o_ref_y = 100 + cand_y + pixel_cnt_y;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // [關鍵修正] 所有暫存器都要清零！
            center_x <= 0; center_y <= 0;
            cand_x <= 0;   cand_y <= 0;   // <--- 加入這行解藥！
            global_min_sad <= 16'hFFFF; 
            o_done <= 0;
            point_idx <= 0;
            pixel_cnt_x <= 0; pixel_cnt_y <= 0;
            o_mv_x <= 0; o_mv_y <= 0;
            current_accum_sad <= 0; // 順便把這個也清一下比較保險
        end else begin
            case (state)
                S_IDLE: begin
                    o_done <= 0;
                    center_x <= 0; center_y <= 0;
                    global_min_sad <= 16'hFFFF;
                    o_mv_x <= 0; o_mv_y <= 0;
                end

                S_INIT_00: begin
                    cand_x <= 0; cand_y <= 0; 
                    
                    if (pixel_cnt_x == 0 && pixel_cnt_y == 0) current_accum_sad <= abs_diff(i_cur_pixel, i_ref_pixel);
                    else current_accum_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);

                    if (pixel_cnt_x == 15) begin
                        pixel_cnt_x <= 0;
                        if (pixel_cnt_y == 15) begin
                            global_min_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);
                            best_vec_x <= 0; best_vec_y <= 0;
                            pixel_cnt_y <= 0;
                            point_idx <= 0; 
                        end else pixel_cnt_y <= pixel_cnt_y + 1;
                    end else pixel_cnt_x <= pixel_cnt_x + 1;
                end

                S_LHP_PICK: begin
                    pixel_cnt_x <= 0; pixel_cnt_y <= 0;
                    current_accum_sad <= 0;
                    case (point_idx)
                        0: begin cand_x <= center_x + 2; cand_y <= center_y + 0; end
                        1: begin cand_x <= center_x + 1; cand_y <= center_y + 2; end
                        2: begin cand_x <= center_x - 1; cand_y <= center_y + 2; end
                        3: begin cand_x <= center_x - 2; cand_y <= center_y + 0; end
                        4: begin cand_x <= center_x - 1; cand_y <= center_y - 2; end
                        5: begin cand_x <= center_x + 1; cand_y <= center_y - 2; end
                    endcase
                end

                S_LHP_CALC: begin
                    if (pixel_cnt_x == 0 && pixel_cnt_y == 0) current_accum_sad <= abs_diff(i_cur_pixel, i_ref_pixel);
                    else current_accum_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);

                    if (pixel_cnt_x == 15) begin
                        pixel_cnt_x <= 0;
                        if (pixel_cnt_y == 15) begin
                            pixel_cnt_y <= 0;
                            if ((current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel)) < global_min_sad) begin
                                global_min_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);
                                best_vec_x <= cand_x;
                                best_vec_y <= cand_y;
                            end
                            if (point_idx < 6) point_idx <= point_idx + 1;
                        end else pixel_cnt_y <= pixel_cnt_y + 1;
                    end else pixel_cnt_x <= pixel_cnt_x + 1;
                end

                S_LHP_DECIDE: begin
                    point_idx <= 0; 
                    if (best_vec_x != center_x || best_vec_y != center_y) begin
                        center_x <= best_vec_x;
                        center_y <= best_vec_y;
                    end
                end

                S_SHP_PICK: begin
                    pixel_cnt_x <= 0; pixel_cnt_y <= 0;
                    current_accum_sad <= 0;
                    case (point_idx)
                        0: begin cand_x <= center_x + 1; cand_y <= center_y + 0; end
                        1: begin cand_x <= center_x + 0; cand_y <= center_y + 1; end
                        2: begin cand_x <= center_x - 1; cand_y <= center_y + 0; end
                        3: begin cand_x <= center_x + 0; cand_y <= center_y - 1; end
                    endcase
                end

                S_SHP_CALC: begin
                    if (pixel_cnt_x == 0 && pixel_cnt_y == 0) current_accum_sad <= abs_diff(i_cur_pixel, i_ref_pixel);
                    else current_accum_sad <= current_accum_sad + abs_diff(i_cur_pixel, i_ref_pixel);

                    if (pixel_cnt_x == 15) begin
                        pixel_cnt_x <= 0;
                        if (pixel_cnt_y == 15) begin
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