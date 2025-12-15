`timescale 1ns/1ps

module Smart_Top (
    input wire clk,         // 100MHz
    input wire btnC,        // Reset
    input wire RsRx,        // UART RX
    output wire RsTx,       // UART TX
    output wire [15:0] led  // 儀表板
);

    // ==========================================
    // 1. 參數與常數
    // ==========================================
    localparam integer UART_CLKS       = 108;    // 100MHz / 921600 ~= 108
    localparam integer FRAME_WIDTH     = 352;
    localparam integer FRAME_HEIGHT    = 240;
    localparam integer FRAME_PIXELS    = FRAME_WIDTH * FRAME_HEIGHT; // 84480
    localparam integer FRAME_ADDR_WIDTH= $clog2(FRAME_PIXELS);
    localparam integer MB_SIZE         = 16;
    localparam [15:0] MAX_MB_X         = FRAME_WIDTH - MB_SIZE;
    localparam [15:0] MAX_MB_Y         = FRAME_HEIGHT - MB_SIZE;
    localparam [31:0] FRAME_BASE_ADDR  = 32'd0;
    localparam [31:0] REF_BASE_ADDR    = FRAME_BASE_ADDR + FRAME_PIXELS;

    localparam [7:0] CMD_LOAD_REF = "R";
    localparam [7:0] CMD_LOAD_CUR = "F";
    localparam [7:0] CMD_RUN_CASE = "C";
    localparam [7:0] ACK_BYTE     = "K";

    // ==========================================
    // 2. 時脈與 GPIO
    // ==========================================
    wire clk_sys = clk;     // 直接使用 100MHz，提供 921600 Baud
    wire rst_n   = ~btnC;

    // ==========================================
    // 3. UART 介面
    // ==========================================
    wire       rx_dv;
    wire [7:0] rx_byte;
    reg        tx_dv;
    reg [7:0]  tx_byte;
    wire       tx_active;
    wire       tx_done;

    // ==========================================
    // 4. 狀態機定義
    // ==========================================
    localparam S_WAIT_CMD       = 3'd0;
    localparam S_RX_FRAME_HEADER= 3'd1;
    localparam S_RX_FRAME_DATA  = 3'd2;
    localparam S_RX_MB_HEADER   = 3'd3;
    localparam S_CALC           = 3'd4;
    localparam S_TX_SEND        = 3'd5;
    localparam S_TX_WAIT_ACK    = 3'd6;

    reg [2:0] state;

    // ==========================================
    // 5. RAM 與資料寄存
    // ==========================================
    (* ram_style = "block" *)
    reg [7:0] ref_mem [0:FRAME_PIXELS-1]; // 參考 Frame 緩衝
    (* ram_style = "block" *)
    reg [7:0] cur_mem [0:FRAME_PIXELS-1]; // 當前 Frame 緩衝
    reg        frame_wr_en;
    reg [FRAME_ADDR_WIDTH-1:0] frame_wr_addr;
    reg [7:0]  frame_wr_data;
    reg        frame_wr_sel_cur;

    reg [FRAME_ADDR_WIDTH-1:0] frame_byte_cnt;
    reg [1:0]  mb_header_idx;
    reg [1:0]  frame_header_idx;
    reg [15:0] temp_x;
    reg [15:0] temp_y;

    reg        load_cur_frame;
    reg        ref_frame_ready;
    reg        cur_frame_ready;
    reg        cmd_ack_pending;

    // 給 HEXBS 的參數
    reg        hex_start;
    reg        hex_busy;
    wire       hex_done;
    wire [31:0] mem_addr;
    wire [7:0]  mem_rdata;
    reg  [31:0] mb_x_pos_reg, mb_y_pos_reg;

    wire signed [5:0] mv_x, mv_y;
    wire [15:0] sad;

    // 結果鎖存
    reg signed [5:0] latch_mv_x, latch_mv_y;
    reg [15:0] latch_sad;
    reg [1:0]  tx_step;

    // 已同步化的 BRAM 讀取控制
    reg        mem_sel_cur_d;
    reg [FRAME_ADDR_WIDTH-1:0] mem_rd_addr_d;
    reg        mem_addr_valid_d;

    // ==========================================
    // 6. 輔助函式
    // ==========================================
    function [15:0] clamp_coord;
        input [15:0] value;
        input [15:0] max_value;
        begin
            if (value > max_value)
                clamp_coord = max_value;
            else
                clamp_coord = value;
        end
    endfunction

    // ==========================================
    // 7. 主控制邏輯
    // ==========================================
    always @(posedge clk_sys or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_WAIT_CMD;
            tx_dv            <= 0;
            tx_byte          <= 0;
            frame_wr_en      <= 0;
            frame_wr_addr    <= 0;
            frame_wr_data    <= 0;
            frame_wr_sel_cur <= 0;
            frame_byte_cnt   <= 0;
            frame_header_idx <= 0;
            mb_header_idx    <= 0;
            temp_x           <= 0;
            temp_y           <= 0;
            load_cur_frame   <= 0;
            ref_frame_ready  <= 0;
            cur_frame_ready  <= 0;
            cmd_ack_pending  <= 0;
            mb_x_pos_reg     <= 0;
            mb_y_pos_reg     <= 0;
            latch_mv_x       <= 0;
            latch_mv_y       <= 0;
            latch_sad        <= 0;
            tx_step          <= 0;
        end else begin
            tx_dv      <= 0;
            frame_wr_en<= 0;

            if (cmd_ack_pending && !tx_active &&
                state != S_TX_SEND && state != S_TX_WAIT_ACK) begin
                tx_byte         <= ACK_BYTE;
                tx_dv           <= 1;
                cmd_ack_pending <= 0;
            end

            case (state)
                S_WAIT_CMD: begin
                    if (rx_dv) begin
                        if (rx_byte == CMD_LOAD_REF) begin
                            load_cur_frame   <= 1'b0;
                            ref_frame_ready  <= 1'b0;
                            frame_header_idx <= 0;
                            frame_byte_cnt   <= 0;
                            state            <= S_RX_FRAME_HEADER;
                        end else if (rx_byte == CMD_LOAD_CUR) begin
                            load_cur_frame   <= 1'b1;
                            cur_frame_ready  <= 1'b0;
                            frame_header_idx <= 0;
                            frame_byte_cnt   <= 0;
                            state            <= S_RX_FRAME_HEADER;
                        end else if (rx_byte == CMD_RUN_CASE) begin
                            mb_header_idx <= 0;
                            state         <= S_RX_MB_HEADER;
                        end
                    end
                end

                S_RX_FRAME_HEADER: begin
                    if (rx_dv) begin
                        if (frame_header_idx == 0) begin
                            temp_x[7:0]      <= rx_byte;
                            frame_header_idx <= 1;
                        end else begin
                            temp_x[15:8]     <= rx_byte;
                            frame_header_idx <= 0;
                            frame_byte_cnt   <= 0;
                            state            <= S_RX_FRAME_DATA;
                        end
                    end
                end

                S_RX_FRAME_DATA: begin
                    if (rx_dv) begin
                        frame_wr_en      <= 1;
                        frame_wr_addr    <= frame_byte_cnt;
                        frame_wr_data    <= rx_byte;
                        frame_wr_sel_cur <= load_cur_frame;
                        if (frame_byte_cnt == FRAME_PIXELS-1) begin
                            frame_byte_cnt <= 0;
                            state          <= S_WAIT_CMD;
                            if (load_cur_frame)
                                cur_frame_ready <= 1;
                            else
                                ref_frame_ready <= 1;
                            cmd_ack_pending <= 1;
                        end else begin
                            frame_byte_cnt <= frame_byte_cnt + 1'b1;
                        end
                    end
                end

                S_RX_MB_HEADER: begin
                    if (rx_dv) begin
                        case (mb_header_idx)
                            2'd0: temp_x[7:0]  <= rx_byte;
                            2'd1: temp_x[15:8] <= rx_byte;
                            2'd2: temp_y[7:0]  <= rx_byte;
                            2'd3: begin
                                temp_y[15:8] <= rx_byte;
                                mb_x_pos_reg <= clamp_coord(temp_x, MAX_MB_X);
                                mb_y_pos_reg <= clamp_coord({rx_byte, temp_y[7:0]}, MAX_MB_Y);
                                if (ref_frame_ready && cur_frame_ready)
                                    state <= S_CALC;
                                else
                                    state <= S_WAIT_CMD;
                                mb_header_idx <= 0;
                            end
                        endcase

                        if (mb_header_idx != 2'd3)
                            mb_header_idx <= mb_header_idx + 1'b1;
                    end
                end

                S_CALC: begin
                    if (hex_done) begin
                        latch_mv_x <= mv_x;
                        latch_mv_y <= mv_y;
                        latch_sad  <= sad;
                        tx_step    <= 0;
                        state      <= S_TX_SEND;
                    end
                end

                S_TX_SEND: begin
                    if (!tx_active) begin
                        tx_dv <= 1;
                        case (tx_step)
                            2'd0: tx_byte <= {2'b00, latch_mv_x};
                            2'd1: tx_byte <= {2'b00, latch_mv_y};
                            2'd2: tx_byte <= latch_sad[15:8];
                            2'd3: tx_byte <= latch_sad[7:0];
                            default: tx_byte <= 8'h00;
                        endcase
                        state <= S_TX_WAIT_ACK;
                    end
                end

                S_TX_WAIT_ACK: begin
                    if (rx_dv && rx_byte == ACK_BYTE) begin
                        if (tx_step == 2'd3) begin
                            state <= S_WAIT_CMD;
                        end else begin
                            tx_step <= tx_step + 1'b1;
                            state   <= S_TX_SEND;
                        end
                    end
                end
            endcase
        end
    end

    // 讓 HEXBS 僅收到單拍的 start，避免重複觸發
    always @(posedge clk_sys or negedge rst_n) begin
        if (!rst_n) begin
            hex_start <= 0;
            hex_busy  <= 0;
        end else begin
            hex_start <= 0;
            if (!hex_busy && state == S_CALC) begin
                hex_start <= 1;
                hex_busy  <= 1;
            end else if (hex_done) begin
                hex_busy <= 0;
            end
        end
    end

    // ==========================================
    // 8. 記憶體回應
    // ==========================================
    reg [7:0] mem_rdata_reg;
    wire [FRAME_ADDR_WIDTH-1:0] cur_idx = mem_addr[FRAME_ADDR_WIDTH-1:0];
    wire [31:0] ref_offset = mem_addr - REF_BASE_ADDR;
    wire [FRAME_ADDR_WIDTH-1:0] ref_idx = ref_offset[FRAME_ADDR_WIDTH-1:0];

    // 實體寫入 Block RAM
    always @(posedge clk_sys) begin
        if (frame_wr_en) begin
            if (frame_wr_sel_cur)
                cur_mem[frame_wr_addr] <= frame_wr_data;
            else
                ref_mem[frame_wr_addr] <= frame_wr_data;
        end
    end

    // 將地址與來源記錄為同步訊號，確保推導為真正的 dual-port BRAM
    always @(posedge clk_sys or negedge rst_n) begin
        if (!rst_n) begin
            mem_sel_cur_d    <= 1'b1;
            mem_rd_addr_d    <= {FRAME_ADDR_WIDTH{1'b0}};
            mem_addr_valid_d <= 1'b0;
        end else if (mem_addr < REF_BASE_ADDR) begin
            mem_sel_cur_d    <= 1'b1;
            mem_rd_addr_d    <= cur_idx;
            mem_addr_valid_d <= (mem_addr < FRAME_PIXELS);
        end else begin
            mem_sel_cur_d    <= 1'b0;
            mem_rd_addr_d    <= ref_idx;
            mem_addr_valid_d <= (ref_offset < FRAME_PIXELS);
        end
    end

    always @(posedge clk_sys or negedge rst_n) begin
        if (!rst_n) begin
            mem_rdata_reg <= 8'h00;
        end else if (!mem_addr_valid_d) begin
            mem_rdata_reg <= 8'h00;
        end else if (mem_sel_cur_d) begin
            mem_rdata_reg <= cur_mem[mem_rd_addr_d];
        end else begin
            mem_rdata_reg <= ref_mem[mem_rd_addr_d];
        end
    end

    assign mem_rdata = mem_rdata_reg;

    // ==========================================
    // 9. 模組連接
    // ==========================================
    uart_rx #(.CLKS_PER_BIT(UART_CLKS)) u_rx (
        .clk(clk_sys),
        .i_Rx_Serial(RsRx),
        .o_Rx_DV(rx_dv),
        .o_Rx_Byte(rx_byte)
    );

    uart_tx #(.CLKS_PER_BIT(UART_CLKS)) u_tx (
        .clk(clk_sys),
        .i_Tx_DV(tx_dv),
        .i_Tx_Byte(tx_byte),
        .o_Tx_Active(tx_active),
        .o_Tx_Serial(RsTx),
        .o_Tx_Done(tx_done)
    );

    hexbs_top #(
        .WIDTH (FRAME_WIDTH),
        .HEIGHT(FRAME_HEIGHT)
    ) u_core (
        .clk(clk_sys),
        .rst_n(rst_n),
        .start(hex_start),
        .mem_addr(mem_addr),
        .mem_rdata(mem_rdata),
        .frame_start_addr(FRAME_BASE_ADDR),
        .ref_start_addr  (REF_BASE_ADDR),
        .mb_x_pos(mb_x_pos_reg),
        .mb_y_pos(mb_y_pos_reg),
        .mv_x(mv_x),
        .mv_y(mv_y),
        .sad(sad),
        .done(hex_done)
    );

    // ==========================================
    // 10. LED 儀表板
    // ==========================================
    assign led[15:8] = {2'b00, latch_mv_x};
    assign led[7:0]  = {2'b00, latch_mv_y};

endmodule
