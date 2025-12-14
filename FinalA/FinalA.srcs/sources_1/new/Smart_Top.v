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
    localparam integer UART_CLKS      = 108;    // 100MHz / 921600 ~= 108
    localparam integer WINDOW_SIZE    = 80;
    localparam integer WINDOW_PIXELS  = WINDOW_SIZE * WINDOW_SIZE; // 6400
    localparam integer MB_SIZE        = 16;
    localparam [6:0] MAX_MB_OFFSET    = WINDOW_SIZE - MB_SIZE;     // 64
    localparam [31:0] FRAME_BASE_ADDR = 32'd0;
    localparam [31:0] REF_BASE_ADDR   = 32'd8192;

    localparam [7:0] CMD_LOAD_WINDOW = "W";
    localparam [7:0] CMD_LOAD_CUR    = "C";
    localparam [7:0] ACK_BYTE        = "K";

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

    reg        echo_pending;
    reg [7:0]  echo_byte;

    // ==========================================
    // 4. 狀態機定義
    // ==========================================
    localparam S_WAIT_CMD      = 3'd0;
    localparam S_RX_WIN_HEADER = 3'd1;
    localparam S_RX_WIN_DATA   = 3'd2;
    localparam S_RX_CUR_HEADER = 3'd3;
    localparam S_RX_CUR_DATA   = 3'd4;
    localparam S_CALC          = 3'd5;
    localparam S_TX_SEND       = 3'd6;
    localparam S_TX_WAIT_ACK   = 3'd7;

    reg [2:0] state;

    // ==========================================
    // 5. RAM 與資料寄存
    // ==========================================
    (* ram_style = "block" *)
    reg [7:0] ref_mem [0:WINDOW_PIXELS-1]; // 參考視窗 (80x80)
    (* ram_style = "block" *)
    reg [7:0] cur_mem [0:WINDOW_PIXELS-1]; // 當前 Frame 的對應範圍
    reg        ref_wr_en;
    reg [12:0] ref_wr_addr;
    reg [7:0]  ref_wr_data;
    reg        cur_wr_en;
    reg [12:0] cur_wr_addr;
    reg [7:0]  cur_wr_data;

    reg [12:0] window_cnt;
    reg [8:0]  cur_cnt;

    reg [1:0]  header_idx;
    reg [15:0] temp_x;
    reg [15:0] temp_y;

    reg [15:0] window_base_x, window_base_y;
    reg [6:0]  rel_mb_x, rel_mb_y;
    reg        window_loaded;

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
    reg [12:0] mem_rd_addr_d;
    reg        mem_addr_valid_d;

    // ==========================================
    // 6. 輔助函式
    // ==========================================
    function [6:0] clamp_offset;
        input [15:0] value;
        input [15:0] base;
        reg [15:0] diff;
        begin
            if (value <= base) diff = 0;
            else diff = value - base;

            if (diff > MAX_MB_OFFSET)
                clamp_offset = MAX_MB_OFFSET;
            else
                clamp_offset = diff[6:0];
        end
    endfunction

    function [12:0] win_index;
        input [6:0] row;
        input [6:0] col;
        begin
            win_index = row * WINDOW_SIZE + col;
        end
    endfunction

    // 需要 echo 的狀態
    wire need_echo = (state == S_WAIT_CMD)      ||
                     (state == S_RX_WIN_HEADER) ||
                     (state == S_RX_WIN_DATA)   ||
                     (state == S_RX_CUR_HEADER) ||
                     (state == S_RX_CUR_DATA);

    // ==========================================
    // 7. 主控制邏輯
    // ==========================================
    always @(posedge clk_sys or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_WAIT_CMD;
            tx_dv         <= 0;
            tx_byte       <= 0;
            echo_pending  <= 0;
            window_cnt    <= 0;
            cur_cnt       <= 0;
            header_idx    <= 0;
            temp_x        <= 0;
            temp_y        <= 0;
            window_base_x <= 0;
            window_base_y <= 0;
            rel_mb_x      <= 0;
            rel_mb_y      <= 0;
            window_loaded <= 0;
            mb_x_pos_reg  <= 0;
            mb_y_pos_reg  <= 0;
            latch_mv_x    <= 0;
            latch_mv_y    <= 0;
            latch_sad     <= 0;
            tx_step       <= 0;
            ref_wr_en     <= 0;
            ref_wr_addr   <= 0;
            ref_wr_data   <= 0;
            cur_wr_en     <= 0;
            cur_wr_addr   <= 0;
            cur_wr_data   <= 0;
        end else begin
            tx_dv <= 0;
            ref_wr_en <= 0;
            cur_wr_en <= 0;

            // Echo 管理
            if (need_echo && rx_dv) begin
                echo_pending <= 1;
                echo_byte    <= rx_byte;
            end

            if (echo_pending && !tx_active &&
                state != S_TX_SEND && state != S_TX_WAIT_ACK) begin
                tx_byte      <= echo_byte;
                tx_dv        <= 1;
                echo_pending <= 0;
            end

            case (state)
                // ------------------------------------------
                S_WAIT_CMD: begin
                    if (rx_dv) begin
                        if (rx_byte == CMD_LOAD_WINDOW) begin
                            header_idx    <= 0;
                            window_cnt    <= 0;
                            window_loaded <= 0;
                            state         <= S_RX_WIN_HEADER;
                        end else if (rx_byte == CMD_LOAD_CUR) begin
                            header_idx <= 0;
                            cur_cnt    <= 0;
                            state      <= S_RX_CUR_HEADER;
                        end
                    end
                end

                // ------------------------------------------
                S_RX_WIN_HEADER: begin
                    if (rx_dv) begin
                        case (header_idx)
                            2'd0: temp_x[7:0]  <= rx_byte;
                            2'd1: temp_x[15:8] <= rx_byte;
                            2'd2: temp_y[7:0]  <= rx_byte;
                            2'd3: begin
                                temp_y[15:8] <= rx_byte;
                                window_base_x <= temp_x;
                                window_base_y <= {rx_byte, temp_y[7:0]};
                                header_idx    <= 0;
                                state         <= S_RX_WIN_DATA;
                            end
                        endcase

                        if (header_idx != 2'd3)
                            header_idx <= header_idx + 1'b1;
                    end
                end

                // ------------------------------------------
                S_RX_WIN_DATA: begin
                    if (rx_dv) begin
                        ref_wr_en   <= 1;
                        ref_wr_addr <= window_cnt;
                        ref_wr_data <= rx_byte;
                        if (window_cnt == WINDOW_PIXELS-1) begin
                            window_loaded <= 1;
                            state         <= S_WAIT_CMD;
                            window_cnt    <= 0;
                        end else begin
                            window_cnt <= window_cnt + 1'b1;
                        end
                    end
                end

                // ------------------------------------------
                S_RX_CUR_HEADER: begin
                    if (rx_dv) begin
                        case (header_idx)
                            2'd0: temp_x[7:0]  <= rx_byte;
                            2'd1: temp_x[15:8] <= rx_byte;
                            2'd2: temp_y[7:0]  <= rx_byte;
                            2'd3: begin
                                temp_y[15:8] <= rx_byte;
                                rel_mb_x     <= clamp_offset(temp_x, window_base_x);
                                rel_mb_y     <= clamp_offset({rx_byte, temp_y[7:0]}, window_base_y);
                                mb_x_pos_reg <= clamp_offset(temp_x, window_base_x);
                                mb_y_pos_reg <= clamp_offset({rx_byte, temp_y[7:0]}, window_base_y);
                                header_idx   <= 0;
                                cur_cnt      <= 0;
                                state        <= S_RX_CUR_DATA;
                            end
                        endcase

                        if (header_idx != 2'd3)
                            header_idx <= header_idx + 1'b1;
                    end
                end

                // ------------------------------------------
                S_RX_CUR_DATA: begin
                    if (rx_dv) begin
                        // 依照相對座標寫入 80x80 當前 Frame 緩衝
                        cur_wr_en   <= 1;
                        cur_wr_addr <= win_index(
                            rel_mb_y + {3'b000, cur_cnt[7:4]},
                            rel_mb_x + {3'b000, cur_cnt[3:0]}
                        );
                        cur_wr_data <= rx_byte;

                        if (cur_cnt == MB_SIZE*MB_SIZE-1) begin
                            cur_cnt <= 0;
                            if (window_loaded)
                                state <= S_CALC;
                            else
                                state <= S_WAIT_CMD;
                        end else begin
                            cur_cnt <= cur_cnt + 1'b1;
                        end
                    end
                end

                // ------------------------------------------
                S_CALC: begin
                    if (hex_done) begin
                        latch_mv_x <= mv_x;
                        latch_mv_y <= mv_y;
                        latch_sad  <= sad;
                        tx_step    <= 0;
                        state      <= S_TX_SEND;
                    end
                end

                // ------------------------------------------
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

                // ------------------------------------------
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
    wire [12:0] cur_idx = mem_addr[12:0];
    wire [31:0] ref_offset = mem_addr - REF_BASE_ADDR;
    wire [12:0] ref_idx = ref_offset[12:0];

    // 實體寫入 Block RAM
    always @(posedge clk_sys) begin
        if (ref_wr_en)
            ref_mem[ref_wr_addr] <= ref_wr_data;
    end

    always @(posedge clk_sys) begin
        if (cur_wr_en)
            cur_mem[cur_wr_addr] <= cur_wr_data;
    end

    // 將地址與來源記錄為同步訊號，確保推導為真正的 dual-port BRAM
    always @(posedge clk_sys or negedge rst_n) begin
        if (!rst_n) begin
            mem_sel_cur_d    <= 1'b1;
            mem_rd_addr_d    <= 13'd0;
            mem_addr_valid_d <= 1'b0;
        end else if (mem_addr < REF_BASE_ADDR) begin
            mem_sel_cur_d    <= 1'b1;
            mem_rd_addr_d    <= cur_idx;
            mem_addr_valid_d <= (mem_addr < WINDOW_PIXELS);
        end else begin
            mem_sel_cur_d    <= 1'b0;
            mem_rd_addr_d    <= ref_idx;
            mem_addr_valid_d <= (ref_offset < WINDOW_PIXELS);
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
        .WIDTH (WINDOW_SIZE),
        .HEIGHT(WINDOW_SIZE)
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
