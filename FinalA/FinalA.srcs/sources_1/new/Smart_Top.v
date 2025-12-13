`timescale 1ns/1ps

module Smart_Top (
    input wire clk,         // 100MHz
    input wire btnC,        // Reset
    input wire RsRx,        // UART RX
    output wire RsTx,       // UART TX
    output wire [15:0] led  // 儀表板
);

    // ==========================================
    // 1. 降速設定 (2400 Baud)
    // ==========================================
    reg [1:0] clk_div;
    always @(posedge clk) clk_div <= clk_div + 1;
    wire clk_sys = clk_div[1]; // 25MHz

    localparam UART_CLKS = 217; // 2400 Baud

    // ==========================================
    // 2. 系統狀態機
    // ==========================================
    wire rst_n = ~btnC; 

    // 狀態定義
    localparam S_RX          = 0;
    localparam S_CALC        = 1;
    localparam S_TX_SEND     = 2; 
    localparam S_TX_WAIT_ACK = 3; 
    
    reg [2:0] state;

    // UART
    wire rx_dv;
    wire [7:0] rx_byte;
    reg tx_dv;
    reg [7:0] tx_byte;
    wire tx_active, tx_done;
    
    // RAM & HEXBS
    reg [7:0] ram [0:255];  
    reg [8:0] write_cnt;    
    
    reg hex_start;
    wire hex_done;
    wire [31:0] mem_addr;
    reg [7:0] mem_rdata;
    wire signed [5:0] mv_x, mv_y;
    wire [15:0] sad;

    // 結果鎖存
    reg signed [5:0] latch_mv_x, latch_mv_y;
    reg [15:0] latch_sad;

    // ==========================================
    // 3. 記憶體行為 (關鍵修正!)
    // ==========================================
    always @(*) begin
        // 區分 Current Block 和 Reference Area
        if (mem_addr < 1000) begin 
            // 【維度對齊邏輯】
            // 當 WIDTH = 32 時：
            // mem_addr = Row * 32 + Col
            // mem_addr 的二進制結構為: [Row 4bits] [0] [Col 4bits]
            // 我們需要的 RAM Index = Row * 16 + Col
            // 也就是要取: [Row 4bits] [Col 4bits]
            // 所以我們把中間那個 0 跳過，直接拼起來：
            
            // mem_addr[8:5] 是 Row
            // mem_addr[3:0] 是 Col
            mem_rdata = ram[ {mem_addr[8:5], mem_addr[3:0]} ];
            
        end else begin 
            // Reference Area 資料產生
            mem_rdata = mem_addr[7:0]; 
        end
    end

    // ==========================================
    // 4. 主控制邏輯
    // ==========================================
    reg [2:0] tx_step; 

    always @(posedge clk_sys or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_RX;
            write_cnt <= 0;
            hex_start <= 0;
            tx_dv <= 0;
            tx_byte <= 0;
            tx_step <= 0;
            latch_mv_x <= 0; latch_mv_y <= 0; latch_sad <= 0;
        end else begin
            tx_dv <= 0; 

            case (state)
                // --- A. 接收輸入 ---
                S_RX: begin
                    if (rx_dv) begin
                        ram[write_cnt] <= rx_byte;
                        write_cnt <= write_cnt + 1;
                        
                        // Echo
                        tx_byte <= rx_byte;
                        tx_dv <= 1;

                        if (write_cnt == 255) begin
                            state <= S_CALC;
                        end
                    end
                end

                // --- B. 計算 ---
                S_CALC: begin
                    if (!hex_start && !hex_done) begin
                        hex_start <= 1;
                    end else if (hex_done) begin
                        hex_start <= 0;
                        latch_mv_x <= mv_x;
                        latch_mv_y <= mv_y;
                        latch_sad  <= sad;
                        state <= S_TX_SEND;
                        tx_step <= 0;
                    end
                end

                // --- C. 發送結果 ---
                S_TX_SEND: begin
                    if (!tx_active) begin
                        tx_dv <= 1; 
                        case (tx_step)
                            0: tx_byte <= {2'b00, latch_mv_x}; 
                            1: tx_byte <= {2'b00, latch_mv_y}; 
                            2: tx_byte <= latch_sad[15:8];     
                            3: tx_byte <= latch_sad[7:0];      
                            default: tx_byte <= 0;
                        endcase
                        state <= S_TX_WAIT_ACK; 
                    end
                end

                // --- D. 等待 ACK ---
                S_TX_WAIT_ACK: begin
                    if (rx_dv) begin
                        if (tx_step == 3) begin
                            state <= S_RX;  
                            write_cnt <= 0;
                        end else begin
                            tx_step <= tx_step + 1;
                            state <= S_TX_SEND; 
                        end
                    end
                end
            endcase
        end
    end

    // ==========================================
    // 5. 模組連接
    // ==========================================
    uart_rx #(.CLKS_PER_BIT(UART_CLKS)) u_rx (
        .clk(clk_sys), .i_Rx_Serial(RsRx), .o_Rx_DV(rx_dv), .o_Rx_Byte(rx_byte)
    );

    uart_tx #(.CLKS_PER_BIT(UART_CLKS)) u_tx (
        .clk(clk_sys), .i_Tx_DV(tx_dv), .i_Tx_Byte(tx_byte), 
        .o_Tx_Active(tx_active), .o_Tx_Serial(RsTx), .o_Tx_Done(tx_done)
    );

    // 【關鍵修改】強制將 WIDTH 設為 32，配合我們的位址解碼邏輯
    hexbs_top #( .WIDTH(32) ) u_core (
        .clk(clk_sys), .rst_n(rst_n), .start(hex_start),
        .mem_addr(mem_addr), .mem_rdata(mem_rdata),
        .frame_start_addr(32'd0), .ref_start_addr(32'd1000),  
        .mb_x_pos(32'd0), .mb_y_pos(32'd0),
        .mv_x(mv_x), .mv_y(mv_y), .sad(sad), .done(hex_done)
    );

    // ==========================================
    // 6. LED 儀表板
    // ==========================================
    assign led[15:8] = {2'b00, latch_mv_x};
    assign led[7:0]  = {2'b00, latch_mv_y};

endmodule