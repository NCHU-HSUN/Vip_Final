`timescale 1ns/1ps
/*
 * =================================================================================
 * Smart_Top - 全功能 FPGA 測試平台 (版本 B)
 *
 * 設計變更日誌:
 * 1. 記憶體擴充: 從 256B 擴充至 172032B (352*240*2+)，足以容納兩個完整的
 *    CIF 影像幀 (352x240)。
 * 2. 通訊協議升級: 引入了指令驅動的 UART 協議，不再是簡單的數據流。
 *    - CMD_SET_FRAME_ADDRS (0xA2): 設定當前幀和參考幀的基地址。
 *    - CMD_WRITE_MEM (0xA0): 向指定地址寫入一塊數據。
 *    - CMD_RUN_TEST (0xA1): 傳入 MB 座標並啟動一次動態估計運算。
 * 3. 參數動態化: hexbs_top 核心的 `mb_x_pos`, `mb_y_pos`, `frame_start_addr`,
 *    `ref_start_addr` 等參數現在由 UART 指令動態設定，不再寫死。
 * 4. 記憶體介面: 讀取邏輯更新，現在從擴充後的 `video_ram` 讀取數據。
 * 5. 狀態機重構: 主狀態機 `state` 被重構以支援新的指令驅動協議。
 * =================================================================================
 */
module Smart_Top (
    input wire clk,         // 100MHz
    input wire btnC,        // Reset
    input wire RsRx,        // UART RX
    output wire RsTx,       // UART TX
    output wire [15:0] led  // 儀表板
);

    // ==========================================
    // 1. 系統參數與時脈
    // ==========================================
    localparam VIDEO_WIDTH  = 352;
    localparam VIDEO_HEIGHT = 240;
    localparam FRAME_SIZE   = VIDEO_WIDTH * VIDEO_HEIGHT; // 84480
    localparam TOTAL_MEM_SIZE = FRAME_SIZE * 2;          // 168960
    
    reg [3:0] clk_div;
    always @(posedge clk) clk_div <= clk_div + 1;
    wire clk_sys = clk_div[0]; // 50MHz, 增加時脈以加速數據傳輸

    localparam UART_CLKS = 434; // 115200 Baud @ 50MHz

    // ==========================================
    // 2. 系統訊號與狀態機
    // ==========================================
    wire rst_n = ~btnC; 

    // --- 指令集 ---
    localparam CMD_WRITE_MEM         = 8'hA0;
    localparam CMD_RUN_TEST          = 8'hA1;
    localparam CMD_SET_FRAME_ADDRS   = 8'hA2;
    localparam CMD_ECHO              = 8'hA3; // 用於測試連線

    // --- 狀態定義 ---
    localparam S_IDLE                  = 0;
    localparam S_CMD_SET_FRAME_PARAM   = 1;
    localparam S_CMD_WRITE_MEM_PARAM   = 2;
    localparam S_CMD_WRITE_MEM_DATA    = 3;
    localparam S_CMD_RUN_TEST_PARAM    = 4;
    localparam S_CALC                  = 5;
    localparam S_TX_SEND               = 6; 
    localparam S_TX_WAIT_ACK           = 7; 
    
    reg [3:0] state;
    reg [7:0] current_cmd;
    reg [6:0] param_cnt; // 參數接收計數器

    // ==========================================
    // 3. 記憶體與核心模組連線
    // ==========================================
    reg [7:0] video_ram [0:TOTAL_MEM_SIZE-1];  
    
    // --- UART ---
    wire rx_dv;
    wire [7:0] rx_byte;
    reg tx_dv;
    reg [7:0] tx_byte;
    wire tx_active, tx_done;
    
    // --- HEXBS Core ---
    reg hex_start;
    wire hex_done;
    wire [31:0] mem_addr;
    wire [7:0] mem_rdata; // 改為 wire，由 `video_ram` 驅動
    wire signed [5:0] mv_x, mv_y;
    wire [15:0] sad;

    // --- 動態參數 ---
    reg [31:0] reg_frame_start_addr;
    reg [31:0] reg_ref_start_addr;
    reg [31:0] reg_mb_x_pos;
    reg [31:0] reg_mb_y_pos;
    
    reg [31:0] temp_addr;
    reg [7:0]  temp_count;

    // --- 結果鎖存 ---
    reg signed [5:0] latch_mv_x, latch_mv_y;
    reg [15:0] latch_sad;

    // --- 記憶體讀取 ---
    // HEXBS 核心讀取記憶體的行為
    assign mem_rdata = video_ram[mem_addr];

    // ==========================================
    // 4. 主控制邏輯
    // ==========================================
    reg [2:0] tx_step; 

    always @(posedge clk_sys or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            current_cmd <= 0;
            param_cnt <= 0;
            hex_start <= 0;
            tx_dv <= 0;
            tx_byte <= 0;
            tx_step <= 0;
            reg_frame_start_addr <= 0;
            reg_ref_start_addr <= FRAME_SIZE;
            reg_mb_x_pos <= 0;
            reg_mb_y_pos <= 0;
            temp_addr <= 0;
            temp_count <= 0;
        end else begin
            tx_dv <= 0; 

            case (state)
                // --- A. 等待指令 ---
                S_IDLE: begin
                    if (rx_dv) begin
                        current_cmd <= rx_byte;
                        param_cnt <= 0;
                        case (rx_byte)
                            CMD_SET_FRAME_ADDRS: state <= S_CMD_SET_FRAME_PARAM;
                            CMD_WRITE_MEM:       state <= S_CMD_WRITE_MEM_PARAM;
                            CMD_RUN_TEST:        state <= S_CMD_RUN_TEST_PARAM;
                            CMD_ECHO: begin
                                tx_byte <= 8'hEE; // Echo back
                                tx_dv <= 1;
                            end
                            default: state <= S_IDLE; // 未知指令，忽略
                        endcase
                    end
                end

                // --- B. 處理指令參數 ---
                S_CMD_SET_FRAME_PARAM: begin
                    if (rx_dv) begin
                        case(param_cnt)
                            0: reg_frame_start_addr[31:24] <= rx_byte;
                            1: reg_frame_start_addr[23:16] <= rx_byte;
                            2: reg_frame_start_addr[15:8]  <= rx_byte;
                            3: reg_frame_start_addr[7:0]   <= rx_byte;
                            4: reg_ref_start_addr[31:24] <= rx_byte;
                            5: reg_ref_start_addr[23:16] <= rx_byte;
                            6: reg_ref_start_addr[15:8]  <= rx_byte;
                            7: reg_ref_start_addr[7:0]   <= rx_byte;
                        endcase
                        param_cnt <= param_cnt + 1;
                        if (param_cnt == 7) state <= S_IDLE;
                    end
                end
                
                S_CMD_WRITE_MEM_PARAM: begin
                    if (rx_dv) begin
                        case(param_cnt)
                            0: temp_addr[23:16] <= rx_byte; // MSB of address
                            1: temp_addr[15:8]  <= rx_byte;
                            2: temp_addr[7:0]   <= rx_byte; // LSB of address
                            3: temp_count       <= rx_byte; // Number of bytes to write
                        endcase
                        param_cnt <= param_cnt + 1;
                        if (param_cnt == 3) begin
                            if (temp_count > 0)
                                state <= S_CMD_WRITE_MEM_DATA;
                            else
                                state <= S_IDLE;
                        end
                    end
                end

                S_CMD_WRITE_MEM_DATA: begin
                    if (rx_dv) begin
                        video_ram[temp_addr + param_cnt] <= rx_byte;
                        param_cnt <= param_cnt + 1;
                        if (param_cnt == temp_count - 1) begin
                            state <= S_IDLE;
                        end
                    end
                end

                S_CMD_RUN_TEST_PARAM: begin
                     if (rx_dv) begin
                        case(param_cnt)
                            0: reg_mb_x_pos[15:8] <= rx_byte;
                            1: reg_mb_x_pos[7:0]  <= rx_byte;
                            2: reg_mb_y_pos[15:8] <= rx_byte;
                            3: reg_mb_y_pos[7:0]  <= rx_byte;
                        endcase
                        param_cnt <= param_cnt + 1;
                        if (param_cnt == 3) state <= S_CALC;
                    end
                end

                // --- C. 計算 ---
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

                // --- D. 發送結果 ---
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

                // --- E. 等待 ACK & 完成 ---
                S_TX_WAIT_ACK: begin
                    if (rx_dv) begin // Python should send an ACK
                        if (tx_step == 3) begin
                            state <= S_IDLE;  // A test is done, wait for next command
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

    hexbs_top #( 
        .WIDTH(VIDEO_WIDTH), 
        .HEIGHT(VIDEO_HEIGHT) 
    ) u_core (
        .clk(clk_sys), .rst_n(rst_n), .start(hex_start),
        .mem_addr(mem_addr), .mem_rdata(mem_rdata),
        .frame_start_addr(reg_frame_start_addr), 
        .ref_start_addr(reg_ref_start_addr),  
        .mb_x_pos(reg_mb_x_pos), 
        .mb_y_pos(reg_mb_y_pos),
        .mv_x(mv_x), .mv_y(mv_y), .sad(sad), .done(hex_done)
    );

    // ==========================================
    // 6. LED 儀表板
    // ==========================================
    // Display current state on LEDs
    assign led[15:12] = state;
    assign led[11:8] = current_cmd[3:0];
    assign led[7:0] = param_cnt;

endmodule
