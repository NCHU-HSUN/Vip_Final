`timescale 1ns/1ps

// ============================================================================
// UART Receiver (RX) - 負責接收電腦傳來的資料
// 設定: 100MHz Clock, 115200 Baud Rate -> CLKS_PER_BIT = 868
// ============================================================================
module uart_rx #(parameter CLKS_PER_BIT = 868) (
    input clk,
    input i_Rx_Serial,
    output reg o_Rx_DV,    // Data Valid: 收到一個 Byte 時會 High 一個週期
    output reg [7:0] o_Rx_Byte
);
    localparam s_IDLE = 3'b000;
    localparam s_RX_START_BIT = 3'b001;
    localparam s_RX_DATA_BITS = 3'b010;
    localparam s_RX_STOP_BIT = 3'b011;
    localparam s_CLEANUP = 3'b100;

    reg [2:0] r_SM_Main = 0;
    reg [15:0] r_Clock_Count = 0;
    reg [2:0] r_Bit_Index = 0;
    reg r_Rx_Data_R = 1;
    reg r_Rx_Data = 1;

    // 雙重暫存器消除亞穩態 (Double Flop)
    always @(posedge clk) begin
        r_Rx_Data_R <= i_Rx_Serial;
        r_Rx_Data   <= r_Rx_Data_R;
    end

    always @(posedge clk) begin
        case (r_SM_Main)
            s_IDLE: begin
                o_Rx_DV <= 0;
                r_Clock_Count <= 0;
                r_Bit_Index <= 0;
                if (r_Rx_Data == 0) r_SM_Main <= s_RX_START_BIT; // 偵測到 Start Bit (Low)
            end
            s_RX_START_BIT: begin
                if (r_Clock_Count == (CLKS_PER_BIT-1)/2) begin
                    if (r_Rx_Data == 0) begin
                        r_Clock_Count <= 0;
                        r_SM_Main <= s_RX_DATA_BITS;
                    end else r_SM_Main <= s_IDLE;
                end else r_Clock_Count <= r_Clock_Count + 1;
            end
            s_RX_DATA_BITS: begin
                if (r_Clock_Count < CLKS_PER_BIT-1) begin
                    r_Clock_Count <= r_Clock_Count + 1;
                end else begin
                    r_Clock_Count <= 0;
                    o_Rx_Byte[r_Bit_Index] <= r_Rx_Data;
                    if (r_Bit_Index < 7) r_Bit_Index <= r_Bit_Index + 1;
                    else begin
                        r_Bit_Index <= 0;
                        r_SM_Main <= s_RX_STOP_BIT;
                    end
                end
            end
            s_RX_STOP_BIT: begin
                if (r_Clock_Count < CLKS_PER_BIT-1) begin
                    r_Clock_Count <= r_Clock_Count + 1;
                end else begin
                    o_Rx_DV <= 1; // 資料接收完成
                    r_Clock_Count <= 0;
                    r_SM_Main <= s_CLEANUP;
                end
            end
            s_CLEANUP: begin
                r_SM_Main <= s_IDLE;
                o_Rx_DV <= 0;
            end
            default: r_SM_Main <= s_IDLE;
        endcase
    end
endmodule

// ============================================================================
// UART Transmitter (TX) - 負責把資料傳回電腦
// ============================================================================
module uart_tx #(parameter CLKS_PER_BIT = 868) (
    input clk,
    input i_Tx_DV,         // Data Valid: 告訴模組"現在開始傳"
    input [7:0] i_Tx_Byte, // 要傳的資料
    output reg o_Tx_Active,// 忙碌中
    output reg o_Tx_Serial,// 序列輸出腳位
    output reg o_Tx_Done   // 傳送完成
);
    localparam s_IDLE = 3'b000;
    localparam s_TX_START_BIT = 3'b001;
    localparam s_TX_DATA_BITS = 3'b010;
    localparam s_TX_STOP_BIT = 3'b011;
    localparam s_CLEANUP = 3'b100;

    reg [2:0] r_SM_Main = 0;
    reg [15:0] r_Clock_Count = 0;
    reg [2:0] r_Bit_Index = 0;
    reg [7:0] r_Tx_Data = 0;

    always @(posedge clk) begin
        case (r_SM_Main)
            s_IDLE: begin
                o_Tx_Serial <= 1; // Idle 為 High
                o_Tx_Done <= 0;
                if (i_Tx_DV == 1) begin
                    o_Tx_Active <= 1;
                    r_Tx_Data <= i_Tx_Byte;
                    r_SM_Main <= s_TX_START_BIT;
                end else o_Tx_Active <= 0;
            end
            s_TX_START_BIT: begin
                o_Tx_Serial <= 0; // Start Bit 為 Low
                if (r_Clock_Count < CLKS_PER_BIT-1) r_Clock_Count <= r_Clock_Count + 1;
                else begin
                    r_Clock_Count <= 0;
                    r_SM_Main <= s_TX_DATA_BITS;
                end
            end
            s_TX_DATA_BITS: begin
                o_Tx_Serial <= r_Tx_Data[r_Bit_Index];
                if (r_Clock_Count < CLKS_PER_BIT-1) r_Clock_Count <= r_Clock_Count + 1;
                else begin
                    r_Clock_Count <= 0;
                    if (r_Bit_Index < 7) r_Bit_Index <= r_Bit_Index + 1;
                    else begin
                        r_Bit_Index <= 0;
                        r_SM_Main <= s_TX_STOP_BIT;
                    end
                end
            end
            s_TX_STOP_BIT: begin
                o_Tx_Serial <= 1; // Stop Bit 為 High
                if (r_Clock_Count < CLKS_PER_BIT-1) r_Clock_Count <= r_Clock_Count + 1;
                else begin
                    o_Tx_Done <= 1;
                    r_Clock_Count <= 0;
                    r_SM_Main <= s_CLEANUP;
                    o_Tx_Active <= 0;
                end
            end
            s_CLEANUP: begin
                o_Tx_Done <= 1;
                r_SM_Main <= s_IDLE;
            end
            default: r_SM_Main <= s_IDLE;
        endcase
    end
endmodule