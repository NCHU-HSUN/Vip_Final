// ============================================================================
// File Name: tb_ME_HEXBS.v
// Author: HSUN's AI Assistant
// Description: Testbench for ME_HEXBS
//              模擬環境，產生時脈並給予假資料
// ============================================================================
`timescale 1ns/1ps

module tb_ME_HEXBS;

    // 1. 訊號宣告
    reg clk;
    reg rst_n;
    reg i_start;

    // 連接到 DUT (Device Under Test) 的線
    wire [5:0]  o_cur_x, o_cur_y;
    reg  [7:0]  i_cur_pixel;
    
    wire [11:0] o_ref_x, o_ref_y;
    reg  [7:0]  i_ref_pixel;

    wire signed [5:0] o_mv_x, o_mv_y;
    wire [15:0] o_min_sad;
    wire        o_done;

    // 2. 實例化 (Instantiate) 你的硬體
    ME_HEXBS u_ME_HEXBS (
        .clk(clk),
        .rst_n(rst_n),
        .i_start(i_start),
        .o_cur_x(o_cur_x),
        .o_cur_y(o_cur_y),
        .i_cur_pixel(i_cur_pixel),
        .o_ref_x(o_ref_x),
        .o_ref_y(o_ref_y),
        .i_ref_pixel(i_ref_pixel),
        .o_mv_x(o_mv_x),
        .o_mv_y(o_mv_y),
        .o_min_sad(o_min_sad),
        .o_done(o_done)
    );

    // 3. 產生時脈 (Clock Generation) - 10ns 週期
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // 4. 模擬記憶體行為 (Mock Memory)
    // 當硬體改變地址時，我們給它一點假資料
    always @(*) begin
        // 簡單製造一個 Pattern：
        // 假設 Current Block 是一個漸層
        i_cur_pixel = o_cur_x + o_cur_y; 
        
        // 假設 Reference Frame 也是類似漸層，但在 (2,2) 的位置最像
        // 這樣我們預期 Motion Vector 應該會算出 (2,2) 或者接近的值
        i_ref_pixel = (o_ref_x - 100 - 2) + (o_ref_y - 100 - 2); 
    end

    // 5. 測試流程 (Test Stimulus)
    initial begin
        // 5.1 初始化
        rst_n = 1;
        i_start = 0;
        
        // 5.2 重置
        #20 rst_n = 0; // 按下 Reset
        #20 rst_n = 1; // 放開 Reset

        // 5.3 啟動運算
        #20 i_start = 1; // 告訴硬體開始跑
        #10 i_start = 0;

        // 5.4 等待完成
        wait(o_done == 1);
        
        // 5.5 顯示結果
        $display("========================================");
        $display("HEXBS Simulation Finished!");
        $display("Result Motion Vector: (%d, %d)", o_mv_x, o_mv_y);
        $display("Minimum SAD: %d", o_min_sad);
        $display("========================================");

        #100 $finish;
    end

    // 產生波形檔 (給 Vivado 或 Verdi 看)
    initial begin
        $dumpfile("hexbs_wave.vcd");
        $dumpvars(0, tb_ME_HEXBS);
    end

endmodule