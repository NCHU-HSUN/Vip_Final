// ============================================================================
// File Name: tb_ME_HEXBS.v
// Description: 使用 XOR 圖案的測試平台，確保 (2,2) 是唯一解
//              (已修正：移除 SystemVerilog 語法，改用傳統 Verilog 以確保相容性)
// ============================================================================
`timescale 1ns/1ps

module tb_ME_HEXBS;

    reg clk, rst_n, i_start;
    wire [5:0]  o_cur_x, o_cur_y;
    reg  [7:0]  i_cur_pixel;
    wire [11:0] o_ref_x, o_ref_y;
    reg  [7:0]  i_ref_pixel;
    wire signed [5:0] o_mv_x, o_mv_y;
    wire [15:0] o_min_sad;
    wire        o_done;

    // 實例化你的硬體
    ME_HEXBS u_ME_HEXBS (
        .clk(clk), .rst_n(rst_n), .i_start(i_start),
        .o_cur_x(o_cur_x), .o_cur_y(o_cur_y), .i_cur_pixel(i_cur_pixel),
        .o_ref_x(o_ref_x), .o_ref_y(o_ref_y), .i_ref_pixel(i_ref_pixel),
        .o_mv_x(o_mv_x), .o_mv_y(o_mv_y), .o_min_sad(o_min_sad), .o_done(o_done)
    );

    // 時脈產生
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // --- 測試圖案 (XOR Pattern) ---
    always @(*) begin
        // Current: (x ^ y) * 8
        i_cur_pixel = (o_cur_x ^ o_cur_y) * 8; 
        
        // Reference: ((x-2) ^ (y-2)) * 8
        // 只有當 o_ref_x = x+2 且 o_ref_y = y+2 時，才會跟上面的 Current 一樣
        i_ref_pixel = ((o_ref_x - 100 - 2) ^ (o_ref_y - 100 - 2)) * 8; 
    end

    // --- 安全機制：超時強制結束 (Watchdog) ---
    // 這是第二條平行的時間線，如果跑太久 (100萬奈秒)，它會強制殺掉模擬
    initial begin
        #1000000; 
        $display("\nError: Simulation Timed Out! (硬體可能卡死或跑太慢)");
        $finish;
    end

    // --- 主測試流程 ---
    initial begin
        // 1. 初始化與重置
        rst_n = 1; i_start = 0;
        #20 rst_n = 0; 
        #20 rst_n = 1; 

        // 2. 啟動硬體
        #20 i_start = 1; 
        #10 i_start = 0;

        // 3. 等待硬體說「我做完了」(o_done 變成 1)
        wait(o_done == 1);
        
        // 4. 顯示結果並檢查
        $display("\n========================================");
        $display("HEXBS Simulation Results (XOR Pattern)");
        $display("Target MV: ( 2,  2)");
        $display("Actual MV: (%2d, %2d)", o_mv_x, o_mv_y);
        $display("Minimum SAD: %d", o_min_sad);
        $display("----------------------------------------");
        
        // 這裡用 === 是為了連不定值 (X) 都能比對出來
        if (o_mv_x === 6'sd2 && o_mv_y === 6'sd2 && o_min_sad === 0)
            $display("✅ PASS: 恭喜！完美命中 (2, 2)！");
        else
            $display("❌ FAIL: 結果不對，請檢查波形。");
        $display("========================================\n");

        // 5. 結束模擬 (這會順便把上面的計時炸彈關掉)
        #100 $finish;
    end

    // 產生波形檔
    initial begin
        $dumpfile("hexbs_wave.vcd");
        $dumpvars(0, tb_ME_HEXBS);
    end

endmodule