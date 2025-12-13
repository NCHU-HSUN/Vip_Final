`timescale 1ns/1ps

`ifndef WAVEFILE
`define WAVEFILE "waveform.vcd"
`endif

`ifndef ENABLE_WAVE_DUMP_DEFAULT
`define ENABLE_WAVE_DUMP_DEFAULT 0
`endif

module tb_system_verify;

    // ==========================================
    // 1. 參數設定 (必須與 Python gen_golden.py 一致)
    // ==========================================
    parameter FRAME_WIDTH   = 352;  
    parameter FRAME_HEIGHT  = 240; 
    parameter MB_SIZE       = 16;
    parameter SEARCH_R      = 32;   // 對應論文 ±32 搜尋範圍
    parameter TOTAL_FRAMES  = 115;   // must cover all frames present in full_video.hex
    
    localparam FRAME_SIZE  = FRAME_WIDTH * FRAME_HEIGHT;
    localparam MEM_DEPTH   = FRAME_SIZE * TOTAL_FRAMES; 

    // ==========================================
    // 2. 訊號宣告
    // ==========================================
    reg clk, rst_n, start;
    wire done;
    wire signed [5:0] hw_mv_x, hw_mv_y;
    wire [15:0] hw_sad;
    wire [31:0] dut_mem_addr;
    wire [7:0]  dut_mem_rdata;

    reg [31:0] cur_frame_start;
    reg [31:0] ref_frame_start; 
    reg [31:0] cur_mb_x, cur_mb_y;
    reg [7:0] dram_mem [0:MEM_DEPTH-1];

    integer file_ans, scan_res;
    integer exp_frame_idx, exp_mb_row, exp_mb_col, exp_mv_x, exp_mv_y, exp_sad;
    integer err_count = 0, total_mbs = 0, watchdog = 0;
    
    reg [8*100:1] dummy_line; 

    // ==========================================
    // 3. 實例化 DUT (Device Under Test)
    // ==========================================
    assign dut_mem_rdata = dram_mem[dut_mem_addr];

    hexbs_top u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .mem_addr(dut_mem_addr),
        .mem_rdata(dut_mem_rdata),
        .frame_start_addr(cur_frame_start), 
        .ref_start_addr(ref_frame_start),    
        .mb_x_pos(cur_mb_x),                
        .mb_y_pos(cur_mb_y),                
        .mv_x(hw_mv_x),
        .mv_y(hw_mv_y),
        .sad(hw_sad),
        .done(done)
    );

    // ==========================================
    // 4. 時脈與測試流程
    // ==========================================
    always #5 clk = ~clk; 

    initial begin
        // --- 初始化 ---
        clk = 0; rst_n = 0; start = 0;
        cur_frame_start = FRAME_SIZE; 
        ref_frame_start = 0;
        cur_mb_x = 0; cur_mb_y = 0;

        // --- 載入記憶體 (注意路徑) ---
        $display("--------------------------------------------");
        $display("Loading Video Memory from '../golden_patterns/full_video.hex'...");
        
        // 請務必確認 full_video.hex 的行數是否足夠
        $readmemh("../golden_patterns/full_video.hex", dram_mem);
        
        // --- 開啟 Trace 檔 ---
        file_ans = $fopen("../golden_patterns/golden_trace.txt", "r");
        if (file_ans == 0) begin
            $display("Error: Could not open golden_trace.txt at ../golden_patterns/");
            $finish;
        end

        #100 rst_n = 1; #20;
        $display("System Reset Done. Starting Verification...");

        // --- 測試迴圈 ---
        while (!$feof(file_ans)) begin
            
            // 嘗試依照 Python 的格式讀取
            scan_res = $fscanf(file_ans, "Frame=%d MB_Row=%d MB_Col=%d | MV_X=%d MV_Y=%d SAD=%d\n", 
                               exp_frame_idx, exp_mb_row, exp_mb_col, exp_mv_x, exp_mv_y, exp_sad);
            
            if (scan_res == 6) begin
                // 讀取成功
                total_mbs = total_mbs + 1;

                cur_mb_x = exp_mb_col; 
                cur_mb_y = exp_mb_row;
                
                // Dynamic Frame Addressing
                cur_frame_start = exp_frame_idx * FRAME_SIZE;
                ref_frame_start = (exp_frame_idx - 1) * FRAME_SIZE;

                if ((cur_frame_start + FRAME_SIZE) > MEM_DEPTH ||
                    (ref_frame_start + FRAME_SIZE) > MEM_DEPTH) begin
                    $display("Error: Frame %0d exceeds allocated MEM_DEPTH=%0d. Increase TOTAL_FRAMES or regenerate data.",
                             exp_frame_idx, MEM_DEPTH);
                    $finish;
                end

                // 啟動硬體
                @(posedge clk); start = 1;
                @(posedge clk); start = 0;
                
                // 等待完成
                watchdog = 0;
                while (!done && watchdog < 50000) begin
                    @(posedge clk); watchdog = watchdog + 1;
                end
                if (watchdog >= 50000) begin
                    $display("Error: Timeout at MB(%0d, %0d)!", exp_mb_row, exp_mb_col);
                    $finish;
                end
                
                // 比對 MV
                if (hw_mv_x !== exp_mv_x || hw_mv_y !== exp_mv_y) begin
                    // [FAIL] 情況，格式維持原樣
                    $display("[FAIL] MB(%3d, %3d) | Exp MV:(%2d, %2d) | HW MV:(%2d, %2d) | SAD Exp:%5d HW:%5d", 
                             exp_mb_row, exp_mb_col, exp_mv_x, exp_mv_y, hw_mv_x, hw_mv_y, exp_sad, hw_sad);
                    err_count = err_count + 1;
                    
                    if (err_count > 50) begin
                         $display("Too many errors. Stopping.");
                         $finish;
                    end
                end 
                else begin
                    // [PASS] 情況：修改這裡！
                    // 原本是每 100 次顯示一次，現在改成每次都顯示詳細資訊，格式模仿上面 FAIL 的樣子
                    $display("[PASS] MB(%3d, %3d) | Exp MV:(%2d, %2d) | HW MV:(%2d, %2d) | SAD Exp:%5d HW:%5d", 
                             exp_mb_row, exp_mb_col, exp_mv_x, exp_mv_y, hw_mv_x, hw_mv_y, exp_sad, hw_sad);
                end
            end 
            else begin
                // 讀取失敗（標題行），吃掉這一行
                scan_res = $fgets(dummy_line, file_ans);
            end
        end 

        // --- 結算 ---
        $display("--------------------------------------------");
        $display("Total MBs: %0d, Errors: %0d", total_mbs, err_count);
        if (err_count == 0 && total_mbs > 0) $display("Result: PERFECT MATCH! (恭喜 HSUN)");
        else $display("Result: FAILED.");
        $display("--------------------------------------------");
        
        // --- 執行額外的邊界條件測試並存檔 ---
        run_boundary_tests();

        $finish;
    end

    // ==========================================
    // 5. 新增功能：邊界與特定狀況測試 (含波形存檔)
    // ==========================================
    task run_boundary_tests;
        integer i, r, c;
        reg [7:0] pixel_val;
        begin
            $display("\n[Extra Test] Starting Boundary Condition Tests (Waveform Enabled)...");
            
            // 1. 啟用波形存檔 (只存這一段)
            $dumpfile("waveform_boundary.vcd");
            $dumpvars(0, tb_system_verify);

            // 重置記憶體指標 (使用記憶體前段作為測試區)
            ref_frame_start = 0;
            cur_frame_start = FRAME_SIZE;
            cur_mb_x = 10; // 選擇中間一點的位置避免邊界問題干擾觀察
            cur_mb_y = 10;

            // -------------------------------------------------------
            // 測試狀況 A: 靜止畫面 (Zero Motion)
            // 預期結果: MV=(0,0), SAD=0
            // -------------------------------------------------------
            $display("[Extra Test] Case A: Zero Motion (Static Pattern)");
            // 填入簡單的測試圖樣 (全部填 100)
            for (i = 0; i < FRAME_SIZE*2; i = i + 1) dram_mem[i] = 8'd100;

            // 執行 DUT
            @(posedge clk); start = 1;
            @(posedge clk); start = 0;
            wait(done);
            $display("    -> Result: MV=(%d, %d), SAD=%d", hw_mv_x, hw_mv_y, hw_sad);

            // -------------------------------------------------------
            // 測試狀況 B: 移動測試 (Motion Test)
            // 模擬物體向右下方移動 (MV應該接近正值)
            // -------------------------------------------------------
            $display("[Extra Test] Case B: Motion Test (Object Moved)");
            
            // 清空背景為 0
            for (i = 0; i < FRAME_SIZE*2; i = i + 1) dram_mem[i] = 8'd0;

            // 在 Reference Frame 的 (MB_X, MB_Y) 畫一個亮塊 (值255)
            // 座標轉換: addr = start + (y*width) + x
            for (r = 0; r < 16; r = r + 1) begin
                for (c = 0; c < 16; c = c + 1) begin
                    dram_mem[ref_frame_start + (cur_mb_y*16 + r)*FRAME_WIDTH + (cur_mb_x*16 + c)] = 8'd255;
                end
            end

            // 在 Current Frame 的 (MB_X+5, MB_Y+3) 畫同樣的亮塊 -> 預期 MV 約為 (5, 3)
            for (r = 0; r < 16; r = r + 1) begin
                for (c = 0; c < 16; c = c + 1) begin
                    dram_mem[cur_frame_start + (cur_mb_y*16 + 3 + r)*FRAME_WIDTH + (cur_mb_x*16 + 5 + c)] = 8'd255;
                end
            end

            // 執行 DUT
            @(posedge clk); start = 1;
            @(posedge clk); start = 0;
            wait(done);
            $display("    -> Result: MV=(%d, %d), SAD=%d", hw_mv_x, hw_mv_y, hw_sad);

            // 關閉波形
            $dumpoff;
            $display("[Extra Test] Done. Waveform saved to 'waveform_boundary.vcd'\n");
        end
    endtask

endmodule
