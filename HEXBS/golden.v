`timescale 1ns/1ps

module tb_system_verify;

    // ==========================================
    // 1. 參數設定 (必須與 Python gen_golden.py 一致)
    // ==========================================
    parameter FRAME_WIDTH  = 352;  
    parameter FRAME_HEIGHT = 240; 
    parameter MB_SIZE      = 16;
    parameter SEARCH_R     = 16;   
    
    localparam FRAME_SIZE  = FRAME_WIDTH * FRAME_HEIGHT;
    localparam MEM_DEPTH   = FRAME_SIZE * 3; 

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
        cur_mb_x = 0; cur_mb_y = 0;

        // --- 載入記憶體 (注意路徑) ---
        $display("--------------------------------------------");
        $display("Loading Video Memory from '../golden_patterns/full_video.hex'...");
        
        // 這裡會讀取 hex 檔。如果你的 python 腳本沒有完整產生資料，這裡就會報 Warning。
        // 請務必確認 full_video.hex 的行數是否足夠 (至少要有 352*240*2 = 168960 行)
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

                // --- 修正：Python 已經給出像素座標，不需要再乘以 MB_SIZE ---
                cur_mb_x = exp_mb_col; 
                cur_mb_y = exp_mb_row;
                
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
                    $display("[FAIL] MB(%3d, %3d) | Exp MV:(%2d, %2d) | HW MV:(%2d, %2d) | SAD Exp:%5d HW:%5d", 
                             exp_mb_row, exp_mb_col, exp_mv_x, exp_mv_y, hw_mv_x, hw_mv_y, exp_sad, hw_sad);
                    err_count = err_count + 1;
                    // 為了除錯，我們允許它多報幾個錯再停，設為 50
                    if (err_count > 50) begin
                         $display("Too many errors. Stopping.");
                         $finish;
                    end
                end 
                else begin
                    if (total_mbs % 100 == 0) $display("[PASS] Processed %0d MBs...", total_mbs);
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
        $finish;
    end
endmodule