`timescale 1ns/1ns

module tb_counter;
    reg clk;
    reg rst;
    wire [3:0] count;

    // 呼叫你的電路 (Instantiate)
    counter uut (
        .clk(clk),
        .rst(rst),
        .count(count)
    );

    // 產生時脈 (Clock)
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 每 5ns 翻轉一次
    end

    // 測試流程
    initial begin
        // --- 關鍵設定：告訴模擬器要把波形存下來 ---
        $dumpfile("waveform.vcd"); // 檔案名稱叫 waveform.vcd
        $dumpvars(0, tb_counter);  // 紀錄 tb_counter 裡面的所有變數
        // ---------------------------------------

        rst = 1;       // 一開始重置
        #10 rst = 0;   // 10ns 後放開重置
        #100;          // 跑個 100ns
        $finish;       // 結束模擬
    end
endmodule