/*
 * Flexible 8-bit Streaming MAC Unit  --  tt_um_flex_mac
 * SPDX-License-Identifier: Apache-2.0
 *
 * ============================================================================
 *  WHAT THIS IS
 * ============================================================================
 *  A weight-stationary multiply-accumulate engine: 9 weights are loaded once
 *  and stay on-chip; activations stream in on ui_in, one per clock, and are
 *  multiplied by the "current" weight and summed into a 24-bit accumulator.
 *  A dot product of a 1x1, 2x2 or 3x3 window is produced every N cycles
 *  (N = 1, 4, 9) with zero dead cycles between windows.
 *
 *  "Weight-stationary" is the key idea from systolic AI accelerators (TPU
 *  etc.): moving data costs far more energy/IO than computing on it. Weights
 *  are reused for every window, so load them ONCE and keep them local; only
 *  the activations (which change every cycle anyway) cross the IO pins.
 *
 * ============================================================================
 *  HOST PROTOCOL  (all inputs sampled on the rising edge of clk)
 * ============================================================================
 *  Every input goes through one register stage (see "INPUT REGISTERS"), so
 *  every effect appears exactly ONE cycle later than without that stage. Because
 *  the delay is the same for all pins, the host never has to account for it:
 *  just keep driving one command per clock.
 *
 *  1. Reset    : hold rst_n low for >= 4 clocks (TT firmware uses 10).
 *  2. Load     : mode=1, put 9 bytes on ui_in on 9 consecutive clocks. The FIRST
 *                byte loaded is the first weight used. For 2x2 or 1x1 windows
 *                the first 4 or 1 bytes are the kernel; the rest are padding.
 *  3. Compute  : mode=0, valid_in=1, one activation per clock.
 *  4. Readout  : raise latch_and_clear on the clock AFTER the last activation of
 *                a window. That clock may also carry valid_in=1 with the first
 *                activation of the next window: the shadow register gets the
 *                finished sum while the accumulator restarts at that product.
 *                Then set read_select = 0,1,2 to read bytes [7:0],[15:8],[23:16]
 *                from uo_out (also delayed one cycle by the input register).
 *
 * ============================================================================
 *  SIGNOFF RESULTS (LibreLane 3.0.14, sky130A, TT 1x1 tile, CLOCK_PERIOD=20ns)
 * ============================================================================
 *    Cells (excl. fill/tap) : 967   (137 flops, 157 hold-delay, 67 clock-tree)
 *    Utilisation            : 61.6 %
 *    Setup slack @ 20 ns    : +2.56 ns worst (ss 100C 1.60V), +11.3 ns typical
 *    Hold slack             : +0.117 ns worst (ff -40C 1.95V)
 *    LVS / DRC / TT precheck: clean;  RTL + gate-level cocotb tests: pass
 *
 *  Design rules used:
 *    - Setup must close at the SLOW corner (ss, 100C, 1.60V), not only typical.
 *      Chips that only meet timing at typical fail when they are hot or the
 *      supply voltage droops.
 *    - Hold must close at the FAST corner (ff, -40C, 1.95V); see the shift
 *      register notes below.
 *    - The remaining max-slew warnings appear only at the ss corner (the
 *      resizer repairs slew at the typical corner). They sit on the high-fanout
 *      nets into the multiplier and are already included in the +2.56 ns.
 *
 *  How the design got here (measured, same flow):
 *    v1 spec-literal, inputs straight from pads : 997 cells, +1.64 ns @ ss
 *    v2 single-adder accum + reset sync         : 972 cells, +0.51 ns @ ss
 *       (fewer cells, but the pad -> multiplier path got slower)
 *    v3 + input register stage                  : 987 cells, +3.42 ns @ ss
 *    v4 (this) + shadow reset trick, 3:1 mux    : 967 cells, +2.56 ns @ ss
 *       (v3 -> v4 slack difference is place-and-route variation on the same
 *        flop-to-flop path, not a logic change)
 */

`default_nettype none   // A typo'd net name becomes a compile ERROR instead of
                        // a silently-created 1-bit wire. Always use this.

module tt_um_flex_mac (
    input  wire [7:0] ui_in,    // Data bus: weights (mode=1) / activations (mode=0)
    output wire [7:0] uo_out,   // One byte of the 24-bit result, chosen by read_select
    input  wire [7:0] uio_in,   // Control bus (all 8 bidirectional pins are inputs)
    output wire [7:0] uio_out,  // Unused -> 0
    output wire [7:0] uio_oe,   // 0 = input. We never drive the bidir pins.
    input  wire       ena,      // Always 1 when powered on TT -> ignored
    input  wire       clk,      // 50 MHz target
    input  wire       rst_n     // Active-low reset from the TT controller
);

  // ==========================================================================
  //  INPUT REGISTERS
  // ==========================================================================
  //  WHY: Tiny Tapeout routes every pin through a shared mux tree before it
  //  reaches the design. TT's SDC models this as `set_input_delay 4.0` (input
  //  data arrives 4 ns after the clock edge), so with direct inputs the path
  //    pad -> 8x8 multiplier -> 24-bit adder -> accum_reg
  //  only gets 16 ns. Measured at the slow corner it left 0.51 ns of slack,
  //  too little margin for a real chip.
  //
  //  Adding one register stage right at the pins splits that into
  //    pad -> data_q             (almost no logic, easily meets 16 ns)
  //    data_q -> mult -> add     (gets the full 20 ns clock period)
  //  Result at the slow corner: 3.42 ns of slack. Cost: 15 flip-flops and one
  //  cycle of latency.
  //
  //  RULE OF THUMB: register inputs and outputs at a chip or block boundary
  //  when you don't control the timing outside it. The timing then depends
  //  only on paths inside the block, which the tools can see and optimise.
  //
  //  These registers have no reset: they are reloaded every clock, so
  //  whatever they start with is overwritten after one cycle. Leaving out
  //  reset logic you don't need saves cells and routing.
  reg [7:0] data_q;
  reg [6:0] ctrl_q;   // uio_in[7] is unused -> not registered at all

  always @(posedge clk) begin
    data_q <= ui_in;
    ctrl_q <= uio_in[6:0];
  end

  wire       mode            = ctrl_q[0];
  wire       valid_in        = ctrl_q[1];
  wire       latch_and_clear = ctrl_q[2];
  wire [1:0] window_size     = ctrl_q[4:3];
  wire [1:0] read_select     = ctrl_q[6:5];

  // ==========================================================================
  //  RESET SYNCHRONIZER
  // ==========================================================================
  //  WHY SYNCHRONOUS RESET: an asynchronous reset is a second, untimed path
  //  into every flop. It needs reset-recovery/removal checks and its own
  //  buffered tree that competes with the clock tree for routing. A
  //  synchronous reset is just another data input to the flop: STA times it
  //  like any other logic and placement treats it like any other net.
  //
  //  WHY SYNCHRONIZE IT: rst_n comes from off-chip and is not aligned to our
  //  clk. If it changes right at a clock edge, some flops may see the change
  //  this cycle and others the next (or go metastable). Passing it through two
  //  flops means the whole design sees one clean reset signal that changes
  //  exactly on a clock edge. It also means the pad drives only 1 flop instead
  //  of dozens (lower fanout -> better slew).
  //
  //  Cost: 2 flops, plus 2 cycles of extra reset latency (harmless).
  reg [1:0] rst_sync;
  always @(posedge clk) rst_sync <= {rst_sync[0], rst_n};
  wire rst = ~rst_sync[1];

  // ==========================================================================
  //  CONTROL DECODE
  // ==========================================================================
  //  `compute` = "a MAC happens this cycle". Gating it with ~rst ensures
  //  nothing is accumulated or rotated while in reset.
  wire compute = ~mode & valid_in & ~rst;

  // Decode only the bits we actually need: 2'b10 and 2'b11 both mean 3x3,
  // so window_size[1] alone selects 3x3 (one wire, no gate).
  wire win1 = (window_size == 2'b00);
  wire win2 = (window_size == 2'b01);
  wire win3 = window_size[1];

  // ==========================================================================
  //  WEIGHT RING  (9 x 8-bit circular shift register)
  // ==========================================================================
  //  The multiplier always reads weight_reg[0]. After each MAC the ring
  //  rotates, bringing the next weight into position 0.
  //
  //  WHY A RING AND NOT AN ADDRESSED REGISTER FILE + POINTER?
  //    Register file : 72 flops + 9:1 read mux on 8 bits (~64 mux2)
  //                    + 4-bit wrap-around counter + compare logic.
  //    Ring          : 72 flops + one 2:1 mux per bit.
  //  Both need the per-bit mux because a register must also be able to HOLD
  //  its value. The ring replaces the large read mux and the counter with
  //  plain wires between neighbours. Shift registers are cheap, and their
  //  wires are short because the placer puts neighbouring stages next to
  //  each other.
  //
  //  Loading (mode=1): plain shift-in, weight_reg[8] <= data bus.
  //    After 9 loads the first byte loaded is in weight_reg[0].
  //
  //  Compute rotation (on valid MAC), ring length N:
  //    1x1 (N=1): nothing moves, w[0] is reused every cycle.
  //    2x2 (N=4): w[0..2] <= w[1..3], w[3] <= w[0]; w[4..8] hold.
  //    3x3 (N=9): w[0..7] <= w[1..8], w[8] <= w[0].
  //  The ring can be shortened by changing only where the wrap-around
  //  feedback lands (w[3] or w[8]). Only those two bytes need an extra 2:1
  //  mux; every other byte has exactly one possible source.
  //
  //  Two enables group the registers by which windows move them:
  //    en_lo : w[0..3] move in load mode, 2x2 and 3x3
  //    en_hi : w[4..8] move in load mode and 3x3 only
  //  Using shared enables instead of per-register logic keeps the control
  //  logic tiny.
  //
  //  WHY NO RESET ON THE WEIGHTS: they are always fully loaded before use, so
  //  resetting them adds ~72 gates (7% of the cell budget) for nothing.
  //  This only works because the accumulator ignores the product when
  //  compute=0 (see the product mask below), so garbage weights after
  //  power-up can never leak into a result.
  //
  //  WHY YOSYS CANNOT DELETE THE RING: synthesis only removes logic whose
  //  output never affects a module output. Every weight bit reaches uo_out
  //  via w[0] -> multiplier -> accum -> shadow -> uo_out, so all 72 flops
  //  are kept without needing a (* keep *) attribute. (A forced keep can
  //  hide real bugs, so it's better to make sure the logic is genuinely used.)
  //
  //  WHY A MUX ENABLE AND NOT CLOCK GATING: gating the clock to w[*] would save
  //  the 72 muxes, but a hand-made clock gate glitches unless you use a
  //  proper integrated clock-gating cell, it breaks the CTS/STA assumptions
  //  of the TT flow, and the saving would not be worth that risk. Keep one
  //  clock and use data enables.
  //
  //  HOLD-TIME NOTE (worth understanding): the path w[i+1] -> w[i] contains
  //  almost no logic. If clock skew makes w[i] see the edge slightly LATER
  //  than w[i+1], the new value of w[i+1] can race through and corrupt w[i]
  //  in the same cycle (a hold violation). The router fixes this by adding
  //  delay cells (dlygate4sd3) on such short paths. That is why ~150 of the
  //  cells are hold buffers. It is normal for shift registers: setup is
  //  limited by long paths (the multiplier), hold by short ones (shift chains).
  //
  //  Yosys prints "Replacing memory \weight_reg with list of registers".
  //  That is the result we want: an array that is shifted as a whole cannot be
  //  a RAM macro, so Yosys turns it into 72 ordinary flops.
  reg [7:0] weight_reg [0:8];

  wire en_lo = mode | (compute & ~win1);
  wire en_hi = mode | (compute &  win3);

  always @(posedge clk) begin
    if (en_lo) begin
      weight_reg[0] <= weight_reg[1];
      weight_reg[1] <= weight_reg[2];
      weight_reg[2] <= weight_reg[3];
      weight_reg[3] <= (~mode & win2) ? weight_reg[0] : weight_reg[4]; // 2x2 wrap
    end
    if (en_hi) begin
      weight_reg[4] <= weight_reg[5];
      weight_reg[5] <= weight_reg[6];
      weight_reg[6] <= weight_reg[7];
      weight_reg[7] <= weight_reg[8];
      weight_reg[8] <= mode ? data_q : weight_reg[0];                  // load / 3x3 wrap
    end
  end

  // ==========================================================================
  //  MULTIPLIER
  // ==========================================================================
  //  Unsigned 8x8 -> 16 bits. Writing `*` lets Yosys choose the multiplier
  //  structure, and ABC then maps it to sky130 cells. Hand-writing a
  //  multiplier rarely beats that.
  //  This is the critical (slowest) path:
  //    data_q -> partial products -> carry chain -> 24-bit add -> accum_reg
  //
  //  PRODUCT MASK (& {16{compute}}): when no MAC happens the accumulator
  //  must add 0. Gating the activation (8 ANDs) would be smaller in silicon,
  //  but in simulation X * 0 = X: before the weights are loaded weight_reg[0]
  //  is X, and that X would pollute accum_reg forever. X & 0 = 0 in both RTL
  //  and gate-level simulation, so masking the product keeps the sims
  //  X-clean for a cost of only 8 extra AND gates.
  //  Lesson: hardware that works in silicon can still fail in sim because
  //  of X propagation. Design so simulation stays deterministic, because the
  //  TT gate-level CI test runs that same simulation.
  wire [15:0] product = (data_q * weight_reg[0]) & {16{compute}};

  // ==========================================================================
  //  ACCUMULATOR  (single adder, no hold mux)
  // ==========================================================================
  //  The spec has three cases: accumulate, clear (latch), and clear-and-
  //  load-product (latch + valid). All three, plus "hold" and reset, collapse
  //  into ONE expression:
  //
  //      accum_next = (clear ? 0 : accum) + (compute ? product : 0)
  //
  //    hold      : clear=0, compute=0 -> accum + 0       = accum
  //    accumulate: clear=0, compute=1 -> accum + product
  //    latch     : clear=1, compute=0 -> 0 + 0           = 0
  //    latch+MAC : clear=1, compute=1 -> 0 + product     (next window starts)
  //    reset     : clear=1 (rst), compute=0 (rst gates it) -> 0
  //
  //  Because adding 0 already holds the value, the register needs no enable
  //  mux. It costs 24 AND gates instead of 24 mux2 + a separate load path.
  //  Lesson: before writing if/else trees, check whether one arithmetic
  //  expression covers every case. Fewer special cases means fewer muxes.
  //
  //  WIDTH: worst case for one 3x3 window is 9 * 255 * 255 = 585,225, which
  //  needs 20 bits. 24 bits gives 16x headroom (~258 full-scale MACs before
  //  wrap-around), so a host can also accumulate across several windows.
  reg [23:0] accum_reg;
  reg [23:0] shadow_reg;

  wire clear = latch_and_clear | rst;

  always @(posedge clk) begin
    accum_reg <= (accum_reg & {24{~clear}}) + {8'd0, product};
  end

  // ==========================================================================
  //  SHADOW REGISTER  (double buffer for readout)
  // ==========================================================================
  //  WHY: the host reads 3 bytes over 3 cycles while the next window is
  //  already accumulating. Copying the result into a separate register lets
  //  compute and readout overlap: the datapath never stalls (1 MAC/cycle,
  //  every cycle). This is the same idea as a frame buffer's front/back buffer.
  //
  //  RESET TRICK: there is no separate "shadow <= 0" branch. During reset,
  //  accum_reg is cleared on the first reset cycle, and because rst also
  //  enables the capture, shadow copies that 0 on the next one. One shared
  //  enable replaces 24 reset AND gates. This needs reset held for >= 2
  //  synchronized cycles, which the protocol guarantees.
  always @(posedge clk) begin
    if (latch_and_clear | rst)
      shadow_reg <= accum_reg;
  end

  // ==========================================================================
  //  READOUT MUX
  // ==========================================================================
  //  A tree of two 2:1 muxes. read_select = 2'b11 deliberately mirrors the top
  //  byte, so the 4th case needs no "output 0" decode logic. Spending no
  //  gates on input values the protocol never uses is free area.
  //  The output comes straight from flops (shadow_reg, ctrl_q) through a
  //  two-level mux, so the uo_out timing is short and predictable.
  assign uo_out = read_select[1] ? shadow_reg[23:16]
                : read_select[0] ? shadow_reg[15:8]
                :                  shadow_reg[7:0];

  // Bidirectional pins are all inputs: never drive them.
  assign uio_out = 8'd0;
  assign uio_oe  = 8'd0;

  // Gather unused inputs into one dummy wire so the linter doesn't warn and
  // reviewers can see they are unused on purpose.
  wire _unused = &{ena, uio_in[7], 1'b0};

endmodule
