# tt_um_flex_mac: Programming Guide

For firmware and software engineers writing a host driver for the Flexible 8-bit Streaming MAC on Tiny Tapeout. You don't need to read the Verilog.

Every behaviour in this guide is checked by a test in [`test/test.py`](../test/test.py). The tests run against both the RTL and the post-layout gate-level netlist. If a behaviour is not listed here, treat it as unsupported.

A reference driver is in [`sw/flexmac.py`](../sw/flexmac.py). It is plain Python, runs on MicroPython or CPython, and has no dependencies. Start from it rather than from scratch.

---

## 1. What the chip does

The chip computes dot products of unsigned 8-bit numbers:

```
result = w[0]*a[0] + w[1]*a[1] + ... + w[N-1]*a[N-1]        (24-bit, unsigned)
```

- You load a **kernel** of up to 9 weights once. It stays on the chip.
- You stream **activations** one byte per clock. The chip does one multiply-accumulate (MAC) per clock.
- A **window** is N consecutive activations: N = 1 (1x1), 4 (2x2) or 9 (3x3). Each window produces one result.
- Results are **double-buffered**. You read window *k* while window *k+1* is computing, so in steady state no clock is wasted.

Typical use: 2D convolution or dense layers of a small quantized neural network, where the same kernel is applied to many windows.

---

## 2. Pins

| Pin | Direction | Name | Meaning |
|---|---|---|---|
| `ui_in[7:0]` | in | DATA | Weight (load mode) or activation (compute mode) |
| `uo_out[7:0]` | out | RESULT | One byte of the latched result, chosen by READ_SEL |
| `uio_in[0]` | in | MODE | 1 = load weights, 0 = compute |
| `uio_in[1]` | in | VALID | 1 = DATA is an activation to MAC this clock (compute mode only) |
| `uio_in[2]` | in | LATCH | 1 = latch the current sum and start a new one |
| `uio_in[4:3]` | in | WIN | Window size: `00` = 1x1, `01` = 2x2, `10` = 3x3 (`11` acts as 3x3) |
| `uio_in[6:5]` | in | READ_SEL | Result byte on `uo_out`: `0` = [7:0], `1` = [15:8], `2` = [23:16], `3` = same as `2` |
| `uio_in[7]` | in | — | Unused, drive 0 |
| `clk` | in | — | Clock, rising edge active, max 50 MHz |
| `rst_n` | in | — | Active-low reset |

All 8 `uio` pins are **inputs** to the chip (it never drives them). On the host side, configure them as **outputs**.

### Command word

Each clock, the host drives one 16-bit command: `ui_in` (data) plus `uio_in` (control).

```python
MODE_LOAD = 1 << 0
VALID     = 1 << 1
LATCH     = 1 << 2
WIN_1X1, WIN_2X2, WIN_3X3 = 0 << 3, 1 << 3, 2 << 3
def RSEL(b): return (b & 3) << 5
```

```c
#define FM_MODE_LOAD  (1u << 0)
#define FM_VALID      (1u << 1)
#define FM_LATCH      (1u << 2)
#define FM_WIN_1X1    (0u << 3)
#define FM_WIN_2X2    (1u << 3)
#define FM_WIN_3X3    (2u << 3)
#define FM_RSEL(b)    (((b) & 3u) << 5)
```

The control bits are independent and can be combined in one clock. For example, `VALID | LATCH | RSEL(1)` does a MAC, closes the previous window and selects byte 1, all at once. This is what makes zero-overhead streaming possible.

---

## 3. Clocking model

### 3.1 Slots

Think of the host as running a sequence of **slots**. One slot is:

1. Drive `ui_in` and `uio_in`.
2. Produce **one** rising edge on `clk`.
3. Wait for outputs to settle (§3.3), then optionally read `uo_out`.

All tables and formulas in this guide count slots. One slot = one clock.

### 3.2 Rules

| Rule | Detail |
|---|---|
| Sampling | Inputs are sampled on the rising edge. Change them while `clk` is low: half a period before the edge is the safe choice. |
| Max frequency | 50 MHz. Timing is signed off at every corner, with +2.56 ns margin at the worst one (slow silicon, 100 °C, 1.60 V). |
| Min frequency | None. The design is fully static. You may stop the clock at any time, single-step it, or stretch individual cycles. |
| Latency | Every input is registered once. You don't need to compensate for this; §3.1 already accounts for it. |
| `ena` | Ignored. |

### 3.3 Reading `uo_out` on real silicon

On Tiny Tapeout, every pin passes through a shared multiplexer. TT [measured](https://tinytapeout.com/specs/gpio) up to about **20 ns** round-trip latency and specifies a **33 MHz maximum output toggle rate**. In practice, `uo_out` may need up to about 20 ns after the rising edge to show the new byte.

- **≤ 25 MHz:** reading at the next falling or rising edge is comfortable.
- **Faster:** run MAC slots at full speed, but **stretch the clock on read slots**: keep `clk` low longer before sampling. This works because the design is static (§3.2).

These figures come from TT's measurements of a different shuttle. Characterise your own board.

---

## 4. Operations

### 4.1 Reset

```
rst_n = 0 for >= 4 clocks   (10 recommended; the TT firmware default)
rst_n = 1, then 2 idle clocks (uio_in = 0) before the first command
```

| After reset | Value |
|---|---|
| Latched result (`uo_out` for any READ_SEL) | 0 |
| Running sum | 0 |
| Weights | **Undefined.** Always load weights before computing. |

### 4.2 Load weights: 9 slots

```
for i in 0..8:   ui_in = weight[i];   uio_in = MODE_LOAD | WIN
```

- Always shift in exactly **9** bytes, even for 1x1 and 2x2.
- The **first** byte loaded is used first. For a kernel of N weights, send the N weights followed by 9-N padding bytes (any value).
- Loading does **not** change the running sum or the latched result. This lets you accumulate across several weight sets (§6.2) and read results while loading (§6.3).
- A full load always restores a known state, whatever happened before. It's the recovery step for every error case in §7.

### 4.3 Compute: 1 slot per MAC

```
ui_in = activation;   uio_in = VALID | WIN
```

- Each VALID slot adds `activation * w[current]` to the running sum and advances to the next weight in the kernel. After N MACs it wraps to `w[0]`.
- Slots with VALID = 0 (and MODE = 0) are **pauses**. Nothing changes, and DATA is ignored. You may pause anywhere, including in the middle of a window.

### 4.4 Latch: 0 extra slots when overlapped

Set `LATCH` on the slot **after** the last MAC of a window:

- The finished sum moves to the result register (readable via `uo_out`).
- The running sum restarts at 0. If the same slot also has VALID = 1, it restarts at that slot's product. That's how back-to-back windows work: latch window *k* on window *k+1*'s first MAC.
- LATCH works in load mode too.

### 4.5 Read: 1 slot per byte

```
uio_in |= RSEL(b)          # b = 0, 1, 2
read uo_out after the slot  -> byte b of the latched result
```

- The latched result can be read from the slot **after** the latch slot.
- It stays readable until the latch after next takes effect, which is one slot after that LATCH is driven.
- READ_SEL can be combined with any other bits, so reads ride along with MAC or load slots.

### 4.6 How many bytes to read

| Window | Max result (all 255s) | Bytes needed |
|---|---|---|
| 1x1 | 65,025 | 2 |
| 2x2 | 260,100 | 3 |
| 3x3 | 585,225 | 3 |

If your data has a smaller range, you can read fewer bytes. `flexmac.bytes_needed(n, max_w, max_a)` computes this.

---

## 5. The streaming schedule

### 5.1 Back-to-back windows (3x3 shown)

```
slot        | ... | W1 mac 8 | W2 mac 0 | W2 mac 1 | W2 mac 2 | W2 mac 3 | ... | W2 mac 8 | W3 mac 0 |
ui_in       |     | a1[8]    | a2[0]    | a2[1]    | a2[2]    | a2[3]    |     | a2[8]    | a3[0]    |
VALID       |     | 1        | 1        | 1        | 1        | 1        |     | 1        | 1        |
LATCH       |     | 0        | 1        | 0        | 0        | 0        |     | 0        | 1        |
READ_SEL    |     | -        | -        | 0        | 1        | 2        |     | -        | -        |
read uo_out |     |          |          | R1 b0    | R1 b1    | R1 b2    |     |          |          |
```

Window *k* is latched on window *k+1*'s first slot, and its R bytes are read on the next R slots. The last read may land on the following latch slot, which is safe (§4.5).

**Constraint:** each window must last at least R slots, where R is the number of bytes you read. If N < R (1x1 with 2 or 3 bytes), pad each window with VALID = 0 slots. A window then lasts `S = max(N, R)` slots.

### 5.2 A complete job

```
[ 9 load slots ][ W windows x S slots ][ 1 latch slot ][ R read slots ]

total slots = 9 + W*S + 1 + R          S = max(N, R)
```

| Job | Slots | At 50 MHz |
|---|---|---|
| 10 x 3x3 windows, 3 bytes each | 9 + 90 + 1 + 3 = **103** | 2.06 µs |
| 10 x 2x2 windows, 3 bytes each | 9 + 40 + 1 + 3 = 53 | 1.06 µs |
| 10 x 1x1 windows, 2 bytes each | 9 + 20 + 1 + 2 = 32 | 0.64 µs |

The final `1 + R` slots disappear if another job follows (§6.3).

### 5.3 Steady-state throughput (chip-limited, 50 MHz)

| Window | Slots/window | Windows/s | MAC/s |
|---|---|---|---|
| 3x3 | 9 | 5.56 M | 50 M |
| 2x2 | 4 | 12.5 M | 50 M |
| 1x1 (2-byte read) | 2 | 25 M | 25 M |

These numbers assume the host can supply one command per clock at 50 MHz. Most hosts can't; see §8.

---

## 6. Patterns

### 6.1 Many windows, same kernel

Load once, then stream windows forever. After every N MACs the kernel is back at `w[0]`, so it never needs reloading.

### 6.2 Summing across weight sets (multi-channel convolution)

Loading does not touch the running sum, so you can do:

```
load kernel (channel 0); stream window (channel 0)   # no latch
load kernel (channel 1); stream window (channel 1)   # no latch
...
latch; read                                          # sum over all channels
```

You can sum up to 28 full-scale 3x3 windows before the 24-bit result wraps (`2^24 / 585,225 ≈ 28.7`). Overflow wraps modulo 2^24 with no flag.

### 6.3 Switching kernels with no overhead

Put the final LATCH on the first load slot of the next kernel, and do the R reads on the following load slots. A job then costs `9 + W*S` slots.

### 6.4 Signed int8 and quantized models

The hardware is **unsigned**. For signed int8, or uint8 with zero points (TFLite/ONNX quantization):

1. Send `x + 128` for signed int8 values (`flexmac.to_offset_binary`).
2. Correct on the host:

```
sum((w - zw)(a - za)) = raw - za*sum(w) - zw*sum(a) + N*zw*za
```

`sum(w)` is constant per kernel, and `sum(a)` is a running add over the activations you already have. Both are cheap. For signed int8, `zw = za = 128` (`flexmac.zero_point_correct`). The signed range of one 3x3 window is ±147,456.

---

## 7. Rules and error recovery

| # | Rule | If broken | Recovery |
|---|---|---|---|
| 1 | Load weights after every reset | Results are garbage | Load weights |
| 2 | Each window has **exactly N** VALID slots between latches | That result is a partial or over-long sum, and the kernel is out of step for later windows | Load weights |
| 3 | Keep WIN constant from a load until the last latch of that job | The kernel position becomes undefined | Load weights |
| 4 | Each window lasts ≥ R slots when reading overlapped (§5.1) | A read returns a byte of the next result | Use `S = max(N, R)` |
| 5 | Latch before reading | You read the previous (stale) result | — |
| 6 | Keep running sums under 2^24 | Silent wrap-around | Latch more often |

---

## 8. Host considerations (RP2040 / Tiny Tapeout demo board)

The chip runs at 50 MHz. Whether *your host* can feed it at that rate is a separate question.

| Host approach | Achievable command rate (rough) | Notes |
|---|---|---|
| MicroPython bit-banging (TT SDK) | kHz range | Easiest. Fine for bring-up and testing. |
| C bit-banging on the RP2040 | ~MHz range | Several GPIO writes plus a clock toggle per slot |
| RP2040 PIO + DMA | Tens of MHz, in principle | PIO runs 1 instruction per system clock and can produce the clock as side-set, so data and clock stay aligned. It needs the data pins to be contiguous GPIOs; check your board's pin map. |
| FPGA or another chip | Up to 50 MHz | Needs a proper synchronous interface |

The RP2040 can easily *generate* a 50 MHz clock; TT supports 1 Hz to 66.5 MHz from the demo board. What's hard is putting a new 16-bit command on the pins every 20 ns. The result is still correct at any rate: the slot counts in §5 stay the same, and only the wall-clock time changes. Measure on your setup before relying on any of the rough figures above.

---

## 9. Reference driver

[`sw/flexmac.py`](../sw/flexmac.py) separates *what to drive* from *how to drive it*:

```python
import flexmac

sched = flexmac.build_schedule(weights, windows, n=9)   # list of Step(ui, uio, capture)
results = flexmac.run(hal, sched, n_windows=len(windows),
                      read_bytes=flexmac.bytes_needed(9))
```

`build_schedule` produces the optimal schedule from §5 and is tested cycle-by-cycle against the chip (`test_reference_driver`). Your platform code only provides a `hal` object with two methods:

```python
class MyHal:
    def step(self, ui, uio):   # drive pins, one rising edge, wait for settle (§3.3)
        ...
    def read(self):            # return uo_out as an int
        ...
```

### Golden test vector

Use this to bring up a new host:

| Input | Value |
|---|---|
| Window | 3x3 |
| Weights | 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Window 1 activations | 1 x 9 → result **45** = bytes `2D 00 00` |
| Window 2 activations | 255 x 9 → result **11475** = bytes `D3 2C 00` |
