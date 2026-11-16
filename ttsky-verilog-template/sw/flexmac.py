# SPDX-License-Identifier: Apache-2.0
"""Reference host driver for tt_um_flex_mac (Flexible 8-bit Streaming MAC).

Pure Python with no imports, so it runs on CPython and MicroPython (e.g. the
RP2040 on the Tiny Tapeout demo board).

The driver is split in two:

  1. build_schedule(...) turns a job (weights + activation windows) into a list
     of per-clock Steps: which bytes to put on ui_in / uio_in, and which output
     byte (if any) to capture after that clock. This part is hardware-independent
     and is verified cycle-exactly against the chip in test/test.py.

  2. run(hal, schedule) executes the schedule on real hardware through a tiny
     HAL object you provide (see the class Hal below).

See docs/PROGRAMMING_GUIDE.md for the full protocol.
"""

# ---- uio_in bit fields ------------------------------------------------------
MODE_LOAD = 1 << 0      # uio[0]  1 = load weights, 0 = compute
VALID = 1 << 1          # uio[1]  activation on ui_in is valid this clock
LATCH = 1 << 2          # uio[2]  latch_and_clear
WIN_SHIFT = 3           # uio[4:3] window size
RSEL_SHIFT = 5          # uio[6:5] read select

WIN_CODE = {1: 0b00, 4: 0b01, 9: 0b10}   # MACs per window -> window_size code
RESULT_BITS = 24
RESULT_MASK = (1 << RESULT_BITS) - 1


def ctrl(mode=0, valid=0, latch=0, n=9, rsel=0):
    """Build a uio_in byte. n = MACs per window (1, 4 or 9)."""
    return ((MODE_LOAD if mode else 0) | (VALID if valid else 0) |
            (LATCH if latch else 0) | (WIN_CODE[n] << WIN_SHIFT) |
            ((rsel & 3) << RSEL_SHIFT))


def bytes_needed(n, max_w=255, max_a=255):
    """Smallest number of result bytes (1..3) that can hold an n-MAC sum."""
    worst = n * max_w * max_a
    b = 1
    while b < 3 and worst >= (1 << (8 * b)):
        b += 1
    return b


class Step:
    """One clock of host activity.

    ui, uio : values to drive BEFORE the rising edge.
    capture : None, or (window_index, byte_index): after this clock has
              settled, read uo_out and store it as that byte of that result.
    """
    __slots__ = ("ui", "uio", "capture")

    def __init__(self, ui, uio, capture=None):
        self.ui = ui
        self.uio = uio
        self.capture = capture

    def __repr__(self):
        return "Step(ui=0x%02x, uio=0x%02x, capture=%r)" % (self.ui, self.uio, self.capture)


def build_schedule(weights, windows, n, read_bytes=None):
    """Build the clock-by-clock schedule for one weight set.

    weights    : list of 1..9 ints 0..255; the first n are the kernel. Padded
                 with zeros to 9 (the chip always shifts in 9 bytes).
    windows    : list of windows; each window is a list of exactly n
                 activations (ints 0..255).
    n          : MACs per window: 1 (1x1), 4 (2x2) or 9 (3x3).
    read_bytes : how many result bytes to read per window (1..3). Default:
                 the minimum that can hold the worst case (see bytes_needed).

    Layout (S = max(n, read_bytes) clocks per window):
      [9 load clocks]
      window k: n MAC clocks, then S - n idle clocks.
                On window k's first clock, latch_and_clear closes window k-1.
                On window k's clocks 1..R, read window k-1's bytes 0..R-1.
      [1 latch clock][R read clocks]   <- closes and reads the last window

    Total clocks = 9 + len(windows) * S + 1 + R.
    For n=9, R=3, 10 windows: 9 + 90 + 1 + 3 = 103.
    """
    if n not in WIN_CODE:
        raise ValueError("n must be 1, 4 or 9")
    w = list(weights) + [0] * (9 - len(weights))
    if len(w) != 9 or any(not 0 <= x <= 255 for x in w):
        raise ValueError("need <= 9 weights in 0..255")
    R = read_bytes if read_bytes is not None else bytes_needed(n)
    if not 1 <= R <= 3:
        raise ValueError("read_bytes must be 1..3")
    S = max(n, R)

    steps = []
    # 1. Load. window_size is ignored while loading but kept constant anyway.
    for x in w:
        steps.append(Step(x, ctrl(mode=1, n=n)))

    # 2. Stream windows back to back: MAC slots first, reads overlaid after.
    starts = []
    for k, acts in enumerate(windows):
        if len(acts) != n:
            raise ValueError("window %d has %d activations, need %d" % (k, len(acts), n))
        starts.append(len(steps))
        for i in range(S):
            valid = i < n
            a = acts[i] if valid else 0
            if not 0 <= a <= 255:
                raise ValueError("activation out of range: %r" % a)
            steps.append(Step(a, ctrl(valid=valid, latch=(k > 0 and i == 0), n=n)))

    # 3. Close the last window, then R slots to read it.
    if windows:
        starts.append(len(steps))
        steps.append(Step(0, ctrl(latch=1, n=n)))
        for _ in range(R):
            steps.append(Step(0, ctrl(n=n)))

    # 4. Overlay reads. Window k is latched at slot starts[k+1]; its bytes are
    #    readable at slots starts[k+1]+1 .. +R. Because R <= S, the last read
    #    is at most the NEXT latch slot. That is still safe: a latch reaches the
    #    shadow register one clock after it is driven, so on that slot the
    #    shadow still holds window k.
    for k in range(len(windows)):
        for b in range(R):
            s = steps[starts[k + 1] + 1 + b]
            s.uio |= (b & 3) << RSEL_SHIFT
            s.capture = (k, b)
    return steps


def assemble(captures, n_windows, read_bytes):
    """captures: dict {(window, byte): uo_out value} -> list of ints."""
    out = []
    for k in range(n_windows):
        v = 0
        for b in range(read_bytes):
            v |= (captures[(k, b)] & 0xFF) << (8 * b)
        out.append(v)
    return out


# ---- Signed / zero-point (quantized NN) support ----------------------------
# The hardware multiplies UNSIGNED bytes. For quantized tensors stored as
# uint8 with zero points (TFLite / ONNX style), real value ~ (q - zero_point):
#
#   sum (w - zw)(a - za) = sum(w*a) - za*sum(w) - zw*sum(a) + n*zw*za
#
# The chip computes sum(w*a). The host adds the three correction terms; they
# need only additions, and sum(w) is fixed per weight set.
# Signed int8 data: send x + 128 (two's complement -> offset binary) and use
# zw = za = 128.

def to_offset_binary(x):
    """int8 (-128..127) -> uint8 to send on the bus."""
    if not -128 <= x <= 127:
        raise ValueError("not int8: %r" % x)
    return x + 128


def zero_point_correct(raw_sum, weights_u8, acts_u8, zw, za):
    """Convert the chip's unsigned sum into sum((w-zw)*(a-za))."""
    n = len(acts_u8)
    return raw_sum - za * sum(weights_u8[:n]) - zw * sum(acts_u8) + n * zw * za


# ---- Hardware execution ----------------------------------------------------
class Hal:
    """Interface the host platform implements. Duck-typed; don't subclass.

    step(ui, uio) : drive ui_in=ui and uio_in=uio, then produce ONE rising
                    clock edge, then wait until uo_out has settled (see the
                    Readout timing section of the guide).
    read() -> int : return the current uo_out byte.
    """

    def step(self, ui, uio):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError


def run(hal, schedule, n_windows, read_bytes):
    """Execute a schedule on hardware and return the list of window sums."""
    caps = {}
    for s in schedule:
        hal.step(s.ui, s.uio)
        if s.capture is not None:
            caps[s.capture] = hal.read()
    return assemble(caps, n_windows, read_bytes)
