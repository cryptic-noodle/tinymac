# SPDX-License-Identifier: Apache-2.0
# cocotb testbench for tt_um_flex_mac
#
# HOW TO RUN
#   RTL :  cd test && make
#   GL  :  cp ../tt_submission/tt_um_flex_mac.v gate_level_netlist.v
#          make clean && PDK_ROOT=<ciel pdk root> make GATES=yes
#
# The same test file runs on the RTL and on the post-route gate-level
# netlist. That is why it only touches top-level pins (ui_in, uio_in, uo_out)
# and never peeks at internal signals: after synthesis, internal names such
# as accum_reg no longer exist.

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge

WIN_1X1, WIN_2X2, WIN_3X3 = 0b00, 0b01, 0b10
WIN_LEN = {WIN_1X1: 1, WIN_2X2: 4, WIN_3X3: 9}   # MACs per window


def ctrl(mode=0, valid=0, latch=0, win=WIN_3X3, rsel=0):
    """Pack the control fields into the uio_in byte (see info.yaml pinout)."""
    return (mode & 1) | (valid & 1) << 1 | (latch & 1) << 2 | (win & 3) << 3 | (rsel & 3) << 5


# ---------------------------------------------------------------------------
# Timing discipline
# ---------------------------------------------------------------------------
# Inputs change on the FALLING edge, so they are stable half a period before
# and after the rising edge where the design samples them. This avoids races
# in RTL sim and, in gate-level sim, mimics a real host that meets setup/hold.
#
# One call to cycle() = one clock. Its outputs are read at the NEXT falling
# edge, after the rising edge has updated the registers.
#
# The design registers every input once (see the RTL "INPUT REGISTERS"
# comment), so each command takes effect one clock later. Because that delay
# is the same for every pin, this testbench never has to correct for it; the
# protocol looks identical to the host.
async def cycle(dut, ui, uio):
    dut.ui_in.value = ui
    dut.uio_in.value = uio
    await FallingEdge(dut.clk)


async def setup(dut):
    # 20 ns = 50 MHz, the clock the design was signed off at.
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    # The reset synchronizer adds 2 cycles and the shadow register needs
    # 2 more to clear (see RTL), so >= 4 cycles is required. 10 matches the TT
    # firmware default and leaves plenty of margin.
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)


async def load_weights(dut, weights, win):
    """Shift in 9 bytes. The first byte ends up in weight_reg[0]."""
    assert len(weights) == 9
    for w in weights:
        await cycle(dut, w, ctrl(mode=1, win=win))


async def stream(dut, acts, win, latch_first=False):
    """Stream one window of activations, one per clock.

    latch_first=True raises latch_and_clear together with the FIRST
    activation. That closes the previous window (its sum goes to the shadow
    register) while this window's first product starts the new sum: back-to-back
    windows with no idle cycle in between.
    """
    for i, a in enumerate(acts):
        await cycle(dut, a, ctrl(valid=1, latch=int(latch_first and i == 0), win=win))


async def read_result(dut, win):
    """Read the 24-bit shadow register as 3 bytes, LSB first."""
    val = 0
    for sel in range(3):
        await cycle(dut, 0, ctrl(win=win, rsel=sel))
        val |= int(dut.uo_out.value) << (8 * sel)
    return val


def dot(weights, acts):
    """Reference model: the plain dot product of the active kernel and window."""
    return sum(w * a for w, a in zip(weights, acts))


# ---------------------------------------------------------------------------
# Test 1: the spec scenario (3x3, weights 1..9, activations all 1 -> 45)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_3x3_dot_product(dut):
    await setup(dut)

    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    acts1 = [1] * 9
    # The second window uses the maximum activation value, so all 3 output
    # bytes are non-zero (11475 = 0x002CD3). It also proves the ring rotated
    # back to its starting position after 9 MACs: if the weights had been
    # corrupted, this sum would be wrong.
    acts2 = [255] * 9

    await load_weights(dut, weights, WIN_3X3)
    await stream(dut, acts1, WIN_3X3)
    await stream(dut, acts2, WIN_3X3, latch_first=True)

    got1 = await read_result(dut, WIN_3X3)
    exp1 = dot(weights, acts1)
    dut._log.info(f"window 1: got {got1}, expected {exp1}")
    assert got1 == exp1

    # Close window 2 with a latch on an idle cycle (valid_in=0)
    await cycle(dut, 0, ctrl(latch=1, win=WIN_3X3))
    got2 = await read_result(dut, WIN_3X3)
    exp2 = dot(weights, acts2)
    dut._log.info(f"window 2: got {got2}, expected {exp2}")
    assert got2 == exp2


# ---------------------------------------------------------------------------
# Test 2: shorter rings (2x2 wraps at w[3], 1x1 never rotates)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_2x2_and_1x1(dut):
    await setup(dut)

    # 2x2: the first 4 bytes are the kernel; the other 5 are padding that must
    # never be used (w[4..8] are frozen in 2x2 mode).
    weights = [10, 20, 30, 40, 0, 0, 0, 0, 0]
    await load_weights(dut, weights, WIN_2X2)
    acts = [3, 5, 7, 11, 2, 4, 6, 8]  # two 2x2 windows, back to back
    await stream(dut, acts[:4], WIN_2X2)
    await stream(dut, acts[4:], WIN_2X2, latch_first=True)
    assert await read_result(dut, WIN_2X2) == dot(weights, acts[:4])
    await cycle(dut, 0, ctrl(latch=1, win=WIN_2X2))
    assert await read_result(dut, WIN_2X2) == dot(weights, acts[4:])

    # 1x1: weight_reg[0] is reused every cycle. 5 x 200 x 250 = 250,000 needs
    # 18 bits, so this also exercises the top output byte.
    await load_weights(dut, [200, 0, 0, 0, 0, 0, 0, 0, 0], WIN_1X1)
    await stream(dut, [250] * 5, WIN_1X1)
    await cycle(dut, 0, ctrl(latch=1, win=WIN_1X1))
    assert await read_result(dut, WIN_1X1) == 200 * 250 * 5


# ---------------------------------------------------------------------------
# Test 3: randomized, with readout overlapping compute
# ---------------------------------------------------------------------------
# Hand-picked tests only check the cases you thought of. Random stimulus
# against a reference model (dot()) finds the cases you didn't. A fixed seed
# keeps failures reproducible.
#
# This test also exercises the shadow register's purpose: the previous
# result is read out (read_select = 0,1,2) WHILE the next window is being
# computed, with valid_in kept high the whole time.
@cocotb.test()
async def test_random_pipelined(dut):
    await setup(dut)
    rng = random.Random(0xC0FFEE)

    for trial in range(12):
        win = rng.choice([WIN_1X1, WIN_2X2, WIN_3X3])
        n = WIN_LEN[win]
        weights = [rng.randrange(256) for _ in range(9)]
        await load_weights(dut, weights, win)

        n_windows = rng.randint(2, 5)
        windows = [[rng.randrange(256) for _ in range(n)] for _ in range(n_windows)]

        prev_expected = None
        for k, acts in enumerate(windows):
            got = 0
            for i, a in enumerate(acts):
                # Latch on the first MAC of every window except the first.
                latch = int(k > 0 and i == 0)
                # The latch from cycle 0 reaches shadow_reg two edges later,
                # so cycles 1..3 of this window can read the previous result.
                rsel = i - 1 if 1 <= i <= 3 else 0
                await cycle(dut, a, ctrl(valid=1, latch=latch, win=win, rsel=rsel))
                if prev_expected is not None and 1 <= i <= 3:
                    got |= int(dut.uo_out.value) << (8 * (i - 1))

            if prev_expected is not None:
                if n >= 4:
                    # Read fully overlapped with compute
                    assert got == prev_expected, (
                        f"trial {trial} win {k - 1}: overlapped read {got} != {prev_expected}")
                else:
                    # A 1x1 window is too short to overlap a 3-cycle read. The
                    # previous result is still in the shadow register (the
                    # next latch hasn't happened yet), so read it now.
                    got = await read_result(dut, win)
                    assert got == prev_expected, (
                        f"trial {trial} win {k - 1}: read {got} != {prev_expected}")
            prev_expected = dot(weights, acts)

        # Close the last window and read it
        await cycle(dut, 0, ctrl(latch=1, win=win))
        got = await read_result(dut, win)
        assert got == prev_expected, f"trial {trial} last window: {got} != {prev_expected}"
        dut._log.info(f"trial {trial}: window={n} MACs x {n_windows} windows OK")


# ---------------------------------------------------------------------------
# Test 4: behaviours promised in docs/PROGRAMMING_GUIDE.md
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_programming_guide_contract(dut):
    await setup(dut)

    # (a) A freshly reset chip reads 0.
    assert await read_result(dut, WIN_3X3) == 0

    # (b) Stalls: valid_in=0 cycles inside a window freeze the ring and the
    #     accumulator, so the host may pause at any point.
    w = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    a = [7, 0, 255, 8, 128, 2, 9, 1, 66]
    await load_weights(dut, w, WIN_3X3)
    for i, x in enumerate(a):
        await cycle(dut, x, ctrl(valid=1, win=WIN_3X3))
        if i % 3 == 1:
            for _ in range(2):  # idle cycles with garbage on the data bus
                await cycle(dut, 0xAA, ctrl(valid=0, win=WIN_3X3))
    await cycle(dut, 0, ctrl(latch=1, win=WIN_3X3))
    assert await read_result(dut, WIN_3X3) == dot(w, a)

    # (c) Multi-channel accumulation: loading new weights does not touch the
    #     accumulator, so sums can span several weight sets before one latch.
    w2 = [255] * 9
    a2 = [255] * 9
    await load_weights(dut, w, WIN_3X3)
    await stream(dut, a, WIN_3X3)
    await load_weights(dut, w2, WIN_3X3)      # accumulator holds during load
    await stream(dut, a2, WIN_3X3)
    await cycle(dut, 0, ctrl(latch=1, win=WIN_3X3))
    assert await read_result(dut, WIN_3X3) == dot(w, a) + dot(w2, a2)

    # (d) Latch overlapping a weight load + readout overlapping the load:
    #     zero-overhead switch to the next weight set.
    await load_weights(dut, w, WIN_3X3)
    await stream(dut, a, WIN_3X3)
    got = 0
    for i, wt in enumerate(w2):
        rsel = i - 1 if 1 <= i <= 3 else 0
        await cycle(dut, wt, ctrl(mode=1, latch=int(i == 0), win=WIN_3X3, rsel=rsel))
        if 1 <= i <= 3:
            got |= int(dut.uo_out.value) << (8 * (i - 1))
    assert got == dot(w, a)
    await stream(dut, a2, WIN_3X3)
    await cycle(dut, 0, ctrl(latch=1, win=WIN_3X3))
    assert await read_result(dut, WIN_3X3) == dot(w2, a2)

    # (e) read_select=3 mirrors byte 2.
    await cycle(dut, 0, ctrl(win=WIN_3X3, rsel=3))
    assert int(dut.uo_out.value) == (dot(w2, a2) >> 16) & 0xFF

    # (f) 24-bit wrap-around: 29 full-scale 3x3 windows without a latch.
    await load_weights(dut, [255] * 9, WIN_3X3)
    for _ in range(29):
        await stream(dut, [255] * 9, WIN_3X3)
    await cycle(dut, 0, ctrl(latch=1, win=WIN_3X3))
    assert await read_result(dut, WIN_3X3) == (29 * 9 * 255 * 255) % (1 << 24)


# ---------------------------------------------------------------------------
# Test 5: the reference host driver (sw/flexmac.py) against the real design
# ---------------------------------------------------------------------------
# The same schedule-builder that firmware uses is run cycle-by-cycle here, so
# the driver and the programming guide are proven against the hardware.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sw"))
import flexmac  # noqa: E402


async def run_schedule(dut, sched, n_windows, read_bytes):
    caps = {}
    for s in sched:
        await cycle(dut, s.ui, s.uio)
        if s.capture is not None:
            caps[s.capture] = int(dut.uo_out.value)
    return flexmac.assemble(caps, n_windows, read_bytes)


@cocotb.test()
async def test_reference_driver(dut):
    await setup(dut)
    rng = random.Random(1234)

    for n in (1, 4, 9):
        for R in (flexmac.bytes_needed(n), 3):
            weights = [rng.randrange(256) for _ in range(9)]
            windows = [[rng.randrange(256) for _ in range(n)] for _ in range(6)]
            sched = flexmac.build_schedule(weights, windows, n, R)
            assert len(sched) == 9 + len(windows) * max(n, R) + 1 + R
            got = await run_schedule(dut, sched, len(windows), R)
            exp = [dot(weights, w) for w in windows]
            assert got == exp, f"n={n} R={R}: {got} != {exp}"
            dut._log.info(f"driver n={n} R={R}: {len(sched)} clocks OK")

    # 10 x 3x3 windows = 103 clocks (the figure quoted in the guide)
    assert len(flexmac.build_schedule([1] * 9, [[1] * 9] * 10, 9)) == 103

    # Signed int8 via offset binary + zero-point correction
    ws = [rng.randrange(-128, 128) for _ in range(9)]
    acts_s = [[rng.randrange(-128, 128) for _ in range(9)] for _ in range(3)]
    wu = [flexmac.to_offset_binary(x) for x in ws]
    au = [[flexmac.to_offset_binary(x) for x in w] for w in acts_s]
    raw = await run_schedule(dut, flexmac.build_schedule(wu, au, 9), 3, 3)
    for r, a_u, a_s in zip(raw, au, acts_s):
        assert flexmac.zero_point_correct(r, wu, a_u, 128, 128) == dot(ws, a_s)
