import unittest
from nmigen import *
from nmigen.hdl.rec import Layout
from nmigen.back.pysim import *

from ..csr.bus import *


class CSRElementTestCase(unittest.TestCase):
    def test_1_ro(self):
        elem = CSRElement(1, "r")
        self.assertEqual(elem.width, 1)
        self.assertEqual(elem.access, "r")
        self.assertEqual(elem.layout, Layout.cast([
            ("r_data", 1),
            ("r_stb", 1),
        ]))

    def test_8_rw(self):
        elem = CSRElement(8, access="rw")
        self.assertEqual(elem.width, 8)
        self.assertEqual(elem.access, "rw")
        self.assertEqual(elem.layout, Layout.cast([
            ("r_data", 8),
            ("r_stb", 1),
            ("w_data", 8),
            ("w_stb", 1),
        ]))

    def test_10_wo(self):
        elem = CSRElement(10, "w")
        self.assertEqual(elem.width, 10)
        self.assertEqual(elem.access, "w")
        self.assertEqual(elem.layout, Layout.cast([
            ("w_data", 10),
            ("w_stb", 1),
        ]))

    def test_0_rw(self): # degenerate but legal case
        elem = CSRElement(0, access="rw")
        self.assertEqual(elem.width, 0)
        self.assertEqual(elem.access, "rw")
        self.assertEqual(elem.layout, Layout.cast([
            ("r_data", 0),
            ("r_stb", 1),
            ("w_data", 0),
            ("w_stb", 1),
        ]))

    def test_width_wrong(self):
        with self.assertRaisesRegex(ValueError,
                r"Width must be a non-negative integer, not -1"):
            CSRElement(-1, "rw")

    def test_access_wrong(self):
        with self.assertRaisesRegex(ValueError,
                r"Access mode must be one of \"r\", \"w\", or \"rw\", not 'wo'"):
            CSRElement(1, "wo")


class CSRMultiplexerTestCase(unittest.TestCase):
    def setUp(self):
        self.dut = CSRMultiplexer(addr_width=16, data_width=8)

    def test_addr_width_wrong(self):
        with self.assertRaisesRegex(ValueError,
                r"Address width must be a positive integer, not -1"):
            CSRMultiplexer(addr_width=-1, data_width=8)

    def test_data_width_wrong(self):
        with self.assertRaisesRegex(ValueError,
                r"Data width must be a positive integer, not -1"):
            CSRMultiplexer(addr_width=16, data_width=-1)

    def test_alignment_wrong(self):
        with self.assertRaisesRegex(ValueError,
                r"Alignment must be a non-negative integer, not -1"):
            CSRMultiplexer(addr_width=16, data_width=8, alignment=-1)

    def test_attrs(self):
        self.assertEqual(self.dut.addr_width, 16)
        self.assertEqual(self.dut.data_width, 8)
        self.assertEqual(self.dut.alignment, 0)

    def test_add_4b(self):
        self.assertEqual(self.dut.add(CSRElement(4, "rw")),
                         (0, 1))

    def test_add_8b(self):
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (0, 1))

    def test_add_12b(self):
        self.assertEqual(self.dut.add(CSRElement(12, "rw")),
                         (0, 2))

    def test_add_16b(self):
        self.assertEqual(self.dut.add(CSRElement(16, "rw")),
                         (0, 2))

    def test_add_two(self):
        self.assertEqual(self.dut.add(CSRElement(16, "rw")),
                         (0, 2))
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (2, 1))

    def test_add_wrong(self):
        with self.assertRaisesRegex(ValueError,
                r"Width must be a non-negative integer, not -1"):
            CSRElement(-1, "rw")

    def test_align_to(self):
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (0, 1))
        self.assertEqual(self.dut.align_to(2), 4)
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (4, 1))

    def test_sim(self):
        elem_4_r = CSRElement(4, "r")
        self.dut.add(elem_4_r)
        elem_8_w = CSRElement(8, "w")
        self.dut.add(elem_8_w)
        elem_16_rw = CSRElement(16, "rw")
        self.dut.add(elem_16_rw)

        def sim_test():
            yield elem_4_r.r_data.eq(0xa)
            yield elem_16_rw.r_data.eq(0x5aa5)

            yield self.dut.addr.eq(0)
            yield self.dut.r_stb.eq(1)
            yield
            yield self.dut.r_stb.eq(0)
            self.assertEqual((yield elem_4_r.r_stb), 1)
            self.assertEqual((yield elem_16_rw.r_stb), 0)
            yield
            self.assertEqual((yield self.dut.r_data), 0xa)

            yield self.dut.addr.eq(2)
            yield self.dut.r_stb.eq(1)
            yield
            yield self.dut.r_stb.eq(0)
            self.assertEqual((yield elem_4_r.r_stb), 0)
            self.assertEqual((yield elem_16_rw.r_stb), 1)
            yield
            yield self.dut.addr.eq(3) # pipeline a read
            self.assertEqual((yield self.dut.r_data), 0xa5)

            yield self.dut.r_stb.eq(1)
            yield
            yield self.dut.r_stb.eq(0)
            self.assertEqual((yield elem_4_r.r_stb), 0)
            self.assertEqual((yield elem_16_rw.r_stb), 0)
            yield
            self.assertEqual((yield self.dut.r_data), 0x5a)

            yield self.dut.addr.eq(1)
            yield self.dut.w_data.eq(0x3d)
            yield self.dut.w_stb.eq(1)
            yield
            yield self.dut.w_stb.eq(0)
            yield
            self.assertEqual((yield elem_8_w.w_stb), 1)
            self.assertEqual((yield elem_8_w.w_data), 0x3d)
            self.assertEqual((yield elem_16_rw.w_stb), 0)

            yield self.dut.addr.eq(2)
            yield self.dut.w_data.eq(0x55)
            yield self.dut.w_stb.eq(1)
            yield
            self.assertEqual((yield elem_8_w.w_stb), 0)
            self.assertEqual((yield elem_16_rw.w_stb), 0)
            yield self.dut.addr.eq(3) # pipeline a write
            yield self.dut.w_data.eq(0xaa)
            yield
            self.assertEqual((yield elem_8_w.w_stb), 0)
            self.assertEqual((yield elem_16_rw.w_stb), 0)
            yield self.dut.w_stb.eq(0)
            yield
            self.assertEqual((yield elem_8_w.w_stb), 0)
            self.assertEqual((yield elem_16_rw.w_stb), 1)
            self.assertEqual((yield elem_16_rw.w_data), 0xaa55)

        with Simulator(self.dut, vcd_file=open("test.vcd", "w")) as sim:
            sim.add_clock(1e-6)
            sim.add_sync_process(sim_test())
            sim.run()


class CSRAlignedMultiplexerTestCase(unittest.TestCase):
    def setUp(self):
        self.dut = CSRMultiplexer(addr_width=16, data_width=8, alignment=2)

    def test_attrs(self):
        self.assertEqual(self.dut.alignment, 2)

    def test_add_two(self):
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (0, 4))
        self.assertEqual(self.dut.add(CSRElement(16, "rw")),
                         (4, 4))

    def test_over_align_to(self):
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (0, 4))
        self.assertEqual(self.dut.align_to(3), 8)
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (8, 4))

    def test_under_align_to(self):
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (0, 4))
        self.assertEqual(self.dut.align_to(1), 4)
        self.assertEqual(self.dut.add(CSRElement(8, "rw")),
                         (4, 4))

    def test_sim(self):
        elem_20_rw = CSRElement(20, "rw")
        self.dut.add(elem_20_rw)

        def sim_test():
            yield self.dut.w_stb.eq(1)
            yield self.dut.addr.eq(0)
            yield self.dut.w_data.eq(0x55)
            yield
            self.assertEqual((yield elem_20_rw.w_stb), 0)
            yield self.dut.addr.eq(1)
            yield self.dut.w_data.eq(0xaa)
            yield
            self.assertEqual((yield elem_20_rw.w_stb), 0)
            yield self.dut.addr.eq(2)
            yield self.dut.w_data.eq(0x33)
            yield
            self.assertEqual((yield elem_20_rw.w_stb), 0)
            yield self.dut.addr.eq(3)
            yield self.dut.w_data.eq(0xdd)
            yield
            self.assertEqual((yield elem_20_rw.w_stb), 0)
            yield self.dut.w_stb.eq(0)
            yield
            self.assertEqual((yield elem_20_rw.w_stb), 1)
            self.assertEqual((yield elem_20_rw.w_data), 0x3aa55)

        with Simulator(self.dut, vcd_file=open("test.vcd", "w")) as sim:
            sim.add_clock(1e-6)
            sim.add_sync_process(sim_test())
            sim.run()
