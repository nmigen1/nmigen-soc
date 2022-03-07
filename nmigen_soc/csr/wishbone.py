from nmigen import *
from nmigen.utils import log2_int

from . import Interface as CSRInterface
from ..wishbone import Interface as WishboneInterface
from ..memory import MemoryMap


__all__ = ["WishboneCSRBridge"]


class WishboneCSRBridge(Elaboratable):
    """Wishbone to CSR bridge.

    A bus bridge for accessing CSR registers from Wishbone. This bridge supports any Wishbone
    data width greater or equal to CSR data width and performs appropriate address translation.

    Latency
    -------

    Reads and writes always take ``self.data_width // csr_bus.data_width + 1`` cycles to complete,
    regardless of the select inputs. Write side effects occur simultaneously with acknowledgement.

    Parameters
    ----------
    csr_bus : :class:`..csr.Interface`
        CSR bus driven by the bridge.
    data_width : int or None
        Wishbone bus data width. If not specified, defaults to ``csr_bus.data_width``.
    name : str
        Window name. Optional.

    Attributes
    ----------
    wb_bus : :class:`..wishbone.Interface`
        Wishbone bus provided by the bridge.
    """
    def __init__(self, csr_bus, *, data_width=None, name=None):
        if not isinstance(csr_bus, CSRInterface):
            raise ValueError("CSR bus must be an instance "
                             "of CSRInterface, not {!r}"
                             .format(csr_bus))
        if csr_bus.data_width not in (8, 16, 32, 64):
            raise ValueError("CSR bus data width must be "
                             "one of 8, 16, 32, 64, not {!r}"
                             .format(csr_bus.data_width))
        if data_width is None:
            data_width = csr_bus.data_width

        self.csr_bus = csr_bus
        self.wb_bus  = WishboneInterface(
            addr_width=max(0, csr_bus.addr_width - 
                              log2_int(data_width // csr_bus.data_width)),
            data_width=data_width,
            granularity=csr_bus.data_width,
            name="wb")

        wb_map = MemoryMap(addr_width=csr_bus.addr_width,
                           data_width=csr_bus.data_width,
                           name=name)
        # Since granularity of the Wishbone interface matches the data width of the CSR bus,
        # no width conversion is performed, even if the Wishbone data width is greater.
        wb_map.add_window(self.csr_bus.memory_map)
        self.wb_bus.memory_map = wb_map

    def elaborate(self, platform):
        csr_bus = self.csr_bus
        wb_bus  = self.wb_bus

        m = Module()
        comb, sync = m.d.comb, m.d.sync
        slen = len(wb_bus.sel)

        # cycle through at the granularity of the Wisbone Bus, updating the CSR
        # note: cycle is up to 1 more than the wb.sel granularity.
        # use cycle to construct the CSR bus address.
        cycle = Signal(range(slen + 1))
        comb += csr_bus.addr.eq(Cat(cycle[:log2_int(slen)], wb_bus.adr))

        # when cyc/stb are first raised, cycle starts progressing.
        # however, on WB4 pipeline-mode requests we cannot rely on stb
        # being held hi indefinitely (like it is in WB3 "classic").
        # use cycle being non-zero to continue sending CSR requests
        is_in_progress = Signal()
        comb += is_in_progress.eq(cycle != 0)

        # WB 4 pipeline mode w/stall
        if hasattr(wb_bus, "stall"):
            comb += wb_bus.stall.eq(~ack)

        with m.If(wb_bus.cyc & (wb_bus.stb | is_in_progress)):
            # cover up to len(sel)+1 cases, last one does the "ack"
            with m.Switch(cycle):
                def segment(index):
                    return slice(wb_bus.granularity * index,
                                 wb_bus.granularity * (index + 1))

                # cycle between 0..len(wb.sel)-1
                for index, sel_index in enumerate(wb_bus.sel):
                    prev_seg = segment(index - 1)
                    seg = segment(index)
                    with m.Case(index):
                        if index > 0:
                            # CSR reads registered: need to re-register them.
                            sync += wb_bus.dat_r[prev_seg].eq(csr_bus.r_data)
                        comb += csr_bus.r_stb.eq(sel_index & ~wb_bus.we)
                        comb += csr_bus.w_data.eq(wb_bus.dat_w[seg])
                        comb += csr_bus.w_stb.eq(sel_index & wb_bus.we)
                        sync += cycle.eq(index + 1)

                # cycle is len(wb.sel). use this to send an ack
                with m.Default():
                    sync += wb_bus.dat_r[seg].eq(csr_bus.r_data)
                    sync += wb_bus.ack.eq(1)

        # one clock later, clear ack and reset cycle back to zero
        with m.If(wb_bus.ack):
            sync += cycle.eq(0)
            sync += wb_bus.ack.eq(0)

        return m
