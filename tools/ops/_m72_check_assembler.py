"""Quick smoke: is rx_ring_assemble_validity_block exported by recv_ring.so?"""
import sys

sys.path.insert(0, "src")

from dsart.transport import recv_ring  # noqa: E402

lib = recv_ring._get_lib()
print("lib path:", lib._name)
print(
    "has assembler symbol:",
    hasattr(lib, "rx_ring_assemble_validity_block"),
)
fn = getattr(lib, "rx_ring_assemble_validity_block", None)
print("fn:", fn)
print("argtypes:", getattr(fn, "argtypes", None))
print(
    "_recv_ring file ts:",
    __import__("os").path.getmtime(lib._name),
)
