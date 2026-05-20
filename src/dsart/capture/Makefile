# dsart_capture_manythread Makefile.
#
# Produces a single executable inside the package directory:
#
#   src/dsart/capture/dsart_capture_manythread
#
# The dsart-rt orchestrator references this via the YAML routine
# `cap_a_real` / `cap_b_real` in configs/dsart_pipeline_rt.yaml.
# setup.py's custom build_ext step invokes `make` after the C
# extensions (_recv_ring, _recv_epoll) are built, so a single
# `python setup.py build_ext --inplace` builds both the .so files
# and this binary in one go (this is the path _sync_fleet.sh uses
# to rebuild on every fleet node).
#
# Build dependencies:
#   * libpsrdada (static .a in /usr/local/lib on n01..n22; from the
#     dsa110-psrdada vendor tree)
#   * libgfortran (transitive via libpsrdada -> sigproc, kept for
#     legacy compatibility even though we don't use sigproc here)
#   * libm, libpthread (standard glibc)
#
# Compiler choice: gcc with -O2. -O3 is not worth the marginal gain
# at our packet rate (~17 k pps / port) and risks reordering atomic
# stores around the recvmmsg loop in ways that take longer to audit.

CC := gcc

PSRDADA_INC := /usr/local/include
PSRDADA_LIB := /usr/local/lib

CFLAGS := -g -O2 -Wall -Wextra -Wno-unused-parameter \
          -pthread -march=native \
          -D_GNU_SOURCE \
          -I$(PSRDADA_INC) \
          -I$(PSRDADA_INC)/src \
          -I.

LDFLAGS := -L$(PSRDADA_LIB)
LIBS    := -lpsrdada -lm -lpthread -lrt

DEPS_H := dsart_capture_def.h dsart_capture_manythread.h dsart_capture_mon.h

OBJS := dsart_capture_manythread.o dsart_capture_mon.o

BIN := dsart_capture_manythread

.PHONY: all clean check

all: $(BIN)

%.o: %.c $(DEPS_H)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BIN): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(OBJS) $(BIN)

# Smoke check: the binary should at least run --help without segfaulting.
# This is the same check setup.py's build wrapper uses to confirm
# the link step succeeded; it does not touch /dev/shm or the network.
check: $(BIN)
	./$(BIN) -h
