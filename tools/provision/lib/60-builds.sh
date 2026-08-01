# Native builds: tarball dependencies, then the dsa110-shell C/CUDA
# stack.
#
# This whole stage is missing from the host provisioning today. The
# legacy install_repos_c does it, but that script was written for the
# LXC "corr" container profile and is not referenced by the bare-metal
# curtin preseed -- so a host rebuilt from the current preseed gets the
# repos cloned and nothing compiled.
#
# dsa110-xengine matters beyond the legacy search path: the beamformer
# weight distribution writes antennas.out into
# dsa110-xengine/utils/ on every corr node, so the tree must exist.

stage_builds() {
    # --- tarball dependencies -------------------------------------------
    # cfitsio, hwloc, sofa, fftw(float), thrust, dedisp, libljm. These are
    # prerequisites for psrdada/xGPU/heimdall below.
    if [[ -f /usr/local/lib/libfftw3f.so || -f /usr/local/lib/libfftw3f.a ]]; then
        ok "tarball deps present (fftw3f found)"
    else
        run_sh "run install_tarballs" \
            "cd /run && curl -fsSLO '$WEB_MAAS/config/lxd/install_tarballs' \
             && chmod 755 install_tarballs && ./install_tarballs"
    fi

    local R="$PROJ_DIR/dsa110-shell"

    # --- psrdada ---------------------------------------------------------
    if have dada_db; then
        ok "psrdada installed (dada_db on PATH)"
    else
        _build "psrdada" "$R/dsa110-psrdada" \
            "autoreconf -i && ./configure --prefix=/usr/local \
                --with-fftw3-dir=/usr/local --with-sofa-dir=/usr/local \
                --with-sofa-include-dir=/usr/local/include \
                --with-xgpu-dir=/usr/local --with-hwloc-dir=/usr/local \
                CFLAGS=-fPIC && make -j\$(nproc) && make install"
    fi

    # --- xGPU ------------------------------------------------------------
    if [[ -f /usr/local/lib/libxgpu.so ]]; then
        ok "xGPU installed"
    else
        _build "xGPU" "$R/dsa110-xGPU/src" "make && make install"
    fi

    # --- sigproc ----------------------------------------------------------
    if [[ -f /usr/local/lib/libsigproc.a ]]; then
        ok "sigproc installed"
    else
        _build "sigproc" "$R/dsa110-sigproc" \
            "./bootstrap && ./configure --prefix=/usr/local \
             && make -j\$(nproc) && make install \
             && cp src/libsigproc.a /usr/local/lib/ \
             && mkdir -p /usr/local/include/src && cp src/*.h /usr/local/include/src/"
    fi

    # --- xengine ----------------------------------------------------------
    # Required: bfweights distribution targets
    # dsa110-xengine/utils/antennas.out on every corr node.
    if [[ -x "$R/dsa110-xengine/src/dsaX_capture" ]]; then
        ok "xengine built"
    else
        _build "xengine" "$R/dsa110-xengine/src" "make clean; make"
    fi
    run_sh "ensure xengine utils/ exists (bfweights target)" \
        "mkdir -p '$R/dsa110-xengine/utils' && chown -R ubuntu:ubuntu '$R/dsa110-xengine'"

    # --- mbheimdall --------------------------------------------------------
    if [[ -x "$R/dsa110-mbheimdall/bin/heimdall" ]]; then
        ok "mbheimdall built"
    else
        # configure is run twice on purpose: the first pass generates the
        # makefiles the `make clean` needs, matching install_repos_c.
        _build "mbheimdall" "$R/dsa110-mbheimdall" \
            "./configure --prefix='$R/dsa110-mbheimdall' \
                 --with-thrust-dir=/usr/local/thrust-1.7 --with-cuda-dir=/usr/local/cuda \
             && make clean \
             && ./configure --prefix='$R/dsa110-mbheimdall' \
                 --with-thrust-dir=/usr/local/thrust-1.7 --with-cuda-dir=/usr/local/cuda \
             && make && make install"
    fi

    run "refresh linker cache" -- ldconfig
    run "chown $PROJ_DIR to ubuntu" -- chown -R ubuntu:ubuntu "$PROJ_DIR"
}

# _build <name> <dir> <shell-snippet>
_build() {
    local name="$1" dir="$2" cmd="$3"
    if [[ ! -d "$dir" ]]; then
        warn "$name: $dir missing — did the repos stage run?"
        return 0
    fi
    run_sh "build $name in $dir" "cd '$dir' && $cmd"
}
