# CUDA + NVIDIA driver, pinned.
#
# The legacy cloud-init ran `apt-get -y install cuda`, which resolves to
# whatever is newest in the NVIDIA repo. On a rebuild today that is many
# major versions past the CUDA 11.1 the fleet runs, and the compiled
# artefacts would not match: dsart's CUDA kernels, dsa110-xGPU and
# dsa110-mbheimdall are all built against /usr/local/cuda. A node that
# came up with CUDA 12.x would build cleanly and then behave differently
# from its 15 siblings -- the worst kind of failure.
#
# So: install the versioned metapackages, never the floating one.

stage_cuda() {
    local want_cuda="$CUDA_VERSION" have_cuda=""
    if have nvcc || [[ -x /usr/local/cuda/bin/nvcc ]]; then
        have_cuda="$(/usr/local/cuda/bin/nvcc --version 2>/dev/null \
                     | grep -oP 'release \K[0-9.]+' || true)"
    fi
    local have_drv=""
    have nvidia-smi && have_drv="$(nvidia-smi --query-gpu=driver_version \
                                   --format=csv,noheader 2>/dev/null | head -1)"

    # Report the two independently: the toolkit and the driver are
    # separate packages and one can match while the other does not.
    local cuda_ok="no" drv_ok="no"
    if [[ "$have_cuda" == "$want_cuda" ]]; then
        ok "cuda $have_cuda (pinned)"; cuda_ok="yes"
    elif [[ -n "$have_cuda" ]]; then
        warn "cuda $have_cuda installed, want $want_cuda"
    else
        log "  [ -- ] cuda not installed"
    fi
    if [[ "$have_drv" == "$NVIDIA_DRIVER_VERSION" ]]; then
        ok "driver $have_drv (pinned)"; drv_ok="yes"
    elif [[ -n "$have_drv" ]]; then
        warn "driver $have_drv installed, want $NVIDIA_DRIVER_VERSION"
    else
        log "  [ -- ] nvidia driver not installed"
    fi

    if [[ "$cuda_ok" == "yes" && "$drv_ok" == "yes" ]]; then
        _cuda_symlink_check
        return 0
    fi

    # The NVIDIA repo pin file keeps the CUDA repo from outranking the
    # distro one for non-CUDA packages.
    run_sh "fetch NVIDIA apt pin" \
        "cd /tmp && curl -fsSLO $WEB_MAAS/config/tarballs/cuda-ubuntu1804.pin \
         && mv -f /tmp/cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600"
    run_sh "add NVIDIA repo key" \
        "apt-key adv --fetch-keys $WEB_MAAS/config/tarballs/7fa2af80.pub"
    run_sh "add NVIDIA CUDA repo" \
        "add-apt-repository -y 'deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /'"
    run "apt update" -- apt-get update -qq

    # Versioned packages, not bare "cuda"/"cuda-drivers".
    run "install $CUDA_APT_PKG + $NVIDIA_APT_PKG (pinned)" \
        -- apt-get -y install "$CUDA_APT_PKG" "$NVIDIA_APT_PKG"

    # CUDA 11.1's nvcc rejects gcc >= 10. 18.04 ships gcc 7, so this is
    # normally already right; assert rather than assume.
    local gccv
    gccv="$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo '?')"
    if [[ "$gccv" == "$GCC_VERSION" ]]; then ok "gcc $gccv"
    else warn "gcc $gccv — CUDA $CUDA_VERSION needs gcc $GCC_VERSION for nvcc"; fi

    _cuda_symlink_check
}

# /usr/local/cuda must point at the pinned toolkit: every downstream
# build (xGPU, mbheimdall, dsart's setup.py) resolves nvcc through it.
_cuda_symlink_check() {
    local tgt
    tgt="$(readlink -f /usr/local/cuda 2>/dev/null || echo '')"
    if [[ "$tgt" == "/usr/local/cuda-$CUDA_VERSION" ]]; then
        ok "/usr/local/cuda -> cuda-$CUDA_VERSION"
    else
        warn "/usr/local/cuda -> ${tgt:-missing}"
        run "point /usr/local/cuda at cuda-$CUDA_VERSION" \
            -- ln -sfn "/usr/local/cuda-$CUDA_VERSION" /usr/local/cuda
    fi
}
