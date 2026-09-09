# Preflight: assert the machine looks like a DSA-110 node before we
# start changing it, and surface anything that would make a later stage
# silently do the wrong thing.
#
# This stage never mutates. Its whole job is to fail loudly here rather
# than half-way through a CUDA install.

stage_preflight() {
    local rel
    rel="$(lsb_release -rs 2>/dev/null || echo '?')"
    if [[ "$rel" == "$OS_RELEASE" ]]; then
        ok "Ubuntu $rel"
    else
        warn "Ubuntu $rel, expected $OS_RELEASE — CUDA $CUDA_VERSION and the"
        warn "  prebuilt dsaX binaries are only validated on $OS_RELEASE"
    fi

    local ncpu mem
    ncpu="$(nproc 2>/dev/null || echo 0)"
    mem="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
    [[ "$ncpu" -ge 32 ]] && ok "cpu: $ncpu cores" || warn "cpu: only $ncpu cores (fleet has 40)"
    [[ "${mem:-0}" -ge 120 ]] && ok "ram: ${mem} GiB" || warn "ram: ${mem} GiB (fleet has ~172)"

    # GPUs. nvidia-smi is absent on a fresh deploy; that is expected and
    # is exactly what the cuda stage fixes, so it is not a warning.
    if have nvidia-smi; then
        local n drv
        n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
        drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        if [[ "$n" == "2" ]]; then ok "gpu: 2 present, driver $drv"
        else warn "gpu: found $n, expected 2"; fi
        [[ "$drv" == "$NVIDIA_DRIVER_VERSION" ]] \
            || warn "gpu: driver $drv != pinned $NVIDIA_DRIVER_VERSION"
    else
        log "  [ -- ] nvidia-smi absent (fresh deploy; the cuda stage installs it)"
    fi

    # Data fabric. Which interface carries 10.41 differs by role: corr
    # uses the raw Mellanox port, search uses a bridge. Getting this
    # wrong means MTU 9000 lands on the wrong device and the capture
    # silently fragments.
    DATA_IFACE="$(data_iface)"
    if [[ -n "$DATA_IFACE" ]]; then
        ok "data fabric: $DATA_IFACE on ${DATA_SUBNET}.x"
        local hint
        hint="$([[ "$ROLE" == corr ]] && echo "$CORR_DATA_IFACE_HINT" || echo "$SEARCH_DATA_IFACE_HINT")"
        [[ "$DATA_IFACE" == "$hint" ]] \
            || warn "  expected '$hint' for role $ROLE — check the netplan before MTU changes"
    else
        warn "no interface on ${DATA_SUBNET}.x — netplan has not applied yet"
    fi
    export DATA_IFACE

    # The MaaS fileserver is the source for every download below. The
    # legacy scripts alternated between this IP and
    # http://lxd110maas.ovro.pvt/, and that name does not resolve --
    # a step pointed at it fetches nothing and carries on regardless.
    if curl -fsS -m 10 -o /dev/null "$WEB_MAAS/config/sysctl/set_sysctl" 2>/dev/null; then
        ok "maas fileserver reachable: $WEB_MAAS"
    else
        warn "cannot fetch from $WEB_MAAS — every download stage will no-op"
    fi

    [[ -d "$PROJ_DIR" ]] && ok "proj dir: $PROJ_DIR" || log "  [ -- ] $PROJ_DIR will be created"
}
