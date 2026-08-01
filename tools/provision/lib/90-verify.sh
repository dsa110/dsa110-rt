# Final assertion pass: does this node now match the fleet baseline?
#
# Read-only in both modes, so it is also useful standalone against an
# existing node to find drift:
#     ./provision_node.sh --role corr --only verify

stage_verify() {
    local fail=0 pass=0
    _v() {  # _v <label> <expected> <actual>
        if [[ "$2" == "$3" ]]; then ok "$1 = $3"; pass=$((pass+1))
        else warn "$1: got '${3:-<none>}', want '$2'"; fail=$((fail+1)); fi
    }

    _v "sysctl net.core.rmem_max"  "$SYSCTL_RMEM_MAX"      "$(sysctl -n net.core.rmem_max 2>/dev/null)"
    _v "sysctl net.core.wmem_max"  "$SYSCTL_WMEM_MAX"      "$(sysctl -n net.core.wmem_max 2>/dev/null)"
    _v "sysctl netdev_max_backlog" "$SYSCTL_NETDEV_BACKLOG" "$(sysctl -n net.core.netdev_max_backlog 2>/dev/null)"
    _v "sysctl kernel.shmmax"      "$SYSCTL_SHMMAX"        "$(sysctl -n kernel.shmmax 2>/dev/null)"
    _v "sysctl kernel.shmall"      "$SYSCTL_SHMALL"        "$(sysctl -n kernel.shmall 2>/dev/null)"

    local dev; dev="$(data_iface)"
    [[ -n "$dev" ]] && _v "mtu on $dev" "$DATA_MTU" \
        "$(cat "/sys/class/net/$dev/mtu" 2>/dev/null)"

    _v "cuda release" "$CUDA_VERSION" \
       "$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+')"
    _v "nvidia driver" "$NVIDIA_DRIVER_VERSION" \
       "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    _v "gpu count" "2" \
       "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"

    have dada_db && { ok "psrdada: dada_db present"; pass=$((pass+1)); } \
                 || { warn "psrdada: dada_db missing"; fail=$((fail+1)); }

    local R="$PROJ_DIR/dsa110-shell"
    for t in dsa110-psrdada dsa110-xGPU dsa110-sigproc dsa110-xengine dsa110-mbheimdall; do
        [[ -d "$R/$t" ]] && { ok "repo $t"; pass=$((pass+1)); } \
                         || { warn "repo $t missing"; fail=$((fail+1)); }
    done
    [[ -d "$R/dsa110-xengine/utils" ]] \
        && { ok "xengine utils/ (bfweights antennas.out target)"; pass=$((pass+1)); } \
        || { warn "xengine utils/ missing — bfweights push will fail"; fail=$((fail+1)); }

    local py="$MINIFORGE_PREFIX/envs/$DSART_ENV_NAME/bin/python"
    if [[ -x "$py" ]] && "$py" -c 'import dsart' >/dev/null 2>&1; then
        ok "dsart importable from $DSART_ENV_NAME"; pass=$((pass+1))
        local loc tag
        loc="$("$py" -c 'import dsart,os;print(os.path.dirname(dsart.__file__))')"
        _v "dsart source tree" "$PROJ_DIR/dsa110-rt/src/dsart" "$loc"
        tag="$("$py" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))' 2>/dev/null)"
        for ext in _recv_ring _recv_epoll; do
            [[ -f "$PROJ_DIR/dsa110-rt/src/dsart/transport/$ext$tag" ]] \
                && { ok "ext $ext$tag"; pass=$((pass+1)); } \
                || { warn "ext $ext$tag missing"; fail=$((fail+1)); }
        done
    else
        warn "dsart not importable from $DSART_ENV_NAME"; fail=$((fail+1))
    fi

    if [[ "$ROLE" == "search" ]]; then
        [[ -f "$SEARCH_DM_PLAN_DIR/$SEARCH_DM_PLAN" ]] \
            && { ok "DM plan"; pass=$((pass+1)); } \
            || { warn "DM plan missing — search_compute will not start"; fail=$((fail+1)); }
    fi

    log ""
    log "  verify: $pass ok, $fail need attention"
    if [[ $fail -gt 0 ]]; then
        log "  ${C_WARN}node does not yet match the fleet baseline${C_OFF}"
    else
        log "  ${C_OK}node matches the fleet baseline${C_OFF}"
    fi
}
