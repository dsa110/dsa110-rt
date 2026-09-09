# Kernel + fabric tuning.
#
# These values are not preferences. The SNAP-UDP capture pushes into a
# 256 MB SO_RCVBUF (search_rx --so-rcvbuf-bytes 268435456), which the
# kernel silently clamps to net.core.rmem_max; leave rmem_max at its
# 208 KB default and the capture drops packets under load with nothing
# in any log to explain it. Likewise PSRDADA allocates ~33 GiB of SysV
# shared memory across ~450 segments, which needs kernel.shmmax raised.

stage_tuning() {
    # --- sysctl --------------------------------------------------------
    # config/sysctl/set_sysctl on the MaaS fileserver already carries
    # exactly these values and matches the live fleet, so prefer it and
    # fall back to writing them locally if the fetch fails.
    local want=(
        "net.core.rmem_max=$SYSCTL_RMEM_MAX"
        "net.core.wmem_max=$SYSCTL_WMEM_MAX"
        "net.core.rmem_default=$SYSCTL_RMEM_DEFAULT"
        "net.core.wmem_default=$SYSCTL_RMEM_DEFAULT"
        "net.core.optmem_max=16777216"
        "net.core.netdev_max_backlog=$SYSCTL_NETDEV_BACKLOG"
        "kernel.shmmax=$SYSCTL_SHMMAX"
        "kernel.shmall=$SYSCTL_SHMALL"
        "net.ipv4.conf.all.rp_filter=0"
        "net.ipv4.tcp_timestamps=0"
        "net.ipv4.tcp_sack=0"
        "net.ipv4.tcp_low_latency=1"
        "net.ipv4.tcp_adv_win_scale=1"
    )
    local conf="/etc/sysctl.d/99-dsa110-node.conf"
    local pending=()
    for kv in "${want[@]}"; do
        local k="${kv%%=*}" v="${kv#*=}" cur
        cur="$(sysctl -n "$k" 2>/dev/null || echo '')"
        if [[ "$cur" == "$v" ]]; then ok "sysctl $k=$v"; else pending+=("$kv"); fi
    done

    if [[ ${#pending[@]} -gt 0 ]]; then
        # Own our settings in a single file rather than appending to
        # /etc/sysctl.conf. set_sysctl appends on every run, so a node
        # provisioned twice ends up with duplicate lines -- harmless but
        # it makes drift impossible to read.
        local body="# DSA-110 node tuning. Managed by tools/provision.\n"
        for kv in "${pending[@]}"; do body+="$kv\n"; done
        body+='net.ipv4.tcp_mem = 16777216 16777216 16777216\n'
        body+='net.ipv4.tcp_rmem = 4096 87380 16777216\n'
        body+='net.ipv4.tcp_wmem = 4096 87380 16777216\n'
        run_sh "write $conf (${#pending[@]} settings)" \
               "printf '%b' '$body' > $conf"
        run "apply sysctl" -- sysctl -q -p "$conf"
    fi

    # --- data-fabric MTU ------------------------------------------------
    # Jumbo frames on the 10.41 fabric. The SNAP packets assume 9000; at
    # 1500 they fragment and the capture falls apart.
    local dev; dev="$(data_iface)"
    if [[ -n "$dev" ]]; then
        local mtu
        mtu="$(cat "/sys/class/net/$dev/mtu" 2>/dev/null || echo 0)"
        if [[ "$mtu" == "$DATA_MTU" ]]; then
            ok "mtu $DATA_MTU on $dev"
        else
            warn "mtu on $dev is $mtu, want $DATA_MTU"
            # Runtime change only. The durable setting belongs in netplan,
            # which cloud-init lays down -- setting it here as well would
            # give two owners for one value and hide netplan being wrong.
            run "set mtu $DATA_MTU on $dev (runtime; fix netplan too)" \
                -- ip link set dev "$dev" mtu "$DATA_MTU"
        fi
    else
        warn "no data interface detected; skipping MTU"
    fi

    # --- /dev/shm -------------------------------------------------------
    # POSIX shm backs the transport receive ring (_recv_ring). Default
    # tmpfs is half of RAM, which is already ample here; report only.
    local shm
    shm="$(df -BG /dev/shm 2>/dev/null | awk 'NR==2{gsub("G","",$2); print $2}')"
    if [[ "${shm:-0}" -ge 40 ]]; then ok "/dev/shm ${shm}G"
    else warn "/dev/shm only ${shm}G (corr runs 87G, search 47G)"; fi
}
