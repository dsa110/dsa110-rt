# Mellanox ConnectX (mlx5_core) on the 10.41 data fabric.
#
# The in-tree mlx5_core that ships with the 18.04 kernel is what the
# fleet actually runs -- `ethtool -i` on n03 reports the distro driver,
# not the MLNX_OFED stack. The MLNX_EN 4.7 bundle on the MaaS
# fileserver is therefore NOT installed by default: pulling it in would
# replace a working in-tree driver with an out-of-tree one that has to
# be rebuilt on every kernel bump, which is a maintenance burden the
# fleet does not currently carry.
#
# What matters here is the firmware tools (mft), used to read/set port
# config, and confirming the link is up at the right speed.

stage_mellanox() {
    local dev; dev="$(data_iface)"
    if [[ -z "$dev" ]]; then
        warn "no data interface; skipping Mellanox checks"
        return 0
    fi

    local drv
    drv="$(basename "$(readlink "/sys/class/net/$dev/device/driver" 2>/dev/null)" 2>/dev/null)"
    if [[ "$drv" == "mlx5_core" ]]; then
        ok "driver mlx5_core on $dev (in-tree)"
    elif [[ -z "$drv" ]]; then
        # A bridge has no ->device/driver; find the enslaved port.
        local slave
        slave="$(ls "/sys/class/net/$dev/brif" 2>/dev/null | head -1)"
        if [[ -n "$slave" ]]; then
            local sdrv
            sdrv="$(basename "$(readlink "/sys/class/net/$slave/device/driver" 2>/dev/null)" 2>/dev/null)"
            ok "$dev is a bridge over $slave (driver ${sdrv:-unknown})"
        else
            warn "$dev has no driver and no bridge members"
        fi
    else
        warn "driver on $dev is '$drv', expected mlx5_core"
    fi

    if have ethtool; then
        local spd
        spd="$(ethtool "$dev" 2>/dev/null | awk -F': ' '/Speed:/{print $2}')"
        [[ -n "$spd" ]] && ok "link speed $spd" || log "  [ -- ] link speed unavailable (bridge)"
    else
        run "install ethtool" -- apt-get -y install ethtool
    fi

    # Firmware tools. Optional -- only needed to change port type or
    # flash firmware, neither of which a routine rebuild does.
    if have mst; then
        ok "mellanox firmware tools (mst) present"
    else
        log "  [ -- ] mft not installed; only needed for firmware/port-type work"
        log "         installer: $WEB_MAAS/scripts/00-maas-00-mftinstall.sh"
    fi
}
