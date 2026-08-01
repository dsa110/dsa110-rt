# Role-specific setup: the handful of things that differ between a corr
# node and a search node.
#
# Deliberately NOT here: starting dsart_rt. The orchestrator is launched
# by tools/ops/_m75_phaseB_16x4_launch.sh once the fleet is ready, and a
# freshly provisioned node joining the fabric on its own would be a
# surprise. Provisioning makes a node *ready*, not *running*.

stage_role() {
    run_sh "ensure /home/ubuntu/data" \
        "mkdir -p /home/ubuntu/data && chown ubuntu:ubuntu /home/ubuntu/data"

    case "$ROLE" in
        corr)   _role_corr ;;
        search) _role_search ;;
    esac
}

_role_corr() {
    # Capture writes hdf5 here and the calibration preprocess on h23
    # rsyncs them off with --remove-source-files.
    ok "corr: data dir is the capture target for hdf5 + voltage staging"
    run_sh "ensure voltage staging dir" \
        "mkdir -p /home/ubuntu/data/voltage_staging \
         && chown ubuntu:ubuntu /home/ubuntu/data/voltage_staging"

    # PSRDADA buffers are created by the orchestrator from the etcd
    # config (/cnf/pipeline_rt), not here -- reader counts are physics-
    # pinned and creating them out of band would fight dsart_rt.
    log "  [ -- ] PSRDADA buffers are created by dsart_rt from /cnf/pipeline_rt"

    local free
    free="$(df -BG --output=avail /home/ubuntu 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [[ "${free:-0}" -ge 200 ]]; then ok "corr: ${free}G free on /home/ubuntu"
    else warn "corr: only ${free}G free on /home/ubuntu; a voltage dump stages ~6.5G/node"; fi
}

_role_search() {
    # search_compute takes --dm-plan-path and will not start without it.
    # The plan is a binary artefact, not in git, so it comes from the
    # MaaS fileserver.
    if [[ -f "$SEARCH_DM_PLAN_DIR/$SEARCH_DM_PLAN" ]]; then
        ok "search: DM plan present ($SEARCH_DM_PLAN)"
    else
        run_sh "fetch DM plan" \
            "mkdir -p '$SEARCH_DM_PLAN_DIR' \
             && cd '$SEARCH_DM_PLAN_DIR' \
             && curl -fsSLO '$WEB_MAAS/config/dm_plans/$SEARCH_DM_PLAN' \
             && chown -R ubuntu:ubuntu '$SEARCH_DM_PLAN_DIR'"
        warn "if that 404s, the plan has not been staged to the MaaS"
        warn "  fileserver yet: copy it to /var/www/html/maas/config/dm_plans/"
    fi

    # The search side receives on a bridge rather than a raw port.
    local dev; dev="$(data_iface)"
    if [[ "$dev" == "$SEARCH_DATA_IFACE_HINT" ]]; then
        ok "search: data fabric on bridge $dev"
    else
        warn "search: data fabric is '$dev', fleet uses '$SEARCH_DATA_IFACE_HINT'"
    fi
}
