# dsart: editable install + compiled extensions.
#
# On a live node dsart is an *editable* install -- site-packages holds
# __editable__.dsart-0.0.1.pth pointing at ~/proj/dsa110-rt/src/dsart --
# so the checked-out tree is the running code. That is deliberate: it is
# how a fix reaches the fleet without a reinstall, and the provisioning
# must reproduce it rather than doing a regular install.
#
# `pip install -e .` triggers setup.py, which compiles three artefacts:
#   _recv_ring, _recv_epoll   (POSIX-shm receive ring, C)
#   dsart_capture_manythread  (SNAP-UDP -> PSRDADA capture binary)
# These link against the PSRDADA from the builds stage, which is why
# this runs after it.

stage_dsart() {
    local env="$MINIFORGE_PREFIX/envs/$DSART_ENV_NAME"
    local py="$env/bin/python"
    local rt="$PROJ_DIR/dsa110-rt"

    if [[ ! -x "$py" ]]; then
        warn "conda env '$DSART_ENV_NAME' missing — run the 'conda' stage first"
        return 0
    fi
    if [[ ! -d "$rt" ]]; then
        warn "$rt missing — run the 'repos' stage first"
        return 0
    fi

    # --- editable install -------------------------------------------------
    if "$py" -c 'import dsart' >/dev/null 2>&1; then
        local loc
        loc="$("$py" -c 'import dsart,os;print(os.path.dirname(dsart.__file__))' 2>/dev/null)"
        if [[ "$loc" == "$rt/src/dsart" ]]; then
            ok "dsart editable install -> $loc"
        else
            warn "dsart imports from $loc, expected $rt/src/dsart (not editable?)"
        fi
    else
        run_sh "pip install -e '.[dev]' into $DSART_ENV_NAME" \
            "cd '$rt' && sudo -u ubuntu '$env/bin/pip' install -e '.[dev]'"
    fi

    # --- compiled extensions ----------------------------------------------
    # Keyed to the interpreter ABI, so an env rebuild on a different
    # python needs these regenerated. Check for the tag this env uses
    # rather than "any .so present", which would pass on a stale build
    # from a previous python.
    local tag
    tag="$("$py" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX") or "")' 2>/dev/null)"
    if [[ -n "$tag" && -f "$rt/src/dsart/transport/_recv_ring$tag" ]]; then
        ok "C extensions built for this interpreter ($tag)"
    else
        run_sh "build C extensions in place" \
            "cd '$rt' && sudo -u ubuntu '$py' setup.py build_ext --inplace"
    fi

    # --- capture binary ----------------------------------------------------
    if [[ -x "$rt/src/dsart/capture/dsart_capture_manythread" ]]; then
        ok "dsart_capture_manythread present"
    else
        warn "dsart_capture_manythread not built — expected from setup.py;"
        warn "  check that PSRDADA headers are under /usr/local/include"
    fi

    run "chown dsa110-rt to ubuntu" -- chown -R ubuntu:ubuntu "$rt"
}
