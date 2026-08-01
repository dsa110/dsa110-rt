# Python environments.
#
# A live node carries two independent installs, and conflating them is
# an easy mistake:
#
#   miniforge3  -> env "dsa110-rt" (py3.11). This is what dsart runs
#                  from; the orchestrator's interpreter on n03 is
#                  /home/ubuntu/miniforge3/envs/dsa110-rt/bin/python3.11.
#   anaconda3   -> env "casa38" (py3.8). Legacy calibration tooling.
#
# The legacy cloud-init provisioned anaconda3 and nothing else, so a
# node rebuilt from it could not run dsart at all. miniforge is the one
# that matters; anaconda is opt-in via INSTALL_ANACONDA.

stage_conda() {
    # --- miniforge ------------------------------------------------------
    if [[ -x "$MINIFORGE_PREFIX/bin/conda" ]]; then
        ok "miniforge at $MINIFORGE_PREFIX ($("$MINIFORGE_PREFIX/bin/conda" --version 2>/dev/null))"
    else
        run_sh "download miniforge" \
            "cd /run && curl -fsSLO '$WEB_MAAS/scripts/$MINIFORGE_INSTALLER'"
        run_sh "install miniforge -> $MINIFORGE_PREFIX" \
            "bash /run/$MINIFORGE_INSTALLER -b -p '$MINIFORGE_PREFIX'"
        run "chown miniforge to ubuntu" \
            -- chown -R ubuntu:ubuntu "$MINIFORGE_PREFIX"
    fi

    # --- dsa110-rt env --------------------------------------------------
    # Built from the repo's own envs/dsa110-rt.yml so the env and the
    # code that needs it version together. Requires the repos stage to
    # have run; on a first pass that ordering is guaranteed by STAGES.
    local yml="$PROJ_DIR/dsa110-rt/$DSART_ENV_YML"
    if [[ -d "$MINIFORGE_PREFIX/envs/$DSART_ENV_NAME" ]]; then
        ok "conda env '$DSART_ENV_NAME' exists"
    elif [[ -f "$yml" ]]; then
        run_sh "create conda env '$DSART_ENV_NAME' from $DSART_ENV_YML" \
            "'$MINIFORGE_PREFIX/bin/conda' env create -f '$yml' -n '$DSART_ENV_NAME'"
        run "chown env to ubuntu" \
            -- chown -R ubuntu:ubuntu "$MINIFORGE_PREFIX/envs/$DSART_ENV_NAME"
    else
        warn "env spec not found at $yml — run the 'repos' stage first"
    fi

    # --- conda init for the ubuntu user ---------------------------------
    # The launch path does `conda activate dsa110-rt` over a
    # non-interactive ssh, so the hook has to be in .bashrc.
    if grep -q "conda initialize" /home/ubuntu/.bashrc 2>/dev/null; then
        ok "conda hook present in ~ubuntu/.bashrc"
    else
        run_sh "conda init bash for ubuntu" \
            "sudo -u ubuntu '$MINIFORGE_PREFIX/bin/conda' init bash"
    fi

    # --- anaconda3 (optional) -------------------------------------------
    if [[ "$INSTALL_ANACONDA" == "yes" ]]; then
        if [[ -x "$ANACONDA_PREFIX/bin/conda" ]]; then
            ok "anaconda3 at $ANACONDA_PREFIX"
        else
            run_sh "download anaconda3" \
                "cd /run && curl -fsSLO '$WEB_MAAS/scripts/$ANACONDA_INSTALLER'"
            run_sh "install anaconda3 -> $ANACONDA_PREFIX" \
                "bash /run/$ANACONDA_INSTALLER -b -p '$ANACONDA_PREFIX'"
            run "chown anaconda3 to ubuntu" \
                -- chown -R ubuntu:ubuntu "$ANACONDA_PREFIX"
        fi
    else
        log "  [ -- ] anaconda3 skipped (INSTALL_ANACONDA=no; legacy casa38 only)"
    fi
}
