# Source checkouts.
#
# Two trees:
#   ~/proj/dsa110-shell/  -- the myrepos (mr) collection: psrdada, xGPU,
#                            sigproc, xengine, mbheimdall, pyutils, ...
#   ~/proj/dsa110-rt/     -- the real-time pipeline (package `dsart`)
#
# dsa110-rt is absent from the legacy provisioning entirely, which alone
# would leave a rebuilt node unable to run the pipeline.
#
# Cloning is over https, not ssh: cloud-init has no deploy key and we
# are deliberately not putting one on the node from git. Anything
# private has to be added by hand afterwards.

stage_repos() {
    run_sh "ensure $PROJ_DIR" "mkdir -p '$PROJ_DIR' && chown ubuntu:ubuntu '$PROJ_DIR'"

    # --- dsa110-shell via myrepos ---------------------------------------
    if [[ -d "$PROJ_DIR/dsa110-shell/.git" ]]; then
        ok "dsa110-shell checked out"
    else
        have mr || run "install myrepos" -- apt-get -y install myrepos

        # .mrtrust must be in the invoking user's HOME or mr refuses the
        # .mrconfig. Under cloud-init that HOME is root's, not ubuntu's,
        # which is why the legacy script copied it to both.
        run_sh "fetch mr dotfiles" \
            "cd /home/ubuntu && for f in .mrtrust .gitconfig .gitignore; do
                 curl -fsSLO '$WEB_MAAS/config/homedirfiles/\$f' || true
             done
             cp -f /home/ubuntu/.mrtrust \"\$HOME/\" 2>/dev/null || true"
        run_sh "clone dsa110-shell" \
            "cd '$PROJ_DIR' && git clone '$SHELL_REPO'"
        # Rewrite the ssh remotes to https so an unauthenticated cloud-init
        # checkout succeeds; keep the original as .mrconfig_ssh so a human
        # can switch back for pushing.
        run_sh "rewrite mr remotes to https" \
            "cd '$PROJ_DIR/dsa110-shell' && cp -n .mrconfig .mrconfig_ssh || true;
             sed -i -e 's+git@github.com:+https://github.com/+g' .mrconfig"
        run_sh "mr checkout (all sub-repos)" \
            "cd '$PROJ_DIR/dsa110-shell' && /usr/bin/mr -t checkout"
    fi

    # --- dsa110-rt -------------------------------------------------------
    if [[ -d "$PROJ_DIR/dsa110-rt/.git" ]]; then
        ok "dsa110-rt checked out ($(git -C "$PROJ_DIR/dsa110-rt" rev-parse --abbrev-ref HEAD 2>/dev/null))"
    else
        run_sh "clone dsa110-rt ($RT_BRANCH)" \
            "cd '$PROJ_DIR' && git clone -b '$RT_BRANCH' '$RT_REPO'"
    fi

    run "chown $PROJ_DIR to ubuntu" -- chown -R ubuntu:ubuntu "$PROJ_DIR"
}
