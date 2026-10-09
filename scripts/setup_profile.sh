# Sourced by setup.sh. Treat saved profile lines as data, never shell code.
add_extra() {
  local candidate="$1"
  local existing
  for existing in "${INSTALL_EXTRAS[@]}"; do
    [[ "$existing" == "$candidate" ]] && return
  done
  INSTALL_EXTRAS+=("$candidate")
}

load_install_profile() {
  local profile="$1" option
  [[ -f "$profile" ]] || return 0
  while IFS= read -r option || [[ -n "$option" ]]; do
    case "$option" in
      --with-viewer) WITH_VIEWER=1 ;;
      --with-remote) add_extra remote ;;
      --with-birefnet) add_extra birefnet ;;
      --with-da3) add_extra da3 ;;
      --with-lpips) add_extra lpips ;;
      --with-tensorboard) add_extra tensorboard ;;
      *) echo "error: invalid saved install profile: $profile; use --reset-profile to replace it" >&2; return 1 ;;
    esac
  done < "$profile"
}

save_install_profile() {
  local profile="$1" temporary extra
  temporary="$(mktemp "${profile}.XXXXXX")" || return
  {
    if [[ "$WITH_VIEWER" -eq 1 ]]; then
      echo --with-viewer
    fi
    for extra in "${INSTALL_EXTRAS[@]}"; do
      echo "--with-$extra"
    done
  } > "$temporary"
  mv -f "$temporary" "$profile"
}
