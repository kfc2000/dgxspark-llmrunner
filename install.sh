#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${1:-$HOME/llmrunner}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --upgrade -q pip
"$INSTALL_DIR/.venv/bin/pip" install -q -r "$REPO_DIR/requirements.txt"

rsync_excludes=(--exclude .venv --exclude .git --exclude '__pycache__')
if [[ -f "$INSTALL_DIR/llms.json" ]]; then
  rsync_excludes+=(--exclude llms.json)
  echo "Keeping existing $INSTALL_DIR/llms.json (not overwriting)"
fi

rsync -a --delete "${rsync_excludes[@]}" \
  "$REPO_DIR/dashboard" "$REPO_DIR/llmrunner" "$REPO_DIR/llms.json" \
  "$INSTALL_DIR/"

service_file=/etc/systemd/system/llmrunner.service
sed -e "s|__USER__|$(id -un)|" -e "s|__INSTALL_DIR__|$INSTALL_DIR|g" \
  "$REPO_DIR/llmrunner.service" | sudo tee "$service_file" >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable llmrunner
# restart: enable --now is a no-op on a running service and the venv caches imported modules
sudo systemctl restart llmrunner
systemctl status llmrunner --no-pager | head -12

echo
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8501"
