#!/usr/bin/env bash
# Set up a headless NVIDIA X display on the HOST for SAPIEN viewer rendering.
# This is the non-fallback path that provides Vulkan present support.

set -euo pipefail
set +m  # Disable job control to avoid "[1] PID" output from background jobs

DISPLAY_NUM="${DISPLAY_NUM:-:0}"
DISPLAY_WIDTH="${DISPLAY_WIDTH:-1920}"
DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-1080}"
VNC_PORT="${VNC_PORT:-5900}"
XORG_CONF="${XORG_CONF:-/tmp/elytar-xorg-headless.conf}"
XORG_LOG="${XORG_LOG:-/tmp/elytar-xorg.log}"

if [[ -f "/.dockerenv" ]]; then
  echo "Error: run scripts/host_display_setup.sh on the HOST, not inside Docker." >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

echo "=== Host NVIDIA display setup ==="

require_cmd nvidia-smi
require_cmd nvidia-xconfig
require_cmd Xorg
require_cmd x11vnc
require_cmd fluxbox
require_cmd xdpyinfo

# If display already works, skip restart (fixes second-run failure)
if DISPLAY="${DISPLAY_NUM}" xdpyinfo >/dev/null 2>&1; then
  pkill -x fluxbox 2>/dev/null || true
  nohup env DISPLAY="${DISPLAY_NUM}" fluxbox >/tmp/elytar-fluxbox.log 2>&1 &
  pkill -x x11vnc 2>/dev/null || true
  nohup x11vnc -display "${DISPLAY_NUM}" -forever -localhost -nopw -rfbport "${VNC_PORT}" >/tmp/elytar-x11vnc.log 2>&1 &
  if command -v xhost >/dev/null 2>&1; then
    DISPLAY="${DISPLAY_NUM}" xhost +SI:localuser:root >/dev/null 2>&1 || true
  fi
  echo ""
  echo "Host NVIDIA display ready on ${DISPLAY_NUM}"
  echo "  Xorg log: ${XORG_LOG}"
  echo "  VNC: localhost:${VNC_PORT} (SSH tunnel: ssh -L ${VNC_PORT}:localhost:${VNC_PORT} <user>@<host>)"
  echo ""
  echo "In Docker: export DISPLAY=${DISPLAY_NUM}"
  echo "          python3 -m benchmark.run --tasks cube_stack --steps 20 --render"
  exit 0
fi

BUS_ID="$(nvidia-xconfig --query-gpu-info | awk -F': ' '/PCI BusID/{print $2; exit}')"
if [[ -z "${BUS_ID}" ]]; then
  echo "Failed to query NVIDIA GPU PCI bus ID via nvidia-xconfig." >&2
  exit 1
fi

cat > "${XORG_CONF}" <<EOF
Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0"
EndSection

Section "Device"
    Identifier "Device0"
    Driver "nvidia"
    BusID "${BUS_ID}"
    Option "AllowEmptyInitialConfiguration" "True"
    Option "UseDisplayDevice" "None"
    Option "HardDPMS" "false"
EndSection

Section "Screen"
    Identifier "Screen0"
    Device "Device0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual ${DISPLAY_WIDTH} ${DISPLAY_HEIGHT}
    EndSubSection
EndSection
EOF

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  require_cmd sudo
  SUDO="sudo"
fi

# Start/restart Xorg on DISPLAY_NUM.
${SUDO} pkill -f "Xorg ${DISPLAY_NUM}" 2>/dev/null || true
sleep 2  # Allow display socket to be released before restart
${SUDO} nohup Xorg "${DISPLAY_NUM}" \
  -ac \
  -config "${XORG_CONF}" \
  -noreset \
  -nolisten tcp \
  +extension GLX \
  +extension RANDR \
  +extension RENDER \
  -logfile "${XORG_LOG}" \
  >/tmp/elytar-xorg-stdout.log 2>&1 &

# Wait for X display to become available.
for _ in $(seq 1 40); do
  if DISPLAY="${DISPLAY_NUM}" xdpyinfo >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
if ! DISPLAY="${DISPLAY_NUM}" xdpyinfo >/dev/null 2>&1; then
  echo ""
  echo "Xorg failed to come up on ${DISPLAY_NUM}. Check ${XORG_LOG}"
  exit 1
fi

pkill -x fluxbox 2>/dev/null || true
nohup env DISPLAY="${DISPLAY_NUM}" fluxbox >/tmp/elytar-fluxbox.log 2>&1 &

pkill -x x11vnc 2>/dev/null || true
nohup x11vnc \
  -display "${DISPLAY_NUM}" \
  -forever \
  -localhost \
  -nopw \
  -rfbport "${VNC_PORT}" \
  >/tmp/elytar-x11vnc.log 2>&1 &

# Let root in container access host X display.
if command -v xhost >/dev/null 2>&1; then
  DISPLAY="${DISPLAY_NUM}" xhost +SI:localuser:root >/dev/null 2>&1 || true
fi

if ! grep -q "export DISPLAY=${DISPLAY_NUM}" "${HOME}/.zshrc" 2>/dev/null; then
  {
    echo ""
    echo "# SAPIEN host NVIDIA display (scripts/host_display_setup.sh)"
    echo "export DISPLAY=${DISPLAY_NUM}"
  } >> "${HOME}/.zshrc"
fi

echo ""
echo "Host NVIDIA display ready on ${DISPLAY_NUM}"
echo "  Xorg log: ${XORG_LOG}"
echo "  VNC: localhost:${VNC_PORT} (SSH tunnel: ssh -L ${VNC_PORT}:localhost:${VNC_PORT} <user>@<host>)"
echo ""
echo "In Docker: export DISPLAY=${DISPLAY_NUM}"
echo "          python3 -m benchmark.run --tasks cube_stack --steps 20 --render"
