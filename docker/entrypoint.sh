#!/usr/bin/env bash
set -euo pipefail

USERNAME=dev
HOME_DIR="/home/${USERNAME}"

# Detect the UID/GID of the mounted /workspace directory.
# This matches whatever user owns the bind-mounted host directory.
HOST_UID="$(stat -c '%u' /workspace 2>/dev/null || echo 1000)"
HOST_GID="$(stat -c '%g' /workspace 2>/dev/null || echo 1000)"

# If running as root (normal case on first start), create/adjust the dev user.
if [ "$(id -u)" = "0" ]; then
    # Create group if it doesn't exist with the target GID
    if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
        groupadd -g "${HOST_GID}" "${USERNAME}"
    fi

    # Create or update user
    if id "${USERNAME}" >/dev/null 2>&1; then
        usermod -u "${HOST_UID}" -g "${HOST_GID}" -d "${HOME_DIR}" "${USERNAME}"
    else
        useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/zsh -d "${HOME_DIR}" "${USERNAME}"
    fi

    # Passwordless sudo
    grep -q "^${USERNAME}" /etc/sudoers 2>/dev/null || \
        echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

    # Set up shell config if missing
    if [ ! -d "${HOME_DIR}/.oh-my-zsh" ] && [ -d /root/.oh-my-zsh ]; then
        cp -r /root/.oh-my-zsh "${HOME_DIR}/.oh-my-zsh"
        cp /root/.zshrc "${HOME_DIR}/.zshrc"
        sed -i "s|/root|${HOME_DIR}|g" "${HOME_DIR}/.zshrc"
    fi

    chown -R "${HOST_UID}:${HOST_GID}" "${HOME_DIR}"

    # Re-run git safe.directory for the new user
    gosu "${USERNAME}" git config --global --add safe.directory '*'

    # Drop privileges and exec the requested command
    exec gosu "${USERNAME}" "$@"
fi

# Already running as non-root (e.g. --user was passed), just exec
exec "$@"
