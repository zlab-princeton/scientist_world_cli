# sciworld agent container — docker/podman build.
# Kept deliberately in step with agent.def (apptainer/singularity equivalent).
# If you change one, change the other.

# Fully qualified, because Podman may refuse a short image name or stop to
# ask which registry is meant.
FROM docker.io/library/ubuntu:24.04

LABEL org.opencontainers.image.authors="ScientistWorld"
LABEL org.opencontainers.image.description="Agent container — runtime for Claude Code / Codex"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl wget git ca-certificates gnupg \
        python3 python3-pip python3-venv \
        build-essential unzip tar gzip \
    && rm -rf /var/lib/apt/lists/*

# Node.js (LTS) — required for both claude-code and codex CLIs
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI. The installer puts it under /root/.local. sciworld runs
# the container as the calling user (--user UID:GID), so /root and that tree
# must be readable by everyone, and the command is linked into /usr/local/bin.
RUN (curl -fsSL https://claude.ai/install.sh | bash || echo "WARNING: Claude Code could not be installed") \
    && if [ -e /root/.local/bin/claude ]; then \
           chmod a+rx /root \
           && chmod -R a+rX /root/.local \
           && ln -sf /root/.local/bin/claude /usr/local/bin/claude; \
       fi

# Codex CLI (npm global install: lands in /usr/lib/node_modules, readable
# by everyone)
RUN npm install -g @openai/codex || echo "WARNING: Codex could not be installed"

# An image without any agent CLI is of no use: fail the build.
RUN command -v claude >/dev/null 2>&1 || command -v codex >/dev/null 2>&1 \
    || { echo "ERROR: neither Claude Code nor Codex could be installed" >&2; exit 1; }

# Python tools
RUN pip install --break-system-packages uv

# pip → uv shim (10-100x faster for install/uninstall)
RUN mv /usr/bin/pip3 /usr/bin/pip3-orig 2>/dev/null || true \
    && mv /usr/bin/pip  /usr/bin/pip-orig  2>/dev/null || true
RUN printf '%s\n' \
    '#!/bin/bash' \
    'case "${1:-}" in' \
    '    install|uninstall|freeze|list|show|check)' \
    '        exec uv pip "$@" ;;' \
    '    *)' \
    '        exec /usr/bin/pip-orig "$@" ;;' \
    'esac' \
    > /usr/local/bin/pip \
    && chmod +x /usr/local/bin/pip \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3

ENV PATH="/usr/local/bin:/root/.local/bin:${PATH}"
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

# No CMD/ENTRYPOINT — sciworld always invokes with explicit argv.
