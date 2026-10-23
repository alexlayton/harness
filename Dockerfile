# syntax=docker/dockerfile:1
# Package the checked release binaries; Rust compilation stays in release CI.
# See docs/openshell.md for the dist/linux/{amd64,arm64}/harness layout.
FROM ubuntu:24.04

ARG TARGETARCH
ARG VERSION=dev
LABEL org.opencontainers.image.title="Harness" \
      org.opencontainers.image.description="Terminal coding agent for container and OpenShell workloads" \
      org.opencontainers.image.source="https://github.com/alexlayton/harness" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"

# Keep the base useful but small. Derived images supply project-specific tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /bin/sh harness \
    && mkdir -p /workspace /home/harness/.config/harness /home/harness/.harness \
    && chown -R harness:harness /workspace /home/harness

COPY --chmod=755 dist/linux/${TARGETARCH}/harness /usr/local/bin/harness
COPY LICENSE /usr/share/doc/harness/LICENSE

ENV HOME=/home/harness \
    HARNESS_CONFIG_DIR=/home/harness/.config/harness \
    HARNESS_STATE_DIR=/home/harness/.harness
USER harness
WORKDIR /workspace
RUN harness --help >/dev/null
# CMD, not ENTRYPOINT: OpenShell must be able to supply its own command.
CMD ["harness"]
