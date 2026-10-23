# Run Harness in OpenShell

OpenShell runs the whole Harness process in a Linux sandbox. Its external
policy applies to Harness file access, shell commands, MCP processes, and
subagents. Harness itself is not a sandbox.

These instructions target OpenShell v0.1.5. They use native OpenRouter requests,
not the removed `openshell inference` route.

## Container image

The Container GitHub Actions workflow publishes to:

```text
ghcr.io/alexlayton/harness:v<VERSION>
ghcr.io/alexlayton/harness:latest
```

Each tag contains Linux AMD64 and ARM64 images. On Apple Silicon, OpenShell
can select the ARM64 image without emulating the Harness process. The workload
is Linux, not macOS. OpenShell requires Docker Desktop and the kernel features
listed in its [support matrix][support].

The Ubuntu 24.04 image contains Harness, CA certificates, Git, curl, and `sh`.
It runs as the unprivileged `harness` user (UID 10001). It does not contain
Rust, Node.js, Python, MCP servers, or project build tools. Add those in a
derived image when needed:

```dockerfile
FROM ghcr.io/alexlayton/harness:v<VERSION>
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
USER harness
```

The executable stays at `/usr/local/bin/harness`, including in derived images.
The image uses `CMD`, not `ENTRYPOINT`, so OpenShell can select the command.
For direct Docker use, supply `harness` explicitly when overriding the command.
Do not treat plain Docker execution as equivalent to OpenShell policy enforcement.

## Configure OpenRouter access

There are three separate names:

- **Harness provider:** `--provider openrouter` selects Harness's API client.
- **OpenShell provider profile:** `harness-openrouter` defines the credential
  name, permitted endpoint, and permitted executable path. Its definition is
  [`examples/openshell/harness-openrouter.yaml`](../examples/openshell/harness-openrouter.yaml).
- **OpenShell provider instance:** `harness-router` below stores your credential
  and uses that profile. A sandbox receives access when you attach the instance.

Start and select an OpenShell gateway as described in the OpenShell
[installation guide][installation]. From this repository, review the profile,
then run these commands on the host:

```sh
openshell profile lint -f examples/openshell/harness-openrouter.yaml
openshell profile import -f examples/openshell/harness-openrouter.yaml

# Set OPENROUTER_API_KEY in the host shell before this command.
openshell provider create \
  --name harness-router \
  --type harness-openrouter \
  --from-existing
```

Keep the real key out of the image, Docker build arguments, project files, and
sandbox command. OpenShell keeps it outside the workload and supplies an opaque
`OPENROUTER_API_KEY` placeholder. It substitutes the real value only for an
allowed request to the bound endpoint.

The profile permits `/usr/local/bin/harness` to call `openrouter.ai:443`.
It does not grant curl, shell processes, or arbitrary interpreters that access.
It does not restrict model selection; Harness selects the model. Extend policy
separately for source hosts, package registries, and remote MCP services.

## First request

Replace `<VERSION>` with a published image version and `<MODEL_ID>` with an
OpenRouter model ID available to your account:

```sh
openshell sandbox create \
  --name harness-demo \
  --from ghcr.io/alexlayton/harness:v<VERSION> \
  --provider harness-router \
  -- harness --provider openrouter --model '<MODEL_ID>' \
     prompt --no-session 'Reply with OK. Do not use tools.'
```

The image must first be published and its GHCR package must be public. No
registry login is needed to pull a public image.

Check this request before adding project access. It checks the executable
path, credential substitution, certificate trust, and streamed response path.
The image installs system CA certificates; OpenShell must also make its proxy
CA available through the sandbox trust setup. Do not disable TLS verification
if that check fails.

To inspect provider-derived policy:

```sh
openshell policy get harness-demo --full
openshell sandbox provider list harness-demo
```

OpenShell denies destinations outside the effective policy. Review proposed
rules before approval; do not grant every denied destination automatically.
See [provider profiles][profiles] and [inference][inference] for diagnosis.

## Project files, state, and terminal use

Use OpenShell's [sandbox file and lifecycle tools][sandboxes] to transfer a
project copy into `/workspace`. Keep it writable by UID 10001. Start Harness
with `/workspace` as the working directory. Do not expose your whole home
directory, host authentication files, or Docker socket to the workload.

The image defines these writable paths:

| Purpose | Path | Environment override |
| --- | --- | --- |
| Project | `/workspace` | Working directory |
| Configuration | `/home/harness/.config/harness` | `HARNESS_CONFIG_DIR` |
| State, including sessions | `/home/harness/.harness` | `HARNESS_STATE_DIR` |

Configure persistent storage or export files through OpenShell before deleting
the sandbox. An image alone does not preserve sessions across sandbox removal.
The first check uses `--no-session` to avoid session state.

For terminal use, run `harness --provider openrouter --model '<MODEL_ID>'`
through an interactive OpenShell connection. Confirm terminal input and output
before using it for a longer task. Add local MCP executables to a derived image
and grant remote MCP endpoints separately. Harness can use process-group
cleanup without writable cgroup delegation; it does not need a privileged
container to start.

## Release and publication

1. The Release workflow builds and smoke-checks native Linux x86-64 and ARM64
   binaries, then creates a **draft** GitHub release with archives and checksums.
2. Publish that release in GitHub. The Container workflow downloads both Linux
   archives, verifies their checksums, and builds the multi-platform image.
3. The workflow pushes `v<VERSION>`. A published stable release also updates
   `latest`; a prerelease does not. Manual runs take an existing release tag
   and publish only that version tag.
4. On the first push, check the package settings and set visibility to **Public**.
   A public repository does not guarantee a newly created package is public.
   Repository policy must permit Actions to write packages.

GHCR container storage and bandwidth are currently free. Standard GitHub-hosted
Actions runners are free for public repositories; private repositories and
larger runners have separate billing rules. See [GitHub Packages billing][billing]
and [Actions billing][actions-billing]. No personal access token is required by
this workflow; it uses the job-scoped `GITHUB_TOKEN` with `packages: write`.

The Dockerfile **packages release binaries**; it does not compile Rust. This
avoids a second Rust build under ARM emulation and keeps local builds small.
For a local image build, download and checksum-verify the release archives,
then extract their executables to:

```text
dist/linux/amd64/harness
dist/linux/arm64/harness
```

Build one native platform locally, or both with a Buildx builder that supports
them:

```sh
docker buildx build --platform linux/arm64 \
  --build-arg VERSION=<VERSION> --load -t harness:local .
```

Linux standalone binaries have different minimum glibc versions: x86-64 uses
2.35 (Ubuntu 22.04); ARM64 uses 2.39 (Ubuntu 24.04). Both run in the Ubuntu 24.04
container image.

[support]: https://docs.nvidia.com/openshell/about/support-matrix
[installation]: https://docs.nvidia.com/openshell/about/installation
[profiles]: https://docs.nvidia.com/openshell/how-it-works/providers/profiles
[inference]: https://docs.nvidia.com/openshell/how-it-works/inference
[sandboxes]: https://docs.nvidia.com/openshell/how-it-works/sandboxes/overview
[billing]: https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-packages/about-billing-for-github-packages
[actions-billing]: https://docs.github.com/en/billing/concepts/product-billing/github-actions
