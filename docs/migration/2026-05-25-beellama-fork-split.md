# Migration plan — split beellama Docker stack into a proper fork

**Created:** 2026-05-24
**Target completion:** before next benchmark session
**Status:** PLANNED — not yet executed
**Owner:** Tommy (executes git/GitHub operations) + Claude (prepares files, drives sessions)

---

## Context

Right now everything lives in `TurboQuant_Benchmark/`:
- Benchmark scripts and reports (research artifacts — belongs here)
- Docker recipe for building/running beellama (operational — belongs elsewhere)
- Pre-built binary snapshot in `docs/deployment/docker/artifacts/` (310 MB blob)

We didn't modify beellama source, but we ARE committing to keep using and updating it. As the toolchain matures, we want:
- **Stable image to consume** that doesn't get surprised by upstream beellama breaks
- **One place** to apply patches if we ever need them (DFlash tuning, bug fixes)
- **CI/CD** that auto-publishes images when we promote new beellama versions
- **TurboQuant_Benchmark stays research-pure** — consumes the image, doesn't build it

Target architecture (post-migration):

```
github.com/<you>/beellama.cpp           ← FORK
   branches: main (tracks upstream), stable (pinned to validated SHA)
   added: docker/, .github/workflows/
                ↓ GHA auto-publishes
ghcr.io/<you>/beellama-server           ← IMAGE REGISTRY
   tags: :stable, :b9459, :b9459-cuda13.2, :latest
                ↑ consumed by
github.com/<you>/TurboQuant_Benchmark   ← RESEARCH REPO
   docs/, bench scripts, deployment guides (no Dockerfile)
```

---

## Sessions overview

| Session | Goal | Duration | Who runs it |
|---|---|---|---|
| **A** | Fork beellama, re-wire local remote, create stable branch | ~10 min | Tommy |
| **B** | Add Docker recipe + GHA workflow to fork | ~15 min Claude prep, ~5 min Tommy push | Both |
| **C** | First automated build to GHCR, verify pull works | ~15 min (mostly wait) | Tommy monitors, Claude diagnoses if it breaks |
| **D** | Clean TurboQuant_Benchmark, push to its own GitHub repo | ~20 min | Both |
| **E** | Ongoing: how to test and promote new upstream beellama versions | n/a (process) | Tommy as needed |

Total active time: ~60 min spread across the sessions. Designed so each session is atomic — if you stop after Session A you've made progress without leaving anything broken.

---

## SESSION A — Fork beellama + re-wire local remote

### Goal
Create your own GitHub fork of `Anbeeld/beellama.cpp`, point your local checkout at it, and create a `stable` branch pinned to the SHA we tested this weekend.

### Prerequisites
- GitHub account with SSH/PAT auth configured for git push
- Local beellama checkout clean (no uncommitted changes — verified: `## main...origin/main`)

### Steps

**A.1.** In GitHub web UI: visit https://github.com/Anbeeld/beellama.cpp → click **Fork** → keep name `beellama.cpp`, owner `<your-username>`.

**A.2.** Re-wire your local clone to use the fork as primary and upstream as the source:

```bash
cd /mnt/i/dev/LLM/beellama.cpp

# Rename the existing remote (currently named "origin", points at Anbeeld)
git remote rename origin upstream

# Add your fork as the new origin
git remote add origin https://github.com/<your-username>/beellama.cpp.git

# Verify
git remote -v
# Expected output:
#   origin   https://github.com/<your-username>/beellama.cpp.git (fetch)
#   origin   https://github.com/<your-username>/beellama.cpp.git (push)
#   upstream https://github.com/Anbeeld/beellama.cpp.git         (fetch)
#   upstream https://github.com/Anbeeld/beellama.cpp.git         (push)
```

**A.3.** Push your local main to your fork:

```bash
git push -u origin main
```

(Should be a no-op since your local main already matches the upstream main you forked from. The `-u` sets your fork's main as the default upstream for your local main.)

**A.4.** Create the `stable` branch pinned to the exact SHA we tested:

```bash
# Verify the SHA we ran this weekend's benchmarks against
git log -1 --format="%H %s" 07ac3cec6
# Expected: "07ac3cec6 ..." (full SHA starts with 07ac3cec6)

# Create stable branch at that SHA
git checkout -b stable 07ac3cec6

# Push it to origin (your fork)
git push -u origin stable
```

**A.5.** Return to main for safety:

```bash
git checkout main
```

### Verification
- GitHub web UI: visit `https://github.com/<you>/beellama.cpp` → you see your fork with main and stable branches
- `git branch -a` locally shows both branches with `origin/` tracking
- `stable` branch HEAD = `07ac3cec6` (the validated commit)

### What could go wrong
- **Auth failure on push:** PAT missing or wrong scope. Fix: generate new PAT with `repo` scope, `git remote set-url origin https://<token>@github.com/<you>/beellama.cpp.git`
- **`stable` branch creation says "SHA not in repo":** the commit hash typo. Use `git log --grep="07ac3cec"` to find the right one
- **Pushed wrong thing to fork:** delete the fork on GitHub and start over (no production users yet)

### Done when
You can `git push origin stable` cleanly and the fork shows both branches in GitHub.

---

## SESSION B — Add Docker recipe + GHA workflow to fork

### Goal
Move the Docker stack (Dockerfile, entrypoint, etc.) from TurboQuant_Benchmark INTO your beellama fork, plus add a GitHub Actions workflow that auto-builds and publishes the image.

### Prerequisites
- Session A complete (fork exists, stable branch exists)
- Working dir: your fork on the `stable` branch

### Steps

**B.1.** Claude prepares a staging directory in TurboQuant_Benchmark containing exactly the files that should go into the fork. Path: `docs/migration/fork-staging/`. Claude generates:

| File | Source | Notes |
|---|---|---|
| `docker/Dockerfile` | Adapted from current `docs/deployment/docker/Dockerfile.source` | CI-friendly path: builds from local source tree (the fork itself), no submodule clone needed since this IS the source repo |
| `docker/entrypoint.sh` | Copied from current `entrypoint.sh` | unchanged |
| `docker/docker-compose.example.yml` | Adapted from current `docker-compose.yml` | Reference example for users who want compose; image is the upstream artifact |
| `docker/.env.example` | Copied from current `.env.example` | unchanged |
| `docker/SBOM.md` | Copied from current `SBOM.md` | Reference matters since image is now publicly distributed |
| `docker/README.md` | NEW — "How to build and run this image"; consolidates README + VALIDATION findings | Audience: image consumers |
| `.github/workflows/docker-publish.yml` | NEW — auto-build on push to stable/main, publish to GHCR | See B.3 |
| `.dockerignore` | Tightened version | Keeps build context small |

**B.2.** Tommy reviews staging dir, then copies into the fork:

```bash
# In TurboQuant_Benchmark: review what Claude prepared
ls docs/migration/fork-staging/

# Copy into the fork (stable branch)
cd /mnt/i/dev/LLM/beellama.cpp
git checkout stable
cp -r /mnt/i/dev/LLM/TurboQuant_Benchmark/docs/migration/fork-staging/docker .
cp -r /mnt/i/dev/LLM/TurboQuant_Benchmark/docs/migration/fork-staging/.github .
cp /mnt/i/dev/LLM/TurboQuant_Benchmark/docs/migration/fork-staging/.dockerignore .

# Review what's about to be committed
git status
git diff --stat
```

**B.3.** GHA workflow design (preview of what Claude will write):

```yaml
# .github/workflows/docker-publish.yml
name: Build and publish beellama-server image to GHCR

on:
  push:
    branches: [stable]                # auto on stable updates
  workflow_dispatch:                  # manual trigger
    inputs:
      tag_suffix:
        description: 'Extra tag (e.g. b9460)'
        required: false

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository_owner }}/beellama-server

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Compute tags
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=stable,enable={{is_default_branch}}
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha,format=short
            type=raw,value=${{ inputs.tag_suffix }},enable=${{ inputs.tag_suffix != '' }}

      - name: Build & push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          build-args: |
            CUDA_DOCKER_ARCH=75;80;86;89;120
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**B.4.** Commit and push:

```bash
git add docker/ .github/ .dockerignore
git commit -m "Add Docker recipe + GHA workflow for ghcr.io/<you>/beellama-server

- Multi-stage build (CUDA 13.2.1 dev → runtime, Ubuntu 24.04)
- Pinned defaults: turbo4 KV, Qwen3.5-9B compatible
- GHA publishes on push to stable, manual trigger available
- See docker/README.md for build + run instructions
- SBOM at docker/SBOM.md"

git push origin stable
```

### Verification
- `git log -1 --stat` on stable shows the docker/ + .github/ additions
- Push succeeds, GitHub web UI shows the new files
- (Action will start running automatically — that's Session C)

### What could go wrong
- **Workflow file in wrong path:** must be exactly `.github/workflows/*.yml`. If you put it in `docker/.github/...` it'll be ignored
- **Conflicts when syncing main from upstream later:** because we added new dirs at the repo root. Will need to be rebased on top of upstream changes. Mitigation: keep docker/ and .github/ ONLY on `stable` branch, not on `main`. Upstream syncs go to main, then merge into stable.

### Done when
- Files are pushed to stable
- GHA shows a run starting (visible in repo's Actions tab)
- Local `stable` branch contains everything needed for the CI build

---

## SESSION C — First automated build to GHCR

### Goal
GHA builds the image, pushes to GHCR, verify a `docker pull` works from a clean state.

### Prerequisites
- Session B complete (workflow file pushed)

### Steps

**C.1.** Watch the GHA run:

```bash
# Or via web UI: https://github.com/<you>/beellama.cpp/actions
gh run watch
```

Expected duration: **10-20 min** for first build (cold caches, compiling beellama from source for 5 CUDA archs). Subsequent runs with cache: ~5-10 min.

**C.2.** When GHA finishes successfully, verify the image is published:

```bash
# Check GHCR
docker pull ghcr.io/<you>/beellama-server:stable
docker image inspect ghcr.io/<you>/beellama-server:stable --format '{{json .Config.Labels}}' | jq

# Should show our LABEL metadata (beellama.binary.version, etc.)
```

**C.3.** Make the package public (one-time, optional but recommended):

GitHub web UI:
1. Visit `https://github.com/<you>?tab=packages`
2. Click `beellama-server`
3. Package settings → Change visibility → **Public**

Now anyone can `docker pull` without auth.

**C.4.** Smoke test pull + run on a clean state:

```bash
# Remove local image to force pull from GHCR
docker rmi ghcr.io/<you>/beellama-server:stable

# Pull and run with the recommended config
docker run --rm --gpus all -p 8083:8083 \
    -v /home/tommy/models:/models \
    -e MODEL_PATH=/models/Qwen3.5-9B-Q8_0.gguf \
    -e CACHE_TYPE_K=turbo4 -e CACHE_TYPE_V=turbo4 \
    -e CONTEXT_SIZE=65536 \
    ghcr.io/<you>/beellama-server:stable

# In another terminal:
curl -N -X POST http://localhost:8083/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"hi"}],
         "max_tokens":40,"stream":true,
         "chat_template_kwargs":{"enable_thinking":false}}'

# Expected: streamed response in ~1 sec
```

### Verification
- GHA run shows green ✓
- `docker pull ghcr.io/<you>/beellama-server:stable` works
- Smoke test returns a generation in ~1 sec
- `docker image inspect` shows the labels we baked into the Dockerfile

### What could go wrong
- **GHA build fails on cmake step:** most likely OOM (ubuntu-latest runner has 7 GB RAM, may need to drop `BUILD_JOBS` to 2). Claude will adjust workflow and re-trigger.
- **GHA build fails on CUDA arch:** the runner has no GPU but the `nvcc --gpu-architecture` flag just generates code, doesn't run it. Should work. If error, set `CUDA_DOCKER_ARCH=89` (single arch) to shrink and speed up.
- **GHCR push permission denied:** verify the workflow has `permissions: packages: write`. Check repo Settings → Actions → General → Workflow permissions = "Read and write".
- **Image too large to push (>10 GB):** unlikely (our local is 2.8 GB), but if so, the registry will reject. Trim base image or split into multi-arch (only amd64 needed anyway).
- **Smoke test fails with "manifest unknown":** make sure the package visibility is public OR you're logged into GHCR.

### Done when
- Image is pullable from a clean state
- Smoke test passes
- Tagged correctly (`:stable`, `:latest`, `:sha-XXXXXX`)

---

## SESSION D — Clean TurboQuant_Benchmark + push to its own GitHub repo

### Goal
TurboQuant_Benchmark becomes the pure research/benchmark repo. All Docker build artifacts move out. Get it pushed to GitHub so this weekend's work isn't only on local disk.

### Prerequisites
- Sessions A-C complete (image is published; we have something to point at)

### Steps

**D.1.** Delete Docker build artifacts from TurboQuant_Benchmark:

```bash
cd /mnt/i/dev/LLM/TurboQuant_Benchmark

# Remove the moved files (they now live in the beellama fork)
rm -rf docs/deployment/docker/Dockerfile
rm -rf docs/deployment/docker/Dockerfile.source
rm -rf docs/deployment/docker/entrypoint.sh
rm -rf docs/deployment/docker/docker-compose.yml
rm -rf docs/deployment/docker/.env.example
rm -rf docs/deployment/docker/.dockerignore
rm -rf docs/deployment/docker/SBOM.md          # lives in fork now
rm -rf docs/deployment/docker/prepare-artifacts.sh
rm -rf docs/deployment/docker/artifacts/       # 310 MB blob, no longer needed

# Keep VALIDATION.md but adapt it
# - it's benchmark data ("we ran the published image, got these numbers")
# - that belongs in the research repo
```

**D.2.** Adapt VALIDATION.md to reference the published image:

Claude updates `docs/deployment/docker/VALIDATION.md` to reference `ghcr.io/<you>/beellama-server:stable` instead of local builds. Move it to `docs/deployment/docker/README.md` (rename — it's now the "how to consume the image" doc) and shorten.

**D.3.** Update `docs/deployment/beellama-qwen35-9b.md`:

Change all "build the image" sections to "pull the image":

```diff
- # Build (one time, ~30 sec after CUDA base is pulled)
- cd docs/deployment/docker
- ./prepare-artifacts.sh
- docker compose build
+ # Pull the pre-built image
+ docker pull ghcr.io/<you>/beellama-server:stable
```

**D.4.** Write `.gitignore`:

```gitignore
# Large model files
*.gguf
*.safetensors

# Cache + temp
.cache/
__pycache__/
*.pyc

# Local IDE / agent state
.claude/

# Run-time outputs (regenerable from scripts)
**/run_logs/
**/raw/*.json

# Build artifacts (if any sneak in)
**/build/
**/node_modules/
**/artifacts/
```

**D.5.** Create the GitHub repo for TurboQuant_Benchmark:

GitHub web UI: New repository → name `TurboQuant_Benchmark` → public/private (your call) → DON'T initialize with README (you have one).

**D.6.** Wire up remote and push:

```bash
cd /mnt/i/dev/LLM/TurboQuant_Benchmark

# Verify clean state (post-cleanup)
git status

# Stage everything sensible
git add .gitignore
git add docs/
git add AGENTS.md CLAUDE.md
git add bench_embeddings_turbo/   # selectively; check what's worth committing
git add tqbench/                   # if you have it

# Review
git status

# Commit
git commit -m "Weekend benchmark: Qwen3.5-9B + TurboQuant + DFlash investigation

- Production config validated: beellama + Q8_0 + turbo4 KV @ 64K (71 tok/s, 9.7 GB)
- Full TurboQuant variant comparison (turbo3/turbo3_tcq/turbo4)
- vLLM ecosystem research: TQ+DFlash combo unavailable in any current vLLM
- DFlash on Qwen3.5-9B inconclusive on RTX 4090 24GB; needs vLLM/SGLang + 40GB GPU
- F16 KV memory math corrected: Qwen3.5 is hybrid attention+SSM (8/32 attn layers)
- Docker image moved to separate fork: ghcr.io/<you>/beellama-server
- Watchlist of upstream PRs/issues to monitor

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"

# Wire up remote
git remote add origin https://github.com/<you>/TurboQuant_Benchmark.git
git branch -M main
git push -u origin main
```

### Verification
- `git status` clean
- `git log -1` shows your weekend work as one commit
- GitHub web UI shows the repo with all the reports, watchlist, deployment guides
- Following the deployment guide from a fresh checkout works (pulls image, runs, generates)

### What could go wrong
- **Accidentally commit a .gguf:** `.gitignore` should prevent it but double-check with `git status` before commit. If one slips in, `git rm --cached path/to/file.gguf` and re-commit.
- **Large commit (slow push):** if commit > 100 MB, GitHub may reject. Use `git lfs` for any large files, or trim.
- **Push rejected (non-fast-forward):** repo was somehow initialized with commits. `git pull --rebase origin main` then re-push.

### Done when
- Repo is on GitHub
- Deployment guide works end-to-end from a clean machine (pull image → run → generate)
- VALIDATION.md is updated with the published-image numbers

---

## SESSION E — Ongoing: testing and promoting new upstream beellama

### When this happens
- You read about an interesting beellama release on GitHub
- An upstream PR fixes something we care about (e.g., DFlash accept rate, new TurboQuant variant)
- Periodically (monthly?) to stay current

### Procedure

**E.1.** Sync your fork's main from upstream:

```bash
cd /mnt/i/dev/LLM/beellama.cpp
git checkout main
git fetch upstream
git merge upstream/main          # or: git rebase upstream/main, your preference
git push origin main
```

**E.2.** Build locally + smoke test:

```bash
# Rebuild the binary locally (your normal build)
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server

# Verify version
build/bin/llama-server --version
# Should show: version: bXXXX (new SHA)

# Quick functional test (use your local models)
build/bin/llama-server \
    -m /home/tommy/models/Qwen3.5-9B-Q8_0.gguf \
    -c 4096 -ngl 99 -fa on \
    -ctk turbo4 -ctv turbo4 \
    --reasoning-format none &

# Hit it with a smoke test, check tok/s is in expected range
# Kill server
```

**E.3.** (Optional but recommended) re-run the 10-NIAH validation through the OLD docker image (which is built from stable) AND build a new docker image locally from new SHA, compare. If accept rate / quality / speed match, the new version is "safe".

**E.4.** Promote to stable:

```bash
cd /mnt/i/dev/LLM/beellama.cpp
git checkout stable
git merge main                   # bring stable up to new SHA

# Optional: tag the release
git tag -a bXXXX -m "Promoted upstream SHA <hash> to stable after validation"

git push origin stable --tags
```

GHA fires automatically, builds new image, publishes to:
- `ghcr.io/<you>/beellama-server:stable` (rolling)
- `ghcr.io/<you>/beellama-server:bXXXX` (immutable)
- `ghcr.io/<you>/beellama-server:sha-XXXXXX` (immutable, full SHA tag)
- `ghcr.io/<you>/beellama-server:latest` (rolling, alias for stable)

**E.5.** Update consumers:

In TurboQuant_Benchmark, if any docs explicitly reference a tag like `:b9459`, bump them. The `:stable` and `:latest` pointers auto-update.

Update `docs/watchlist.md` with the new SHA + notes on what changed.

**E.6.** If new version REGRESSES:

```bash
cd /mnt/i/dev/LLM/beellama.cpp
git checkout stable
git revert <merge-commit-sha>    # un-do the merge from main
git push origin stable
```

GHA republishes the previous image as `:stable`. Specific old tags (`:b9459`) still work.

### Cadence
- **Monthly check** of upstream: recommended via the existing `docs/watchlist.md` + suggested `/schedule monthly` cron
- **Ad-hoc** if you spot something on GitHub that matters

### Done when
You're consistently using a known-good `:stable` image and have a process to advance it when needed without breaking anything in flight.

---

## Risk register (whole migration)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Fork creates noise in upstream's network/forks view | High (will happen) | Cosmetic | Accept it; that's what forks are |
| Forgetting to sync upstream → drift | Medium | Slow drift | `docs/watchlist.md` already addresses this |
| GHA hits free-tier minute limit | Low (2000 free/mo for public, 3000 for personal) | Builds queue | Cron less frequently; use cache aggressively (already configured) |
| Docker image too large to push | Low (was 2.8 GB local) | Build fails | Trim CUDA archs to just 89 if needed |
| GHCR namespace conflict | Very low | Naming change | Use a different image name (e.g., `beellama-cuda`) |
| Future upstream Anbeeld releases include their own `.github/workflows/docker-publish.yml` | Medium-low | Merge conflict on main → stable | Keep docker/ + .github/ ONLY on stable branch; main mirrors upstream cleanly |
| Loss of local-build path (someone wants to build without Docker) | Low | Documentation gap | Keep build instructions in fork's docker/README.md; still works |
| beellama license changes | Very low | Re-distribution issue | Currently MIT — check at each sync |

---

## What we are NOT doing

To keep this migration small:

- **Not multi-arch (arm64/Jetson)** — separate concern, not needed for our use case
- **Not building images on every push to `main`** — only on `stable`. Main is for upstream sync, not production
- **Not setting up Docker Hub mirror** — GHCR is sufficient; can add later if discoverability matters
- **Not adding image signing (cosign)** — nice-to-have for supply chain security, skip for v1
- **Not setting up vulnerability scanning in CI** — `docker scout` can run later, not blocking
- **Not packaging Lucebox / vLLM / sglang into separate images** — out of scope, different effort
- **Not forking models** — GGUFs stay on HuggingFace, downloaded as runtime data

---

## What "done with migration" looks like

```bash
# Anyone on any Linux + NVIDIA host can do this:
docker pull ghcr.io/<you>/beellama-server:stable

docker run --rm --gpus all -p 8083:8083 \
    -v ~/models:/models \
    -e MODEL_PATH=/models/Qwen3.5-9B-Q8_0.gguf \
    ghcr.io/<you>/beellama-server:stable

# And they get the validated 71 tok/s, 9.7 GB VRAM, 100% NIAH config out of the box.
```

Meanwhile:
- `TurboQuant_Benchmark` on GitHub holds the benchmark methodology, findings, watchlist
- `<you>/beellama.cpp` fork on GitHub holds the source, Docker build, GHA workflow
- `ghcr.io/<you>/beellama-server` holds the published image

Three repos, one image registry, one weekend's work properly stewarded.

---

## Next concrete step (when you're ready)

When you want to start, just say "let's do session A" and I'll walk you through it command by command, verify each step, and prepare Session B's staging files in parallel.
