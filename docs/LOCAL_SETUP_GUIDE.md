# Local Setup Guide (for new contributors)

This guide walks you through cloning the **EthoInsight** project and running it locally for the first time.
EthoInsight is an AI analysis assistant for behavioral researchers: you upload EthoVision XT tracking exports, and the agent runs statistical analysis, expert interpretation, and generates an APA-style report.

> **Audience:** a colleague who just got repo access and wants to test the project locally.
> **Platform:** Linux / macOS (the steps below assume a Unix shell). Windows users should use Git Bash or WSL.

---

## 1. Prerequisites

Before you start, make sure the following tools are installed on your machine:

| Tool | Required version | Why | Install |
|------|------------------|-----|---------|
| **Git** | any recent | clone the repo, sync upstream | https://git-scm.com/ |
| **Python** | **3.12+** | agent backend | https://www.python.org/ |
| **[uv](https://docs.astral.sh/uv/)** | latest | Python dependency manager (`uv sync` installs the venv + `ethoinsight` lib) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Node.js** | **22+** | frontend | https://nodejs.org/ |
| **pnpm** | latest | frontend package manager (Corepack is fine) | `corepack enable pnpm` or https://pnpm.io/ |
| **nginx** | any recent | local reverse proxy (frontend + gateway on one port) | apt/brew package |

> Docker is **optional**. The fastest path to try the project is the **local dev** flow below; Docker is only needed if you want the container-based sandbox or a production-like stack.

You can verify the toolchain in one go:

```bash
git --version
python3 --version        # must be >= 3.12
uv --version
node --version           # must be >= 22
pnpm -v
nginx -v
```

---

## 2. Clone the repository

```bash
git clone https://gitlab.com/noldus-core-team/china-devs/etho-insight.git
cd etho-insight
```

If GitLab asks for credentials, use an **access token** (Profile → Preferences → Access Tokens → create one with `read_repository` scope) as the password — your normal login password will not work over HTTPS.

The repo layout you care about:

```
etho-insight/
├── packages/
│   ├── agent/            # the runnable app (DeerFlow-based agent harness)
│   │   ├── backend/      # Python backend (LangGraph + Gateway)
│   │   ├── frontend/     # Next.js frontend
│   │   ├── config.yaml   # ← you will create/edit this (models, tools, sandbox)
│   │   ├── extensions_config.json
│   │   └── Makefile      # ← all the commands you need live here
│   └── ethoinsight/      # behavioral-data analysis Python library
├── docs/
├── golden-cases/
└── scripts/
```

> **Tip:** `packages/agent/` is where you'll spend almost all your time. Everything below runs from that directory.

---

## 3. Configure (create `config.yaml` and set your model key)

From the `packages/agent/` directory:

```bash
cd packages/agent
make config          # generates config.yaml + extensions_config.json from the examples
make check           # verifies the toolchain above is present
make install         # runs `uv sync` (backend + ethoinsight lib) and `pnpm install` (frontend)
```

`make install` creates `packages/agent/backend/.venv/` and installs everything, including the `ethoinsight` analysis library (it's wired in as a workspace dependency, so you do **not** install it separately).

### Set your model API key

The project's lead agent runs on **DeepSeek** (via Alibaba DashScope's OpenAI-compatible endpoint). Open `config.yaml` and find the two model entries (`deepseek-v4-pro` and `deepseek-v4-pro-summary`). They reference the key by environment variable — export it in your shell **before** starting the app:

```bash
export DASHSCOPE_API_KEY="sk-..."        # use the real key value from your key provider
```

> Replace `DASHSCOPE_API_KEY` with the exact variable name used in your `config.yaml`'s `openai_api_key:` line. If you'd rather use a different provider, edit the `models:` block in `config.yaml` following the commented examples (Volcengine Doubao, OpenAI, Ollama, etc.) — at least one model entry must be present for the agent to start.

Put the export in your `~/.bashrc` / `~/.zshrc` so it persists across terminals, but **never commit the key**.

---

## 4. Run the app

Still in `packages/agent/`:

```bash
make dev
```

This starts **all three services** (Gateway-embedded agent runtime + Frontend + Nginx) with hot-reloading. The first run is slow (frontend build + model warmup).

When it's ready, open:

```
http://localhost:2026
```

To stop everything:

```bash
make stop
```

Other useful commands:

| Command | What it does |
|---------|--------------|
| `make dev-daemon` | same as `make dev` but in the background |
| `make stop` | stop all services |
| `make check` | re-verify prerequisites |
| `make help` | list all available make targets |

### Backend-only development (optional)

If you only want to work on the Python side:

```bash
cd packages/agent/backend
source .venv/bin/activate
make dev          # LangGraph server only (port 2024)
make gateway      # Gateway API only (port 8001)
make test         # run backend tests
make lint         # ruff check
```

---

## 5. Test it with your own data

There is **no bundled sample data in the repo** — upload your own EthoVision XT export.

1. Make sure `make dev` is running and you're on http://localhost:2026.
2. Start a new chat / thread.
3. Upload your EthoVision XT export file(s). Supported formats: **TXT, CSV, XLSX, XLS**. Use the actual files exported from EthoVision XT (they typically contain tracking data plus a `Treatment` / `Animal ID` group column).
4. The agent will inspect the file, recognize the paradigm, and may **ask you clarifying questions** (e.g. which column is which analysis zone, how groups map). Answer these — the analysis only proceeds once they're resolved.
5. When it finishes you'll get statistical results, publication-grade charts, and an APA-style report.

### What's currently supported

v0.1 supports **six mammalian anxiety/depression paradigms**:

- **EPM** (Elevated Plus Maze)
- **OFT** (Open Field Test)
- **LDB** (Light–Dark Box)
- **FST** (Forced Swim Test)
- **Zero Maze**
- **TST** (Tail Suspension Test)

For any other paradigm (fish, learning & memory mazes, social/novel object, etc.) the agent will tell you it's not implemented yet and ask for confirmation rather than guessing.

---

## 6. Common first-run problems

- **`make dev` hangs on "Waiting for Gateway on port 8001"** — usually a missing/invalid API key or a `config.yaml` problem. Check the gateway log (printed inline) for the bottom error line, confirm your API key is exported, and that `config.yaml` has at least one model entry.
- **Port 2026 already in use** — run `make stop`, or kill whatever holds the port, then `make dev` again.
- **`uv: command not found` / `pnpm: command not found`** — re-run `make check`; it tells you exactly which tool is missing.
- **Frontend build fails on first run** — re-run `make install` to make sure `pnpm install` completed; then `make dev`.

---

## 7. Where to read more

Once it's running and you want to go deeper:

- **`CLAUDE.md`** (repo root) — the full project context: architecture, conventions, the agent analysis pipeline, and detailed gotchas.
- **`packages/agent/README.md`** — DeerFlow upstream docs (the harness this project is built on).
- **`packages/agent/Install.md`** — the coding-agent bootstrap spec (handy if you automate setup with an AI agent).
- **`docs/milestone/README.md`** — project roadmap / current status of each feature track.

---

Happy testing! If anything in setup blocks you, the `make check` output + the bottom of the gateway log are the two things worth pasting when asking for help.
