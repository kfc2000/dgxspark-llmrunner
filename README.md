# DGX Spark LLM Runner

A Python (FastAPI) + vanilla HTML/JS/CSS dashboard for monitoring and controlling local LLM servers on an
NVIDIA DGX Spark (GB10, e.g. Asus GX10). It runs on the Spark itself and gives you one web
page for:

- **Hardware telemetry** — CPU utilization / temperature / power, GPU utilization /
  temperature / power / VRAM, RAM and storage usage, with live sparkline charts.
- **LLM metrics** — decode (generation) and prefill tokens/sec, scraped from each server's
  Prometheus `/metrics` endpoint. Supports `docker_vllm`, `docker_sglang`, `sparkrun_vllm`,
  and `sparkrun_sglang`.
- **Server control** — start/stop each LLM by running its configured bash scripts, see
  container status and `docker logs` in the browser. **Single-model mode:** only one LLM can
  run at a time (the GB10 can't hold two large models); starting a model stops any loaded
  one first, and the start aborts if that stop fails. Before every start/stop the API
  re-checks all models' live statuses so it never loads two models at once (avoids OOM).

## Screens / files

| Path | Purpose |
|---|---|
| `llmrunner/api.py` | FastAPI app: JSON API + serves the single-page frontend (`uvicorn llmrunner.api:app`) |
| `web/index.html` / `web/app.js` / `web/style.css` | The whole frontend: dashboard + servers/logs views. Hardware polls every 1s; LLM server status (which loops all models) polls every 5s |
| `prototype/index.html` | Static design prototype the UI is modeled on |
| `llmrunner/config.py` | Loads/validates `llms.json` |
| `llmrunner/hw_metrics.py` | NVML (GPU) + psutil + `/sys/class/hwmon` (CPU temp/power) probes |
| `llmrunner/sampler.py` | Background thread collecting hardware samples (history ring buffer) |
| `llmrunner/llm_metrics.py` | Prometheus parser + rate computer for token throughput |
| `llmrunner/control.py` | Docker status, container logs, bash script runner |
| `install.sh` | One-shot installer for the Spark (venv + rsync + systemd) |
| `llmrunner.service` | systemd unit template |

## Configuration: `llms.json`

The app reads a JSON file (path from the `LLMRUNNER_CONFIG` env var, default `./llms.json`).
Every entry needs:

| Key | Meaning |
|---|---|
| `name` | Display name |
| `type` | `docker_vllm`, `docker_sglang`, `sparkrun_vllm`, or `sparkrun_sglang` |
| `workdir` | Directory the scripts run in |
| `start_script` | Bash command run to start the server (e.g. `./start.sh`) |
| `stop_script` | Bash command run to stop the server |
| `endpoint` | (optional, default `http://localhost:8000`) base URL of the OpenAI-compatible server; metrics are read from `<endpoint>/metrics` |
| `container_id` | (optional) the docker container name (for `docker_*` types) or the sparkrun target/recipe name (for `sparkrun_*` types). Used for status checks and the log viewer. Backward-compatible with the old `container` key |
| `sparkrun_id` | (optional) for `sparkrun_*` types, the identifier used to query `sparkrun status`/`sparkrun logs` to detect whether the sparkrun process is running. Defaults to `container_id` |
| `sparkrun_path` | (optional, top-level) absolute path to the `sparkrun` executable. Defaults to `sparkrun` on `PATH` (or the `SPARKRUN_PATH` env var). Set this if the service can't find `sparkrun` |

```json
{
  "sparkrun_path": "/usr/local/bin/sparkrun",
  "llms": [
    {
      "name": "qwen3-32b",
      "type": "docker_vllm",
      "workdir": "/home/kfc/llms/qwen3-32b",
      "start_script": "./start.sh",
      "stop_script": "./stop.sh",
      "endpoint": "http://localhost:8000",
      "container_id": "vllm-qwen3"
    },
    {
      "name": "qwen3.8-27b",
      "type": "sparkrun_vllm",
      "workdir": "/home/kfc/llms/qwen3.8-27b",
      "start_script": "./qwen3.8-27b.sh",
      "stop_script": "./stop.sh",
      "endpoint": "http://localhost:8000",
      "container_id": "qwen3.8-27b",
      "sparkrun_id": "qwen3.8-27b"
    }
  ]
}
```

For `sparkrun_*` types the `sparkrun_id` is the recipe/target name passed to
`sparkrun status`/`sparkrun logs <target>` (the log viewer queries sparkrun instead of
`docker logs`). The `sparkrun_path` top-level key tells the API where that executable lives.

### How running/starting state is detected

The API reports each LLM as `running`, `starting`, or `stopped`:

- **running** — the model name (the `name` entry in `llms.json`) is served at the
  OpenAI-compatible `/v1/models` endpoint, i.e. the server is up and the model is loaded.
- **starting** — the model isn't served yet, but its sparkrun process (`sparkrun status`,
  for `sparkrun_*` types) or docker container (`docker ps`, for `docker_*` types) is present.
  This works even if the model was started manually on the command line.
- **stopped** — no served model and no active sparkrun process / docker container.

### How throughput is computed

vLLM and SGLang expose cumulative counters (`vllm:prompt_tokens_total`,
`vllm:generation_tokens_total`, `sglang:*`, …). The app scrapes `/metrics` repeatedly and
computes per-second deltas:

- **Decode tok/s** = Δ generation tokens / Δ time
- **Prefill tok/s** = Δ prompt tokens / Δ `vllm:request_prefill_time_seconds_sum`
  (falls back to Δ prompt tokens / Δ time when the timer isn't exposed)

Metrics only move while the server is actually generating; idle servers read 0.

### Hardware metric sources

- GPU: NVML via `pynvml`. If NVML can't see the GB10, the GPU tiles show a warning instead of failing.
- CPU/RAM/storage: `psutil`.
- CPU temperature and power: hwmon sensors under `/sys/class/hwmon` (label-matched on
  `cpu`/`soc`/`package`, best match shown). If a sensor isn't exposed by the kernel, the tile reads `n/a`.

## Install on the Spark (Linux, aarch64)

Prerequisites: Python 3.10+, git, rsync, and your user in the `docker` group:

```bash
sudo usermod -aG docker $USER   # then re-login
```

Then:

```bash
git clone <your-fork> && cd dgxspark-llmrunner
./install.sh                   # optional arg: install dir (default ~/llmrunner)
```

The installer creates a venv, installs dependencies, copies the app to `~/llmrunner`,
writes the systemd unit `/etc/systemd/system/llmrunner.service` (via `sudo tee`) and
enables it. The dashboard is then served on all interfaces at port **8501**:

```
http://<spark-ip>:8501        # or via Tailscale
```

Re-run `./install.sh` after pulling changes — it is idempotent and restarts the service.
Logs: `journalctl -u llmrunner -f`. Disable with `sudo systemctl disable --now llmrunner`.

### Which `llms.json` does the service use?

The generated unit hardcodes `Environment=LLMRUNNER_CONFIG=<install-dir>/llms.json`, so
after installing you configure LLMs by editing **`~/llmrunner/llms.json`** on the Spark
(`sudo systemctl restart llmrunner` afterwards). `install.sh` seeds this file from
the repo's `llms.json` on first install and leaves it alone on re-installs, so your edits
survive updates. To use a config file elsewhere, edit
`Environment=LLMRUNNER_CONFIG=...` in the unit (`sudo systemctl edit llmrunner`).

### Manual run (no systemd)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
LLMRUNNER_CONFIG=/path/to/llms.json .venv/bin/uvicorn llmrunner.api:app --port 8501
```

### Development (any machine)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest ruff
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check llmrunner tests
node --check web/app.js
```

Hardware tiles degrade to `n/a` off-Spark; the Prometheus parser and config loader are unit-tested.

## Security note

The dashboard has no authentication and Start/Stop buttons execute bash on the box. Keep it
loopback-only or behind Tailscale/a reverse proxy — do not expose port 8501 to an untrusted network.

Optional: set `LLMRUNNER_TOKEN` in the environment (e.g. in `Environment=` of the unit) to
require `Authorization: Bearer <token>` on the start/stop endpoints.
