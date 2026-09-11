# DGX Spark LLM Runner

A Python + Streamlit dashboard for monitoring and controlling local LLM servers on an
NVIDIA DGX Spark (GB10, e.g. Asus GX10). It runs on the Spark itself and gives you one web
page for:

- **Hardware telemetry** — CPU utilization / temperature / power, GPU utilization /
  temperature / power / VRAM, RAM and storage usage, with live sparkline charts.
- **LLM metrics** — decode (generation) and prefill tokens/sec, scraped from each server's
  Prometheus `/metrics` endpoint. Supports `docker_vllm` and `docker_sglang`.
- **Server control** — start/stop each LLM by running its configured bash scripts, see
  container status and `docker logs` in the browser.

## Screens / files

| Path | Purpose |
|---|---|
| `dashboard/main.py` | Streamlit entry point: hardware + throughput dashboard (auto-refreshing) |
| `dashboard/pages/01_LLM_servers.py` | Start/Stop buttons and log viewer per LLM |
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
| `type` | `docker_vllm` or `docker_sglang` |
| `workdir` | Directory the scripts run in |
| `start_script` | Bash command run to start the server (e.g. `./start.sh`) |
| `stop_script` | Bash command run to stop the server |
| `endpoint` | (optional, default `http://localhost:8000`) base URL of the OpenAI-compatible server; metrics are read from `<endpoint>/metrics` |
| `container` | (optional) docker container name — enables status checks and the log viewer |

```json
{
  "llms": [
    {
      "name": "qwen3-32b",
      "type": "docker_vllm",
      "workdir": "/home/kfc/llms/qwen3-32b",
      "start_script": "./start.sh",
      "stop_script": "./stop.sh",
      "endpoint": "http://localhost:8000",
      "container": "vllm-qwen3"
    }
  ]
}
```

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
LLMRUNNER_CONFIG=/path/to/llms.json .venv/bin/streamlit run dashboard/main.py
```

### Development (any machine)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest ruff
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check llmrunner dashboard tests
```

Hardware tiles degrade to `n/a` off-Spark; the Prometheus parser and config loader are unit-tested.

## Security note

The dashboard has no authentication and Start/Stop buttons execute bash on the box. Keep it
loopback-only or behind Tailscale/a reverse proxy — do not expose port 8501 to an untrusted network.
