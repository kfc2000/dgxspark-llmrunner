from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import psutil

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


@dataclass
class GpuStat:
    name: str = ""
    util_pct: float | None = None
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None


class _NvmlProbe:
    def __init__(self) -> None:
        self.nv = None
        self.handle = None
        self.name = ""
        self.ok = False
        if pynvml is None:
            return
        try:
            self.nv = pynvml.NVML()
            self.nv.nvmlInit()
            count = self.nv.nvmlDeviceGetCount()
            if count < 1:
                self.nv = None
                return
            self.handle = self.nv.nvmlDeviceGetHandleByIndex(0)
            self.name = self.nv.nvmlDeviceGetName(self.handle)
            self.ok = True
        except Exception:
            self.nv = None
            self.ok = False

    def read(self) -> GpuStat | None:
        if not self.ok or self.nv is None or self.handle is None:
            return None
        stat = GpuStat(name=self.name)
        try:
            stat.util_pct = float(
                self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu
                if hasattr(self.nv, "nvmlDeviceGetUtilizationRates")
                else self.nv.nvmlDeviceGetUtilization(self.handle, pynvml.NVML_UTILIZATION_GPU)
            )
        except Exception:
            pass
        try:
            stat.temp_c = float(self.nv.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            stat.power_w = float(self.nv.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
            stat.power_limit_w = float(self.nv.nvmlDeviceGetPowerManagementLimit(self.handle)) / 1000.0
        except Exception:
            pass
        try:
            mem = self.nv.nvmlDeviceGetMemoryInfo(self.handle)
            stat.mem_used_mb = mem.used / (1024 * 1024)
            stat.mem_total_mb = mem.total / (1024 * 1024)
        except Exception:
            pass
        return stat


_nvml = _NvmlProbe()


def gpu_available() -> bool:
    return _nvml.ok


def read_gpu() -> GpuStat | None:
    return _nvml.read()


def _read_hwmon() -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    temps: list[tuple[str, float]] = []
    powers: list[tuple[str, float]] = []
    for chip_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        chip = Path(chip_dir)
        try:
            chip_name = (chip / "name").read_text().strip()
        except OSError:
            continue
        for node in sorted(chip.glob("temp*_input")):
            try:
                val = float(node.read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
            label = chip_name
            label_file = node.with_name(node.name.replace("_input", "_label"))
            if label_file.exists():
                try:
                    label = label + ":" + label_file.read_text().strip()
                except OSError:
                    pass
            temps.append((label, val))
        for node in sorted(chip.glob("power*_input")):
            try:
                val = float(node.read_text().strip()) / 1_000_000.0
            except (OSError, ValueError):
                continue
            label = chip_name
            label_file = node.with_name(node.name.replace("_input", "_label"))
            if label_file.exists():
                try:
                    label = label + ":" + label_file.read_text().strip()
                except OSError:
                    pass
            powers.append((label, val))
    return temps, powers


def _pick(entries: list[tuple[str, float]], keywords: tuple[str, ...]) -> float | None:
    if not entries:
        return None
    lowered = [(label.lower(), val) for label, val in entries]
    for kw in keywords:
        matched = [val for label, val in lowered if kw in label]
        if matched:
            return max(matched)
    return max(val for _, val in lowered)


def read_cpu_temp_c() -> float | None:
    try:
        temps, _ = _read_hwmon()
    except OSError:
        temps = []
    if temps:
        return _pick(temps, ("cpu", "soc", "package", "central", "xp"))
    if hasattr(psutil, "sensors_temperatures"):
        try:
            mapping = psutil.sensors_temperatures()
        except Exception:
            mapping = {}
        values = [v.celsius for v in mapping.values() if v.celsius]
        if values:
            return max(values)
    return None


def read_cpu_power_w() -> float | None:
    try:
        _, powers = _read_hwmon()
    except OSError:
        powers = []
    if not powers:
        return None
    return _pick(powers, ("cpu", "package", "soc", "vdd"))


def read_system() -> dict[str, float]:
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    freq = psutil.cpu_freq()
    return {
        "cpu_pct": psutil.cpu_percent(),
        "load_1m": psutil.getloadavg()[0],
        "mem_pct": mem.percent,
        "mem_used_gb": mem.used / 2**30,
        "mem_total_gb": mem.total / 2**30,
        "disk_pct": disk.percent,
        "disk_used_gb": disk.used / 2**30,
        "disk_total_gb": disk.total / 2**30,
        "cpu_freq_mhz": freq.current or 0.0,
        "cpu_temp_c": read_cpu_temp_c() or float("nan"),
        "cpu_power_w": read_cpu_power_w() or float("nan"),
    }
