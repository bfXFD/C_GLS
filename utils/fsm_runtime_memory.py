import resource


def get_process_memory_mb() -> tuple[float, float]:
    """Return current and peak resident memory without optional dependencies."""
    values: dict[str, float] = {}
    try:
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value, _ = line.split()
                    values[key.rstrip(":")] = float(value) / 1024.0
    except OSError:
        pass
    peak_mb = values.get("VmHWM")
    if peak_mb is None:
        max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_mb = max_rss / 1024.0
    rss_mb = values.get("VmRSS", peak_mb)
    return float(rss_mb), float(max(peak_mb, rss_mb))
