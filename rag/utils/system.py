import os

from loguru import logger


def effective_cpu_count() -> int:
    """Return the number of CPUs available to this process, respecting cgroup limits.

    Detection priority:
      1. cgroup v2  (``/sys/fs/cgroup/cpu.max``)  — modern Kubernetes
      2. cgroup v1  (``cpu.cfs_quota_us``)         — older clusters
      3. ``os.sched_getaffinity``                  — cpuset pinning
      4. ``os.cpu_count``                          — bare-metal fallback
    """
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if parts[0] != "max":
            count = max(1, int(int(parts[0]) / int(parts[1])))
            logger.info("Detected {} usable CPUs (cgroup v2)", count)
            return count
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        if quota > 0:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read().strip())
            count = max(1, int(quota / period))
            logger.info("Detected {} usable CPUs (cgroup v1)", count)
            return count
    except (FileNotFoundError, OSError, ValueError):
        pass
    # cpuset
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
        logger.info("Detected {} usable CPUs (sched_getaffinity)", count)
        return count
    # fallback
    count = os.cpu_count() or 4
    logger.info("Detected {} usable CPUs (os.cpu_count fallback)", count)
    return count
