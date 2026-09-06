"""Extract completed ignition lineages without truncating large events."""
import numpy as np


def completed_fire_sizes(totals, ignition_times, active_ids, start_frame, perimeters=None):
    """Return sizes/perimeters for events ignited after the transient.

    Active events are right-censored. No cap based on grid width is applied:
    fire size counts burned cells, rather than a linear distance.
    """
    active = set(active_ids)
    eligible = [fid for fid in totals
                if ignition_times.get(fid, -1) > start_frame and fid not in active]
    sizes = np.asarray([totals[fid] for fid in eligible], dtype=np.int64)
    borders = np.asarray([perimeters.get(fid, 0) for fid in eligible], dtype=np.int64) if perimeters is not None else np.array([], dtype=np.int64)
    return sizes, borders
