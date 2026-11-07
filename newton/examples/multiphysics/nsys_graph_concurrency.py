"""Achieved kernel concurrency inside replayed CUDA graphs, from an nsys
profile taken with `-t cuda --cuda-graph-trace=node`.

    nsys export --type sqlite -o run.sqlite run.nsys-rep
    python nsys_graph_concurrency.py run.sqlite [label]

Only kernels that ran as graph nodes are counted (graphId set), so eager
warm-up and capture-time work are excluded in both arms. Reported:
  sum      total kernel time (sum of durations)
  union    time with >= 1 kernel running
  span     first start to last end of graph kernels
  concurrency = sum / union   (1.0 = fully serialised)
  overlap_frac = fraction of `union` with >= 2 kernels running
  graph_ids    distinct graphIds seen (one per instantiated graph)
"""
import sqlite3
import sys
from collections import Counter


def main(path, label=""):
    db = sqlite3.connect(path)
    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    gcol = "graphId" if "graphId" in cols else None
    assert gcol, "no graphId column: profile without --cuda-graph-trace=node?"
    rows = db.execute(f"SELECT start, end, {gcol}, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE {gcol} IS NOT NULL ORDER BY start").fetchall()
    if not rows:
        print(f"CONCURRENCY {label} no graph kernels")
        return
    total = sum(e - s for s, e, _, _ in rows)
    span = rows[-1][1] - rows[0][0]
    # sweep
    events = []
    for s, e, _, _ in rows:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    depth, last, union, overlap, maxdepth = 0, events[0][0], 0, 0, 0
    for t, d in events:
        if depth >= 1:
            union += t - last
        if depth >= 2:
            overlap += t - last
        depth += d
        maxdepth = max(maxdepth, depth)
        last = t
    per_graph = Counter(g for _, _, g, _ in rows)
    streams = len({st for _, _, _, st in rows})
    print(f"CONCURRENCY {label} graph_ids={len(per_graph)} kernels={len(rows)} streams={streams} "
          f"sum_ms={total/1e6:.2f} union_ms={union/1e6:.2f} span_ms={span/1e6:.2f} "
          f"concurrency={total/union:.2f} overlap_frac={overlap/union:.2f} max_depth={maxdepth} busy_frac={union/span:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
