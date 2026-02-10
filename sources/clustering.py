#!/usr/bin/env python3
"""Minimal fiber clustering debugger with lightweight diagnostics."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .datastructures import FiberFragment, FragmentEndpoint

INF = float("inf")


class EndpointIndex:
    def __init__(self, endpoints: Iterable[FragmentEndpoint], skip_endpoint_px: float = 0.0):
        self.eps = list(endpoints)
        self.pos = np.vstack([e.shifted_pos(skip_endpoint_px) for e in self.eps]).astype(np.float32)
        self.fid = np.array([e.fragment.fragment_id for e in self.eps], np.int32)
        self.idx_of = {ep: i for i, ep in enumerate(self.eps)}

    def query_neighbors(
        self,
        ep: FragmentEndpoint,
        d_max: float,
        exclude_same_fragment: bool = True,
        max_neighbors: int | None = None,
    ) -> List[FragmentEndpoint]:
        i0 = self.idx_of[ep]
        p = self.pos[i0]
        diff = self.pos - p
        d2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
        mask = d2 <= (d_max * d_max)
        mask[i0] = False
        if exclude_same_fragment:
            mask &= self.fid != ep.fragment.fragment_id
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        order = np.argsort(d2[idx], kind="quicksort")
        idx = idx[order]
        if max_neighbors is not None:
            idx = idx[:max_neighbors]
        return [self.eps[i] for i in idx]


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------
def line_to_line_rms_cost(
    ep1: FragmentEndpoint,
    ep2: FragmentEndpoint,
    *,
    step_px: float = 2.0,
    max_samples: int = 64,
    require_facing: bool = True,
    cos_min: float = 0.95,
    length_weight: float = 0.0,
    radius_weight: float = 0.0,
    skip_endpoint_px: float = 0.0,
) -> float:
    if hasattr(ep1, "same_fragment") and ep1.same_fragment(ep2):
        return INF

    p1 = ep1.shifted_pos(skip_endpoint_px).astype(np.float32)
    p2 = ep2.shifted_pos(skip_endpoint_px).astype(np.float32)
    v = p2 - p1
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        return INF

    d1 = ep1.dir.astype(np.float32)
    d1 /= np.linalg.norm(d1) + 1e-12
    d2 = ep2.dir.astype(np.float32)
    d2 /= np.linalg.norm(d2) + 1e-12
    if require_facing:
        u12 = v / L
        u21 = -u12
        if (float(np.dot(d1, u12)) < cos_min) or (float(np.dot(d2, u21)) < cos_min):
            return INF

    n = max(3, min(max_samples, int(L / max(step_px, 1e-6)) + 1))
    s = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    P = p1[None, :] + s * v[None, :]

    a1, a2 = p1, p2
    V1 = P - a1[None, :]
    V2 = P - a2[None, :]
    t1 = np.einsum("ij,j->i", V1, d1)
    t2 = np.einsum("ij,j->i", V2, d2)
    proj1 = a1[None, :] + t1[:, None] * d1[None, :]
    proj2 = a2[None, :] + t2[:, None] * d2[None, :]

    diff = proj1 - proj2
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    rms = float(np.sqrt(np.mean(dist * dist)))
    rms_norm = rms / L
    rad_diff = abs(ep1.radius() - ep2.radius())
    return rms_norm + length_weight * L + radius_weight * rad_diff


def stitch_and_cluster(
    fragments: Sequence[FiberFragment],
    d_max: float,
    threshold: float,
    *,
    max_neighbors: int | None = None,
    length_weight: float = 0.0,
    radius_weight: float = 0.0,
    skip_endpoint_px: float = 0.0,
) -> None:
    endpoints = [ep for f in fragments for ep in f.endpoints()]
    index = EndpointIndex(endpoints, skip_endpoint_px=skip_endpoint_px)

    proposals = {}
    for ep in endpoints:
        nbrs = index.query_neighbors(ep, d_max, True, max_neighbors)
        if not nbrs:
            continue
        best_ep, best_cost = None, float("inf")
        for nbr in nbrs:
            cost = line_to_line_rms_cost(
                ep,
                nbr,
                step_px=2.0,
                max_samples=64,
                require_facing=True,
                cos_min=0.95,
                length_weight=length_weight,
                radius_weight=radius_weight,
                skip_endpoint_px=skip_endpoint_px,
            )
            if cost < best_cost:
                best_cost = cost
                best_ep = nbr
        if best_ep is not None and best_cost < threshold:
            proposals[ep] = (best_ep, best_cost)

    used = set()
    for ep, (cand, _) in proposals.items():
        reciprocal = proposals.get(cand)
        if reciprocal and reciprocal[0] is ep:
            key = tuple(sorted((id(ep), id(cand))))
            if key in used:
                continue
            used.add(key)
            ep.connect(cand)

    uf = UnionFind()
    for fragment in fragments:
        for ep in fragment.endpoints():
            other = ep.connected_to
            if other is not None:
                uf.union(fragment.fragment_id, other.fragment.fragment_id)

    comp2pid, next_id = {}, 1
    for fragment in fragments:
        root = uf.find(fragment.fragment_id)
        if root not in comp2pid:
            comp2pid[root] = next_id
            next_id += 1
        fragment.parent_fiber_id = comp2pid[root]


# ---------------------------------------------------------------------------
# Diagnostics + visualization helpers
# ---------------------------------------------------------------------------
def endpoint_label(ep: FragmentEndpoint) -> str:
    return f"{ep.fragment.fragment_id}{ep.end}"


def load_fragments_from_txt(path: str) -> List[FiberFragment]:
    seg_pts = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            sid = int(float(parts[0]))
            pid = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            seg_pts[sid].append((pid, x, y))

    fragments = []
    for sid, pts in seg_pts.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        arr = np.array([[x, y] for (_, x, y) in pts_sorted], dtype=np.float32)
        fragments.append(FiberFragment(fragment_id=sid, centerline_xy=arr))
    return fragments


def compute_canvas_size(
    fragments: Sequence[FiberFragment], margin: int = 20
) -> Tuple[int, int, np.ndarray]:
    xs = np.concatenate([f.centerline_xy[:, 0] for f in fragments])
    ys = np.concatenate([f.centerline_xy[:, 1] for f in fragments])
    minx, maxx = np.floor(xs.min()), np.ceil(xs.max())
    miny, maxy = np.floor(ys.min()), np.ceil(ys.max())
    width = int(maxx - minx + 1 + 2 * margin)
    height = int(maxy - miny + 1 + 2 * margin)
    origin = np.array([minx - margin, miny - margin], dtype=np.float32)
    return width, height, origin


def draw_debug(fragments: Sequence[FiberFragment], window_name: str = "Clustering Debug"):
    width, height, origin = compute_canvas_size(fragments, margin=20)
    canvas = np.full((height, width, 3), 255, np.uint8)

    for fragment in fragments:
        pts = (fragment.centerline_xy - origin).astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 1, cv2.LINE_AA)

    for fragment in fragments:
        for ep in fragment.endpoints():
            p = (ep.pos() - origin).astype(np.int32)
            if ep.end == "a":
                cv2.circle(canvas, tuple(p), 3, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                endpoint_label(ep),
                (p[0] + 4, p[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    seen = set()
    for fragment in fragments:
        for ep in fragment.endpoints():
            other = ep.connected_to
            if other is None:
                continue
            key = tuple(sorted((id(ep), id(other))))
            if key in seen:
                continue
            seen.add(key)
            p = (ep.pos() - origin).astype(np.int32)
            q = (other.pos() - origin).astype(np.int32)
            cv2.line(canvas, tuple(p), tuple(q), (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow(window_name, canvas)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _perp_mean_to_line(anchor: np.ndarray, dir_unit: np.ndarray, pts: np.ndarray) -> float:
    n = np.array([-dir_unit[1], dir_unit[0]], dtype=np.float32)
    v = pts - anchor[None, :]
    d = np.abs(v @ n)
    return float(d.mean()) if len(d) else float("inf")


def build_pairs_table(
    endpoints: Sequence[FragmentEndpoint],
    index: EndpointIndex,
    *,
    d_max: float,
    topk: int = 8,
    n_eval: int = 5,
    cos_min: float = 0.98,
    length_weight: float = 0.0,
    radius_weight: float = 0.0,
) -> List[dict]:
    rows = []
    for ep in endpoints:
        nbrs = index.query_neighbors(ep, d_max, exclude_same_fragment=True, max_neighbors=topk)
        p1 = ep.pos()
        for nb in nbrs:
            p2 = nb.pos()
            gap = float(np.linalg.norm(p1 - p2))
            facing = ep.is_facing(nb, cos_min=cos_min)
            pd12 = (
                _perp_mean_to_line(p1, ep.dir, nb.last_n_points(n_eval))
                if facing
                else float("inf")
            )
            pd21 = (
                _perp_mean_to_line(p2, nb.dir, ep.last_n_points(n_eval))
                if facing
                else float("inf")
            )
            rms_sym = 0.5 * (pd12 + pd21) if facing else float("inf")
            cost = line_to_line_rms_cost(
                ep,
                nb,
                step_px=2.0,
                max_samples=64,
                require_facing=True,
                length_weight=length_weight,
                radius_weight=radius_weight,
            )

            rows.append(
                {
                    "ep": endpoint_label(ep),
                    "cand": endpoint_label(nb),
                    "gap_px": round(gap, 3),
                    "facing": int(bool(facing)),
                    "pd12_px": None if not facing else round(pd12, 3),
                    "pd21_px": None if not facing else round(pd21, 3),
                    "rms_sym_px": None if not facing else round(rms_sym, 3),
                    "cost": None if not np.isfinite(cost) else round(float(cost), 4),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal debugger for fiber clustering")
    parser.add_argument("txt", help="Path to centerlines .txt")
    parser.add_argument("--dmax", type=float, default=150.0, help="Neighbor search radius (pixels)")
    parser.add_argument("--thr", type=float, default=10, help="Connection cost threshold")
    parser.add_argument(
        "--maxn", type=int, default=None, help="Max neighbors to evaluate per endpoint"
    )
    parser.add_argument(
        "--lenw",
        type=float,
        default=0.1,
        help="Weight for gap length penalty component inside the cost function",
    )
    parser.add_argument(
        "--radw",
        type=float,
        default=0.0,
        help="Weight for endpoint radius mismatch penalty component inside the cost function",
    )
    args = parser.parse_args(argv)

    fragments = load_fragments_from_txt(args.txt)
    if not fragments:
        print("No fragments parsed. Check file format.")
        return 1

    stitch_and_cluster(
        fragments,
        d_max=args.dmax,
        threshold=args.thr,
        max_neighbors=args.maxn,
        length_weight=args.lenw,
        radius_weight=args.radw,
    )

    endpoints = [ep for f in fragments for ep in f.endpoints()]
    index = EndpointIndex(endpoints)
    rows = build_pairs_table(
        endpoints,
        index,
        d_max=args.dmax,
        topk=8,
        n_eval=5,
        cos_min=0.98,
        length_weight=args.lenw,
        radius_weight=args.radw,
    )

    if rows:
        rows_sorted = sorted(
            rows, key=lambda r: (float("inf") if r["cost"] is None else r["cost"])
        )
        print("\nTop candidate pairs by cost (preview):")
        for row in rows_sorted[:25]:
            print(
                f'{row["ep"]:>5} -> {row["cand"]:<5}  '
                f'cost={row["cost"]}  gap={row["gap_px"]}  '
                f'facing={row["facing"]}  rms={row["rms_sym_px"]}'
            )
        try:
            import csv

            with open("pairs_debug.csv", "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(rows_sorted[0].keys()))
                writer.writeheader()
                writer.writerows(rows_sorted)
            print('Saved diagnostics to "pairs_debug.csv"')
        except Exception as exc:
            print(f"Could not save CSV: {exc}")
    else:
        print("No candidate pairs found; skipping CSV export.")

    draw_debug(fragments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
