"""Candidate point sources: a redshift and a source-plane position, drawn so that multiple images are common.

Candidate ``index`` is drawn from its own generator, ``numpy.random.default_rng([seed, index])``, so the
candidate list is fixed by the seed whatever the number of worker processes or the order of evaluation.

Seeding kinds (``cfg["seed_kinds"]`` gives their weights):

``negative``  a random mesh node of the image plane with det(A) < 0 (between the critical curves, where an
              image is always one of at least three), jittered uniformly within half a mesh step in x and y,
              is mapped to the source plane (the jittered point is the seed and one of the images).
``critical``  the same with |det(A)| below a threshold drawn from ``critical_det``: sources close to a caustic,
              highly magnified images, folds and cusps.
``source``    uniform in the source plane over the bounding box of the det(A) < 0 region mapped back, which
              contains every multiply imaged source.  The only kind whose sources are uniform in the source
              plane (restricted to the box).

``subhalo``   a mesh node with det(A) < 0 close to a massive satellite (the mask is ``SourceSampler.subhalo_mask``,
              set from ``tnglens.satellites.Satellites.seed_mask``; without it the kind finds no seed).

Image-plane seeding weights a source by the image-plane area of its negative-parity (or near-critical) images,
so the kept systems are not an area-limited source population; the catalog records which kind made each one.
New kinds can be added to ``KINDS``.

``draw`` always returns a ``Candidate``; its ``outcome`` is "ok", "foreground" (redshift at or in front of the
lens, no lensing) or "no_seed" (the kind found no seed node at that redshift), and ``beta`` is None unless "ok".
``check_redshifts`` tells how much of a redshift distribution lies at or in front of the lens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULTS = dict(
    # redshift distribution: bins with weights, uniform within a bin (tngc014 mock); or "values" (+ weights)
    redshift=dict(edges=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0, 8.0],
                  weights=[0.10, 0.17, 0.20, 0.17, 0.12, 0.12, 0.08, 0.04]),
    seed_kinds=dict(negative=0.45, critical=0.55, source=0.0, subhalo=0.0),
    seed_radius=[5.0, 95.0],                 # image-plane seeds between these radii [arcsec]
    critical_det=[0.05, 0.02, 0.008, 0.003],  # |det A| thresholds of the critical kind (one drawn per source)
)


@dataclass
class Candidate:
    index: int
    kind: str
    z: float
    s: float
    beta: np.ndarray | None     # source position [arcsec]; None unless outcome == "ok"
    seed: np.ndarray | None     # image-plane seed position, if any
    outcome: str = "ok"         # "ok", "foreground" or "no_seed"


def draw_redshift(rng, spec):
    """One source redshift from ``spec``: {edges, weights} (uniform in a bin) or {values[, weights]}."""
    if "values" in spec and spec["values"] is not None:
        vals = np.asarray(spec["values"], float)
        w = np.ones(len(vals)) if spec.get("weights") is None else np.asarray(spec["weights"], float)
        return float(vals[rng.choice(len(vals), p=w / w.sum())])
    edges, w = np.asarray(spec["edges"], float), np.asarray(spec["weights"], float)
    if len(w) != len(edges) - 1:
        raise ValueError("redshift weights need one entry per bin (len(edges) - 1)")
    b = rng.choice(len(w), p=w / w.sum())
    return float(np.round(rng.uniform(edges[b], edges[b + 1]), 4))


def foreground_fraction(spec, zl: float) -> float:
    """Probability that ``draw_redshift(spec)`` gives a redshift at or in front of ``zl``."""
    if "values" in spec and spec["values"] is not None:
        vals = np.asarray(spec["values"], float)
        w = np.ones(len(vals)) if spec.get("weights") is None else np.asarray(spec["weights"], float)
        return float(w[vals <= zl].sum() / w.sum())
    edges, w = np.asarray(spec["edges"], float), np.asarray(spec["weights"], float)
    lo, hi = edges[:-1], edges[1:]
    frac = np.clip((zl - lo) / (hi - lo), 0.0, 1.0)
    return float((w * frac).sum() / w.sum())


def check_redshifts(spec, zl: float) -> float:
    """The foreground fraction of ``spec`` (see ``foreground_fraction``); ValueError when it is the whole of it."""
    f = foreground_fraction(spec, zl)
    if f >= 1.0:
        raise ValueError(f"sources.redshift {spec} lies entirely at or in front of the lens redshift {zl:.4f}: "
                         "no source can be lensed")
    return f


class SourceSampler:
    """Draws candidate ``index`` on the mesh of an ``ImageFinder``."""

    def __init__(self, finder, cfg: dict, seed: int):
        self.finder, self.lens, self.seed = finder, finder.lens, int(seed)
        self.cfg = {**DEFAULTS, **(cfg or {})}
        kinds = {k: float(v) for k, v in self.cfg["seed_kinds"].items() if float(v) > 0}
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown seed kinds {sorted(unknown)}; known {sorted(KINDS)}")
        if not kinds:
            raise ValueError("seed_kinds needs at least one positive weight")
        self.kinds = sorted(kinds)
        w = np.array([kinds[k] for k in self.kinds])
        self.kind_p = w / w.sum()
        X, Y = finder.mesh
        self.X, self.Y = X, Y
        self.rad = np.hypot(X, Y)
        lo, hi = self.cfg["seed_radius"]
        self.outer = self.rad < min(hi, finder.half)
        self.rmask = (self.rad > lo) & self.outer
        self.subhalo_mask = None                 # mesh nodes near massive satellites, for the subhalo kind

    def draw(self, index: int) -> Candidate:
        """Candidate ``index``; its ``outcome`` says whether it has a source position (see the module notes)."""
        rng = np.random.default_rng([self.seed, int(index)])
        kind = self.kinds[rng.choice(len(self.kinds), p=self.kind_p)]
        z = draw_redshift(rng, self.cfg["redshift"])
        if z <= self.lens.zl:
            return Candidate(int(index), kind, z, 0.0, None, None, "foreground")
        s = float(self.lens.efficiency(z))
        if s <= 0:
            return Candidate(int(index), kind, z, s, None, None, "foreground")
        beta, seed = KINDS[kind](self, rng, s)
        if beta is None:
            return Candidate(int(index), kind, z, s, None, None, "no_seed")
        return Candidate(int(index), kind, z, s, beta, seed)

    # --------------------------------------------------------------- kinds
    def _image_seed(self, rng, s, mask):
        pool = np.flatnonzero(mask.ravel())
        if not len(pool):
            return None, None
        p = rng.choice(pool)
        half_step = 0.5 * self.finder.step
        th = np.array([self.X.ravel()[p], self.Y.ravel()[p]]) + rng.uniform(-half_step, half_step, 2)
        beta = th - s * self.lens.alpha(th[None])[0]
        return beta, th

    def _negative(self, rng, s):
        return self._image_seed(rng, s, (self.finder.det(s) < 0) & self.rmask)

    def _critical(self, rng, s):
        thr = rng.choice(np.asarray(self.cfg["critical_det"], float))
        return self._image_seed(rng, s, (np.abs(self.finder.det(s)) < thr) & self.rmask)

    def _subhalo(self, rng, s):
        if self.subhalo_mask is None:
            return None, None
        return self._image_seed(rng, s, (self.finder.det(s) < 0) & self.subhalo_mask & self.outer)

    def _source(self, rng, s):
        neg = self.finder.det(s) < 0
        if not neg.any():
            return None, None
        bx, by = self.finder.source_plane(s)
        bx, by = bx[neg], by[neg]
        beta = np.array([rng.uniform(bx.min(), bx.max()), rng.uniform(by.min(), by.max())])
        return beta, None


KINDS = {"negative": SourceSampler._negative, "critical": SourceSampler._critical,
         "source": SourceSampler._source, "subhalo": SourceSampler._subhalo}
