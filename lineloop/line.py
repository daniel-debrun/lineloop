"""A line spec: everything Line Loop needs to know about one production line, in a small TOML file.

The rest of the framework never names a station, a line or a time unit; it asks a `Line`. A spec says:

  segments      the line's sections, each with its stations in flow order (a part may skip stations, and
                parallel alternatives are simply listed one after the other)
  checkpoints   named decision points: a part is scored live when it reaches that station or any later one,
                using only what is known by then. Each checkpoint becomes a model recipe of the same name.
  time          how many hours one unit of the data's clock is, and the clock's resolution (times within one
                step count as "the same time")
  data          where the data is and in which format: "events" (generic tables of station visits, measurements
                and results) or "bosch" (the Kaggle competition's wide CSVs)
  families      route blocks before a checkpoint, for the inspection policy (default: the segments)
  shifts        shift length, for the inspection cell's staffing (default 8 h)
  economics     costs and inspection-cell assumptions (defaults in lineloop.optimize.inspection.ECONOMICS)
  uncertainty   how far each assumption may be off: [low, high] multipliers for the twin's robustness sweep
                (defaults in lineloop.twin.robust.RANGES)
  signal        optional [[signal]] tables: raw waveforms a station records that only the edge node sees
                (lineloop.edge.signals); their per-part features are measurements of the signal's last station
  edge          optional, per checkpoint: the decision's deadline and the network to a server, for
                `lineloop place` (where should this checkpoint's model run?)
  simulate      optional: how to generate a synthetic history of this line (lineloop.data.simulate)

At ingest the spec is resolved against the data (stations seen, measurement columns, the group stations) and
written next to the cache as line.json, so every later command reads the line from the cache it works on.
See docs/LINE_SPEC.md for the full reference and lines/ for examples.
"""
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:                                           # pragma: no cover
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[1]
LINE_FILE = "line.json"
FORMATS = ("events", "bosch")
WEEK_HOURS = 168.0


@dataclass
class Line:
    name: str
    description: str = ""
    segments: dict = field(default_factory=dict)        # segment -> [stations in flow order]
    checkpoints: dict = field(default_factory=dict)     # checkpoint name -> station
    unit_hours: float = 1.0                             # hours per unit of the data's clock
    resolution: float = 1.0                             # the clock's step, in data units
    group_min_share: float = 0.10                       # stations at least this share of parts visit ...
    group_stations: list | None = None                  # ... become group stations (or list them here)
    families: dict = field(default_factory=dict)        # family -> [first station, last station] (inclusive)
    shift_hours: float = 8.0
    economics: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)     # assumption -> [low, high] multipliers
    data: dict = field(default_factory=dict)            # format, source, cache, file names
    simulate: dict | None = None
    signals: list = field(default_factory=list)         # [[signal]] tables, as written
    edge: dict = field(default_factory=dict)            # checkpoint -> deadline and network figures
    measurements: dict = field(default_factory=dict)    # measurement column -> station (resolved at ingest)
    spec: str | None = None                             # the TOML file this came from

    # ------------------------------------------------------------------ stations
    @property
    def stations(self):
        """Every station, in flow order."""
        return [s for sts in self.segments.values() for s in sts]

    def segment_of(self, station):
        return self._seg()[station]

    def _seg(self):
        if not hasattr(self, "_seg_cache"):
            self._seg_cache = {s: g for g, sts in self.segments.items() for s in sts}
            self._pos_cache = {s: i for i, s in enumerate(self.stations)}
        return self._seg_cache

    def position(self, station):
        self._seg()
        return self._pos_cache[station]

    def checkpoint_station(self, checkpoint):
        """A checkpoint name or a station name -> the station; None stays None (end of line)."""
        if checkpoint is None:
            return None
        if checkpoint in self.checkpoints:
            return self.checkpoints[checkpoint]
        if checkpoint in self._seg():
            return checkpoint
        raise ValueError(f"{self.name}: no checkpoint or station {checkpoint!r}; checkpoints: "
                         f"{', '.join(self.checkpoints) or 'none'}")

    def before(self, checkpoint, present=None):
        """Stations strictly before the checkpoint in flow order (all of them for None), optionally only those in
        `present` (e.g. the cache's columns)."""
        cut = None if checkpoint is None else self.position(self.checkpoint_station(checkpoint))
        keep = None if present is None else set(present)
        return [s for i, s in enumerate(self.stations) if (cut is None or i < cut) and (keep is None or s in keep)]

    def after(self, checkpoint, present=None):
        """The checkpoint's station and every later one."""
        cut = self.position(self.checkpoint_station(checkpoint))
        keep = None if present is None else set(present)
        return [s for i, s in enumerate(self.stations) if i >= cut and (keep is None or s in keep)]

    def family_blocks(self, checkpoint):
        """{family: [stations]} before the checkpoint: the blocks of a route the inspection policy may tell apart.
        Defaults to the segments; blocks with no station before the checkpoint are dropped."""
        allowed = set(self.before(checkpoint))
        if self.families:
            out = {}
            for name, (a, b) in self.families.items():
                i, j = self.position(a), self.position(b)
                out[name] = [s for s in self.stations[i:j + 1] if s in allowed]
        else:
            out = {g: [s for s in sts if s in allowed] for g, sts in self.segments.items()}
        return {k: v for k, v in out.items() if v}

    def signal_specs(self):
        from .edge.signals import from_spec
        return [from_spec(d) for d in self.signals]

    # ------------------------------------------------------------------ time
    @property
    def day(self):
        """One day in data units."""
        return 24.0 / self.unit_hours

    @property
    def week(self):
        return WEEK_HOURS / self.unit_hours

    @property
    def ticks_per_unit(self):
        """Clock steps per data unit: times are keyed on round(t * ticks_per_unit)."""
        return 1.0 / self.resolution

    @property
    def n_shifts(self):
        return int(round(WEEK_HOURS / self.shift_hours))

    # ------------------------------------------------------------------ files
    def path(self, key, default=None):
        """A path from the [data] table, relative to the repository unless absolute."""
        v = self.data.get(key, default)
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else REPO / p

    @property
    def source(self):
        return self.path("source", f"data/{self.name}")

    @property
    def cache(self):
        return self.path("cache", f"data/cache/{self.name}")

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, indent=1)

    def save(self, cache):
        Path(cache).mkdir(parents=True, exist_ok=True)
        (Path(cache) / LINE_FILE).write_text(self.to_json())


# ---------------------------------------------------------------------- reading
def _spec_path(spec):
    p = Path(spec)
    if p.exists():
        return p
    for cand in (REPO / "lines" / f"{spec}.toml", REPO / "lines" / spec, REPO / spec):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no line spec {spec!r} (a .toml path, or a name in {REPO / 'lines'})")


def read(spec):
    """Line from a TOML spec (a path, or the name of a file in lines/). Not yet resolved against data."""
    p = _spec_path(spec)
    d = tomllib.loads(p.read_text())
    return from_dict(d, spec=str(p))


def from_dict(d, spec=None):
    known = {"name", "description", "segment", "checkpoints", "time", "stations", "families", "shifts",
             "economics", "uncertainty", "data", "simulate", "signal", "edge"}
    extra = set(d) - known
    if extra:
        raise ValueError(f"unknown keys in line spec: {sorted(extra)}; expected {sorted(known)}")
    if "name" not in d:
        raise ValueError("a line spec needs a name")
    t = d.get("time", {})
    st = d.get("stations", {})
    segs = {}
    for s in d.get("segment", []):
        if s["name"] in segs:
            raise ValueError(f"segment {s['name']!r} listed twice")
        segs[s["name"]] = list(s["stations"])
    ln = Line(name=d["name"], description=d.get("description", ""), segments=segs,
              checkpoints=dict(d.get("checkpoints", {})), unit_hours=float(t.get("unit_hours", 1.0)),
              resolution=float(t.get("resolution", 1.0)), group_min_share=float(st.get("group_min_share", 0.10)),
              group_stations=list(st["group_stations"]) if "group_stations" in st else None,
              families={k: list(v) for k, v in d.get("families", {}).items()},
              shift_hours=float(d.get("shifts", {}).get("hours", 8.0)), economics=dict(d.get("economics", {})),
              uncertainty={k: [float(x) for x in v] for k, v in d.get("uncertainty", {}).items()},
              data=dict(d.get("data", {})), simulate=d.get("simulate"), signals=list(d.get("signal", [])),
              edge={k: dict(v) for k, v in d.get("edge", {}).items()}, spec=spec)
    fmt = ln.data.setdefault("format", "events")
    if fmt not in FORMATS:
        raise ValueError(f"data.format must be one of {FORMATS}, not {fmt!r}")
    if fmt == "events" and not segs:
        raise ValueError("an events-format line must list its segments and stations ([[segment]] tables)")
    check(ln)
    return ln


def check(ln):
    """Consistency of a spec (and of a resolved line): unique stations, known checkpoints, valid times, shifts."""
    sts = ln.stations
    dup = sorted({s for s in sts if sts.count(s) > 1})
    if dup:
        raise ValueError(f"stations listed more than once: {dup}")
    if sts:
        for name, s in ln.checkpoints.items():
            if s not in sts:
                raise ValueError(f"checkpoint {name!r} names station {s!r}, which no segment lists")
        for name, ab in ln.families.items():
            if len(ab) != 2 or any(x not in sts for x in ab):
                raise ValueError(f"family {name!r} must be [first station, last station] of listed stations")
        for s in ln.group_stations or []:
            if s not in sts:
                raise ValueError(f"group station {s!r} is not on the line")
        for sg in ln.signal_specs():
            if any(x not in sts for x in sg.stations):
                raise ValueError(f"signal {sg.name!r} names stations that are not on the line: {sg.stations}")
        for cp in ln.edge:
            if cp not in ln.checkpoints:
                raise ValueError(f"[edge.{cp}]: no checkpoint named {cp!r}")
    for k, v in ln.uncertainty.items():
        if len(v) != 2 or not 0 < v[0] <= v[1]:
            raise ValueError(f"uncertainty.{k} must be [low, high] multipliers with 0 < low <= high, not {v}")
    clash = set(ln.checkpoints) & {"offline", "live"}
    if clash:
        raise ValueError(f"checkpoint names {sorted(clash)} are reserved for the end-of-line recipes")
    if ln.unit_hours <= 0 or ln.resolution <= 0:
        raise ValueError("time.unit_hours and time.resolution must be positive")
    if abs(WEEK_HOURS / ln.shift_hours - round(WEEK_HOURS / ln.shift_hours)) > 1e-9:
        raise ValueError(f"shifts.hours must divide a week (168 h) evenly, not {ln.shift_hours}")
    return ln


def load(cache):
    """The resolved line stored with a cache (written by `python -m lineloop line ingest`)."""
    p = Path(cache) / LINE_FILE
    if not p.exists():
        raise FileNotFoundError(f"{cache} has no {LINE_FILE}; run `python -m lineloop line ingest --line <spec>`")
    d = json.loads(p.read_text())
    return check(Line(**d))
