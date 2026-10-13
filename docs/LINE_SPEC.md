# Describing a line

Line Loop works on any production line you can describe in a short TOML file, the **line spec**, and export as
three plain tables. This page is the reference for both, and ends with a step-by-step guide for adding a line.

Three complete examples are in [`lines/`](../lines):

| spec | what it shows |
|---|---|
| [`battery-module.toml`](../lines/battery-module.toml) | a simulated EV battery-module line: named stations, parallel machines, three checkpoints, raw weld signals only an edge node sees, a minute clock, 12-hour shifts, string serial numbers, the generic `events` format, placement figures |
| [`bosch.toml`](../lines/bosch.toml) | the real Bosch line from Kaggle: stations read from the data, the `bosch` format |
| [`bosch-synthetic.toml`](../lines/bosch-synthetic.toml) | a simulated line in the Bosch layout, so the Bosch adapter runs without the Kaggle files |

`python -m lineloop line show --line <spec>` prints what Line Loop understood from a spec. After ingest it also
shows what it learned from the data.

## The words used here

| term | meaning |
|---|---|
| **part** | one unit moving down the line (a module, a board, an engine block). It has an identifier. |
| **station** | a place on the line that records a time when a part passes it, and may measure the part |
| **segment** | a named section of the line: an ordered list of stations |
| **flow order** | the order of every station, segment after segment. A part may skip stations, but never goes backwards in this order. |
| **final test** | the check at the end of the line that passes or fails a part. Its result is the label Line Loop learns to predict. |
| **checkpoint** | a station where a part can be scored *before* it reaches the end, using only what is known when it arrives there |
| **labelled part** | a part whose final-test result is in the data. Unlabelled parts still count: they occupy the line and shape every "neighbour" feature. |
| **origin** | the station where a failed part's defect arose, if failure analysis records it. An inspection at a checkpoint can only find defects that already exist there. |
| **signal** | a raw waveform a station records while it works (a weld's power, a press's force), too large for the factory system to keep. Only an edge node at the cell sees it. |

## The spec

```toml
name = "battery-module"                       # required: names the cache folder, the runs, the model card
description = "EV battery-module assembly"    # optional, shown by `line show`

[data]
format = "events"                             # "events" (generic tables) or "bosch" (the Kaggle CSV layout)
source = "data/sim/battery-module"            # folder with the data files (relative to the repository, or absolute)
cache = "data/cache/battery-module"           # where `line ingest` writes the cache (default data/cache/<name>)
# visits = "visits"                           # file names without extension (defaults shown)
# results = "results"
# measurements = "measurements"
# [data.columns]                              # if your export uses other column names:
# part = "serial_no"                          #   framework name = "your name"
# time = "timestamp"

[[segment]]                                   # one table per segment, in flow order
name = "cell-prep"
stations = ["ocv-sort", "plasma-clean", "glue-dispense", "stack-a", "stack-b"]

[[segment]]
name = "welding"
stations = ["busbar-weld-1", "busbar-weld-2", "weld-vision"]

[[segment]]
name = "closing"
stations = ["module-cover"]

[[segment]]
name = "test"
stations = ["insulation-test", "bms-flash", "eol-test"]

[time]
unit_hours = 0.016666666666666666             # hours per unit of the data's time column (here: minutes)
resolution = 1.0                              # the clock's step, in those units: times in one step are "the same"

[checkpoints]                                 # name = station; the demo runs the first
post-weld = "module-cover"
pre-weld = "busbar-weld-1"

[[signal]]                                    # optional: raw waveforms a station records (the edge node's input)
name = "weld"
stations = ["busbar-weld-1", "busbar-weld-2"]
channels = ["power", "photodiode", "depth"]
rate_hz = 20000
samples = 500                                 # per repeat (one weld)
per_part = 24                                 # repeats per part

[stations]
group_min_share = 0.10                        # stations at least this share of parts visit get group features
# group_stations = ["stack-a", "stack-b"]     # ... or name them yourself

[families]                                    # optional: route blocks the inspection policy may treat differently
stacker-a = ["stack-a", "stack-a"]            # [first station, last station], inclusive, in flow order
stacker-b = ["stack-b", "stack-b"]

[shifts]
hours = 12                                    # must divide a week (168 h): 4, 6, 8, 12, 24 ...

[economics]                                   # optional: overrides the defaults below
value_per_defect = 700.0

[economics.pre-weld]                          # optional: a checkpoint's own values
value_per_defect = 900.0

[uncertainty]                                 # optional: how far each assumption may be off (the twin's sweep)
value_per_defect = [0.33, 3.0]

[edge.post-weld]                              # optional: deadline and network, for `lineloop place`
deadline_s = 30.0
undecided = "stop"

[simulate]                                    # optional: lets `line simulate` generate a history of this line
parts = 100000
```

### `[data]`

`format` selects the adapter that turns the files into the cache.

- **`events`** reads the three tables described [below](#the-events-format). It is the format for a new line.
- **`bosch`** reads the Kaggle competition's `train_numeric`, `train_date` and `test_date` files (`.csv` or
  `.csv.zip`). Station names are taken from the column names (`L{line}_S{station}_F{n}` for measurements,
  `_D{n}` for timestamps). With this format the spec may leave out the `[[segment]]` tables: the stations are
  then ordered by station number, one segment per line (L0 to L3).

### `[[segment]]`

Each segment has a `name` and its `stations` in flow order. Together they define the flow order of the whole line,
which decides what a checkpoint may see. List parallel machines (two welders doing the same job) one after the
other. A part visits one of them, and the checkpoint rule "this station or any later one" still works.

Segment names become feature names: `<segment>_start`, `<segment>_dur` (time spent in the segment) and
`<segment>_n` (stations visited in it).

### `[time]`

| key | meaning |
|---|---|
| `unit_hours` | hours per unit of the time column. 1 for hours, 1/60 for minutes, 10 for Bosch (one Bosch unit is about 10 hours, inferred from the day and week cycles). |
| `resolution` | the clock's step in those units. Two parts started in the same step "started together". This drives the group features and the edge node's tick. Use the precision your stations actually log with. |

If the time column holds date-times instead of numbers, the ingest converts them to minutes since Monday
1970-01-05 and sets `unit_hours` to 1/60 itself, so time of day and day of week keep their meaning.
`resolution` is then in minutes.

### `[checkpoints]`

`name = "station"` pairs. A part is decided at a checkpoint when it first reaches that station **or any later one in
flow order**. A part that never gets that far is decided after its last earlier station. Each checkpoint becomes
a model recipe of the same name (see [recipes](#recipes)). The names `offline` and `live` are reserved.

A line without checkpoints can still train end-of-line models. The live loop (edge node, twin, optimiser) needs at
least one checkpoint, because it is about acting on a part before the end.

### `[stations]`

Group features compare a part with the other parts that were at the same station at the same time. By default the
stations at least `group_min_share` of all parts visit get them (on Bosch: 31 of 52 stations, which is exactly the
hand-picked list of the earlier Bosch-only code). Set `group_stations` to choose them yourself.

### `[families]`

The inspection policy sets a separate flagging threshold for each **family**: parts that took the same route
blocks before the checkpoint. A family is named `[first station, last station]`, a block in flow order. A part's
family is the set of blocks it touched before the checkpoint, such as `L0a+L2`. The most common patterns get
their own threshold; the rest share one. Without this table, the blocks are the segments.

### `[[signal]]`

A station that records a waveform while it works: a welder's power and spatter photodiode, a press's force, a
nut-runner's torque. Factory systems rarely keep these (here 144 kB per module), so only an edge node at the cell
can use them. Each `[[signal]]` table gives the `stations` that record it (parallel machines: whichever the part
visits), its `channels`, `rate_hz`, `samples` per repeat and repeats `per_part` (welds per module).

The edge node's **signal stage** ([`lineloop/edge/signals.py`](../lineloop/edge/signals.py)) turns a part's raw
block into 9 numbers per channel: per repeat, the mean of the middle 80 %, the high-frequency energy and the
mean of the last 5 %, then their mean, max and min over the repeats. They are stored as measurements of the
signal's **last** station, named `<station>.<signal>.<channel>.<feature>_<summary>`. So a checkpoint after that
station sees them, and the rest of the framework treats them like any other measurement. A model that uses them
is compiled with the signal stage in front of its trees: TensorRT or ONNX Runtime takes the raw block and
computes the features itself ([How it works § edge compile](HOW_IT_WORKS.md#the-compiler-trees-as-tensors)).

Replaying raw signals needs them recorded. The simulator keeps, per part, the latent quality of each weld and a
seed, and re-creates the exact waveforms on demand; real recorded waveforms are not yet an input format.

### `[edge.<checkpoint>]`

Where should a checkpoint's model run? `python -m lineloop place` answers per checkpoint
([How it works § placement](HOW_IT_WORKS.md#stage-5--place)), from these figures, all assumptions
about the plant:

| key | default | meaning |
|---|---|---|
| `deadline_s` | 60 | seconds a decision has to reach the line |
| `uplink_mbps` | 10 | the cell's share of the network to a plant server |
| `rtt_ms` | [4, 60] | round trip to the server: median and 99th percentile |
| `outages_per_month`, `outage_minutes` | 2, 30 | network outages and maintenance windows |
| `undecided` | `"pass"` | what the line does without a decision in time: `"pass"` (the part goes on unchecked, as today) or `"stop"` (an interlocked gate: the line waits) |
| `downtime_per_hour` | 0 | the cost of an hour of a stopped line (for `"stop"`) |
| `edge_cost_per_month` | 150 | an edge node at the cell: hardware over its life, power, upkeep |
| `record_bytes` | 4 per feature | what a server needs per part besides any raw signal |

### `[shifts]` and `[economics]`

The twin and the optimiser staff the inspection cell per shift slot. A week is cut into slots of `hours`: 21 of
8 h, or 14 of 12 h. `[economics]` overrides any of the assumed costs and cell parameters, and an
`[economics.<checkpoint>]` table overrides them for one checkpoint (catching a defect before welding is worth
more than after). Every one of them is also a flag of `python -m lineloop plan`:

| key | default | meaning |
|---|---|---|
| `value_per_defect` | 150 | saved when a failing part is caught at the checkpoint instead of at the final test |
| `value_per_station` | 8 | saved for each station after the checkpoint that a caught failing part no longer visits |
| `handling` | 3 | cost per inspection, besides the inspector's time |
| `inspector_hour` | 45 | fully loaded labour cost |
| `wait_hour` | 1 | cost per part-hour spent waiting at the cell |
| `svc_mean_h`, `svc_cv` | 0.25, 0.5 | inspection time: mean (hours) and coefficient of variation (lognormal) |
| `detect` | 0.9 | chance an inspection finds a real defect |
| `max_wait_h` | 2.0 | a flagged part that would wait longer moves on uninspected |

These are assumptions, not data. The plan reports a break-even sweep over `value_per_defect` for that reason, and
the twin replays the chosen policies with every assumption drawn from a range around its value. `[uncertainty]`
sets those ranges as `[low, high]` multipliers (defaults: `value_per_defect` 0.33 to 3, `detect` 0.67 to 1.1,
capped at 0.99, `svc_mean_h` 0.6 to 2, the other costs 0.5 to 2, `inspector_hour` 0.7 to 1.5).

### `[simulate]`

Optional. It lets `python -m lineloop line simulate` (and `demo` on a line with no data) generate a history in
the spec's own format, with the structure Line Loop looks for:

| key | default | meaning |
|---|---|---|
| `parts`, `parts_per_week` | 60000, 20000 | history length and rate |
| `batch` | [1, 8] | parts released together (smallest, largest) |
| `weekend` | 0.2 | release rate on Saturday and Sunday, relative to weekdays |
| `route` | every segment | list of `{share, segments, risk}`: the routes parts take, and extra failure log-odds per route |
| `parallel` | none | lists of interchangeable stations: a part visits one of each group |
| `step_hours` | 0.5 | mean extra time at a station (exponential, for half the visits) |
| `measurements_per_station`, `measurement_names` | 3, `"m{j}"` | values each station measures, and their names |
| `noise`, `missing_share` | 0.1, 0.05 | measurement noise, and the share of values not recorded |
| `bad_share`, `bad_window_hours`, `bad_shift`, `bad_risk` | 0.03, 0.1, 0.08, 4.0 | "bad batch" windows: share of windows, length, measurement shift, log-odds of a defect from it |
| `driver` | none | list of `{station, measurement, weight}`: a defect mechanism at that station, log-odds `offset + weight x value` |
| `drift` | none | list of `{station, measurement, per_week}`: the measurement's mean moves every week |
| `signal_driver` | none | list of `{signal, weight, above, bad_shift}`: every repeat of a `[[signal]]` (each weld) has a quality z ~ N(0, 1) and is its own defect mechanism, log-odds `offset + weight x (z - above)`; its waveforms show z through noisy signatures |
| `fail_rate`, `labelled_share` | 0.012, 1.0 | the expected share of parts that fail (the offset is solved to hit it), and the share whose result is kept |
| `id_prefix`, `id_jitter_hours` | none, 3 | string serial numbers (e.g. `"MOD-"`), and how loosely numbering follows release order |
| `seed` | 0 | random seed |

**How a part fails.** Each driver, each signal repeat, a bad window (at the part's first station) and a background
risk (at a random station on its route, raised by its route's `risk`) are independent defect mechanisms. The part
fails if any fires, and its **origin** is the earliest station among those that fired. The results table carries
it, so the twin only credits an inspection with defects that already exist where it stands. Put a driver before a
checkpoint and the checkpoint model can find it, and an inspection there can catch it; put one after, and
neither can.

## The events format

Three tables, as `.parquet` or `.csv` files in the `source` folder:

**visits** (required): one row per timestamp a part logged at a station.

| column | type | meaning |
|---|---|---|
| `part` | integer or string | the part's identifier |
| `station` | string | must be one of the spec's stations |
| `time` | number (spec units) or date-time | when |

A part may log several times at one station; the first and last are kept.

**results** (required): the final test.

| column | type | meaning |
|---|---|---|
| `part` | as in visits | |
| `failed` | 0 or 1 | 1 = failed the final test |
| `origin` | string, optional | for a failed part: the station where its defect arose, if failure analysis says |

A part with no row is on the line but unlabelled. Without `origin`, every failure counts as findable at any
checkpoint, and the inspection cell's `detect` share has to account for defects made later.

**measurements** (optional): what stations measured.

| column | type | meaning |
|---|---|---|
| `part`, `station` | as above | |
| `name` | string | the measurement's name at that station (e.g. `bead_width`) |
| `value` | number | |

Each (station, name) pair becomes a feature column named `<station>.<name>`. A checkpoint model only sees
measurements taken before its checkpoint.

**Part identifiers.** Integers keep their order: if they already run 1, 2, 3 ... they are used as they are,
otherwise they are renumbered by rank. Strings are numbered in order of first appearance. Either way the numbering
should follow production order as closely as the data allows: "the part numbered just before this one" is a
feature. The mapping is saved in the cache as `ids.parquet`.

The ingest checks the data against the spec. It stops on a station the spec does not list. It notes stations
the spec lists but the data never shows, and results for parts that never appear in visits.

## The cache

`python -m lineloop line ingest --line <spec>` writes the cache that every later step reads. No step after this
one knows which format the data came in.

| file | content |
|---|---|
| `stations.parquet` | one row per part: `Id`, `is_train` (1 = result known), `<station>_t0` / `_t1` (first / last time at each station), `date_hash`, `date_nnz` |
| `numeric.parquet` | one row per labelled part, in `Id` order: `Id`, `Response` (1 = failed), the measurement columns, `num_hash`, `num_nnz` |
| `origins.parquet` | failed parts with a known origin: `Id`, `origin` (only if the data has origins) |
| `line.json` | the spec resolved against the data: stations seen, measurement-to-station map, group stations |
| `meta.json`, `ids.parquet` | counts and timings; the identifier mapping (events format, when renumbered) |

The internal column names `Id`, `Response` and `is_train` come from the Bosch data, where the framework started.

## Recipes

A recipe says when a part is scored and what the model may know at that moment:

| recipe | scored at | may use |
|---|---|---|
| `offline` | end of line | everything, including the parts made *after* this one. The competition setting: the strongest score, but not something a live system can compute. |
| `live` | end of line | only what existed when the part finished |
| one per checkpoint | the checkpoint | only what existed when the part reached the checkpoint, and none of its own data from that station on |

`kaggle` and `causal` are accepted as old names for `offline` and `live`.

## Adding your line, step by step

1. **Export three tables** from your MES, historian or test stands: visits, results and (if you have them)
   measurements, with the columns above. CSV is fine. Put them in a folder, e.g. `data/my-line/`.
2. **Write the spec.** Copy [`lines/battery-module.toml`](../lines/battery-module.toml) to `lines/my-line.toml`.
   Set `name`, `[data] source`, the segments and stations in flow order, the clock, and one checkpoint where
   acting early would help. Leave out `[simulate]`.
3. **Check it:** `python -m lineloop line show --line my-line` prints the stations, checkpoints, recipes and
   families it understood.
4. **Ingest:** `python -m lineloop line ingest --line my-line` reads the tables, checks them against the spec and
   writes the cache. Fix whatever it reports.
5. **Run the loop:** `python -m lineloop demo --line my-line` runs the benchmark cross-validation, the checkpoint
   model, the edge compile and replay, the inspection plan, the twin's robustness sweep and the placement check,
   and prints a scoreboard. Or run the steps one by one (see [How Line Loop works](HOW_IT_WORKS.md#the-pipeline-in-run-order)).
6. **Set the economics** in `[economics]` to your costs and `[uncertainty]` to how sure you are of them, and read
   the plan's break-even table and the twin's robustness table before trusting a saving.
7. **Set `[edge.<checkpoint>]`** to your plant's network and what the line does without a decision, before
   deciding where the model runs.

No code changes are needed for a new line. If a step needs to know something about your line that the spec can't
express, that is a gap in the framework, not in your line; see the limits in
[How Line Loop works](HOW_IT_WORKS.md#limits).
