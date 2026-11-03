"""Draw the documentation figures, assets/<figure>-light.svg and assets/<figure>-dark.svg: docs/architecture.md's
block graph (block-graph), docs/batched.md's batched engine (batched-launch, batched-stages, batched-schedule) and
B200 charts from docs/batched.md's measurements (throughput-b200 for the README, canvas-b200, phased-b200).

usage: python tools/architecture_svg.py [repo root]
"""

import sys
from pathlib import Path

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": {"fg": "#1F2328", "muted": "#59636E", "line": "#8C959F", "box": "#FFFFFF", "box_line": "#D0D7DE",
              "host": "#F6F8FA", "device": "#F0FAF3", "device_line": "#2EA44F", "loop": "#FFFFFF",
              "layer": "#E6F6EA", "chip": "#FFFFFF", "accent": "#2EA44F", "accent_soft": "#8DDCA2",
              "prefill": "#0969DA", "prefill_fill": "#DDF4FF", "commit": "#BC4C00", "commit_fill": "#FFF1E5",
              "lane": "#8250DF", "lane_fill": "#FBEFFF", "switch": "#9A6700", "switch_fill": "#FFF8C5",
              "fixed256": "#80CCFF", "fixed128": "#218BFF", "fixed32": "#0550AE"},
    "dark": {"fg": "#E6EDF3", "muted": "#9198A1", "line": "#6E7681", "box": "#161B22", "box_line": "#3D444D",
             "host": "#151B23", "device": "#0F1F15", "device_line": "#2EA44F", "loop": "#0D1117",
             "layer": "#12301C", "chip": "#161B22", "accent": "#3FB950", "accent_soft": "#2EA44F",
             "prefill": "#4493F8", "prefill_fill": "#0D2645", "commit": "#DB6D28", "commit_fill": "#3A2212",
             "lane": "#AB7DF8", "lane_fill": "#28194A", "switch": "#D29922", "switch_fill": "#33270A",
             "fixed256": "#1F6FEB", "fixed128": "#58A6FF", "fixed32": "#A5D6FF"},
}


def text(x, y, s, size=13, weight=400, fill="fg", anchor="start", family=FONT, style=""):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{{{fill}}}" text-anchor="{anchor}"{style}>{s}</text>')


def rect(x, y, w, h, fill, stroke, r=10, dash="", width=1.2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{{{fill}}}" stroke="{{{stroke}}}" '
            f'stroke-width="{width}"{d}/>')


def arrow(points, color="line", width=1.6, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    path = "M " + " L ".join(f"{x} {y}" for x, y in points)
    return (f'<path d="{path}" fill="none" stroke="{{{color}}}" stroke-width="{width}" stroke-linejoin="round"{d} '
            f'marker-end="url(#head-{color})"/>')


def curve(path, color="accent", width=1.6, dash=""):
    """An arrow along an SVG path (`path`: its d attribute)."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{path}" fill="none" stroke="{{{color}}}" stroke-width="{width}"{d} '
            f'marker-end="url(#head-{color})"/>')


def line(x1, y1, x2, y2, color="line", width=1.2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{{{color}}}" stroke-width="{width}"{d}/>'


def badge(x, y, label):
    """A numbered circle centred at (x, y)."""
    return (f'<circle cx="{x}" cy="{y}" r="9" fill="{{box}}" stroke="{{muted}}" stroke-width="1.2"/>'
            + text(x, y + 4, label, size=11, weight=700, fill="muted", anchor="middle"))


def chips(x, y, items, h=30, gap=18, fill="chip", stroke="box_line"):
    """A row of labelled chips joined by arrows; items are (label, width). Returns the parts and the right edge."""
    parts = []
    for i, (label, w) in enumerate(items):
        parts.append(rect(x, y, w, h, fill, stroke, r=7))
        parts.append(text(x + w / 2, y + h / 2 + 4.5, label, size=12.5, anchor="middle"))
        if i + 1 < len(items):
            parts.append(arrow([(x + w + 3, y + h / 2), (x + w + gap - 3, y + h / 2)]))
        x += w + gap
    return parts, x - gap


def block_graph():
    p = []
    # Host lane.
    p.append(rect(20, 20, 236, 520, "host", "box_line", r=16))
    p.append(text(40, 52, "HOST", size=12, weight=700, fill="muted", style=' letter-spacing="1.2"'))
    p.append(text(84, 52, "once per block", size=12, fill="muted"))
    p.append(rect(40, 76, 196, 64, "box", "box_line"))
    p.append(text(56, 102, "Launch the block", size=14, weight=600))
    p.append(text(56, 124, "params → pinned H2D", size=12.5, fill="muted"))
    p.append(arrow([(138, 152), (138, 396)], color="line", width=1.2, dash="4 5"))
    p.append(text(150, 262, "no host sync", size=12, fill="muted"))
    p.append(text(150, 279, "between rounds", size=12, fill="muted"))
    p.append(rect(40, 404, 196, 56, "box", "box_line"))
    p.append(text(56, 428, "Read back the tokens", size=14, weight=600))
    p.append(text(56, 448, "block ids ← D2H", size=12.5, fill="muted"))
    p.append(arrow([(138, 462), (138, 472)]))
    p.append(rect(40, 476, 196, 48, "box", "box_line"))
    p.append(text(56, 496, "EOS / budget check", size=14, weight=600))
    p.append(text(56, 514, "stream the block", size=12.5, fill="muted"))

    # Device: one CUDA graph per block.
    p.append(rect(290, 20, 690, 520, "device", "device_line", r=16, width=1.6))
    p.append(text(312, 52, "DEVICE", size=12, weight=700, fill="accent", style=' letter-spacing="1.2"'))
    p.append(text(372, 52, "one CUDA graph per block", size=12, fill="accent"))

    p.append(rect(312, 72, 646, 52, "box", "box_line"))
    p.append(text(330, 104, "begin", size=14, weight=600))
    p.append(text(390, 104, "tokens ← prompt tail | deterministic noise", size=13, fill="muted"))

    p.append(arrow([(635, 124), (635, 140)]))
    p.append(rect(312, 144, 646, 308, "loop", "accent", r=12, dash="6 5", width=1.4))
    p.append(text(330, 170, "WHILE", size=12, weight=700, fill="accent", family=MONO))
    p.append(text(378, 170, "not done and step &lt; limit", size=13, weight=600))
    p.append(text(940, 170, "conditional node, decided on the GPU", size=12, fill="muted", anchor="end"))

    row, _ = chips(332, 198, [("self-conditioning", 148), ("embed", 70), ("L × layer", 96)])
    p += row
    p.append(text(940, 217, "probabilities → next round", size=12, fill="accent", anchor="end"))

    # One layer, expanded.
    p.append(arrow([(634, 230), (634, 246)], color="accent_soft", width=1.2))
    p.append(rect(332, 248, 606, 92, "layer", "accent_soft", r=10))
    p.append(text(348, 268, "layer", size=12, weight=700, fill="accent"))
    row, _ = chips(348, 280, [("norm", 46), ("QKV", 46), ("Q/K norm · RoPE · KV write", 172), ("attention", 76),
                              ("O", 30), ("norm", 46), ("MLP / MoE", 80)], gap=12)
    p += row
    p.append(text(348, 330, "attention reads the KV cache of every earlier block", size=11.5, fill="muted"))

    p.append(arrow([(378, 342), (378, 358)], color="accent_soft", width=1.2))
    row, end_c = chips(332, 360, [("final norm", 92), ("LM head", 82), ("row statistics", 120),
                                  ("accept · edit · re-noise", 176)])
    p += row
    p.append(text(332, 428, "every round re-reads the device counters; the loop exits when the block is settled",
                  size=12, fill="muted"))
    # Back to the next round's self-conditioning, around the layer.
    p.append(arrow([(end_c + 3, 375), (950, 375), (950, 184), (406, 184), (406, 196)], color="accent", width=1.6))

    p.append(arrow([(635, 452), (635, 468)]))
    p.append(rect(312, 470, 646, 52, "box", "box_line"))
    p.append(text(330, 501, "commit", size=14, weight=600))
    p.append(text(396, 501, "causal forward over the accepted block · writes its K/V", size=13, fill="muted"))

    # Host ↔ device.
    p.append(arrow([(236, 108), (300, 108), (300, 98), (310, 98)], color="accent", width=1.8))
    p.append(arrow([(312, 496), (284, 496), (284, 432), (238, 432)], color="accent", width=1.8))
    return p


def batched_launch():
    """The batched engine's launch: the host between launches, the launch graph, and one forward's rows."""
    p = []
    # Host lane: steps between two launches.
    p.append(rect(20, 20, 236, 520, "host", "box_line", r=16))
    p.append(text(40, 52, "HOST", size=12, weight=700, fill="muted", style=' letter-spacing="1.2"'))
    p.append(text(84, 52, "between launches", size=12, fill="muted"))

    def step(y, title, sub):
        p.append(rect(40, y, 196, 50, "box", "box_line"))
        p.append(text(56, y + 21, title, size=14, weight=600))
        p.append(text(56, y + 40, sub, size=12, fill="muted"))

    step(72, "Launch", "the graph for the row count")
    p.append(arrow([(60, 126), (60, 194)], color="line", width=1.2, dash="4 5"))
    p.append(text(74, 150, "no host step until", size=12, fill="muted"))
    p.append(text(74, 166, "a request finishes", size=12, fill="muted"))
    p.append(text(74, 182, "or parks", size=12, fill="muted"))
    steps = [("Collect", "finished requests ← D2H"), ("Canvas length", "policy(n): 256 · 128 · 32"),
             ("Admit", "waiting requests ≤ max_rows"), ("Lay out rows", "fill gaps, move rows"),
             ("Commit lanes", "count: least expected cost")]
    for i, (title, sub) in enumerate(steps):
        y = 200 + 66 * i
        step(y, title, sub)
        if i + 1 < len(steps):
            p.append(arrow([(138, y + 52), (138, y + 64)]))
    # Back to the next launch.
    p.append(arrow([(40, 489), (30, 489), (30, 97), (38, 97)]))

    # Device: one launch.
    p.append(rect(290, 20, 690, 520, "device", "device_line", r=16, width=1.6))
    p.append(text(312, 52, "DEVICE", size=12, weight=700, fill="accent", style=' letter-spacing="1.2"'))
    p.append(text(372, 52, "one launch: a StagedLoopGraph over every request", size=12, fill="accent"))

    p.append(rect(312, 72, 646, 50, "box", "box_line"))
    p.append(text(330, 102, "admit", size=14, weight=600))
    p.append(text(384, 102, "admitted → its first prompt rows · parked → a canvas of its new length", size=13,
                  fill="muted"))

    p.append(arrow([(635, 122), (635, 138)]))
    p.append(rect(312, 140, 646, 300, "loop", "accent", r=12, dash="6 5", width=1.4))
    p.append(text(330, 166, "WHILE", size=12, weight=700, fill="accent", family=MONO))
    p.append(text(378, 166, "some request runs and none finished or parked since the launch", size=13, weight=600))

    stages = [("forward · all rows", 150, ""), ("LM head + decoder", 140, "4 4"), ("advance · claim", 124, ""),
              ("self-conditioning", 140, "4 4")]
    x, centers = 332, []
    for i, (label, w, dash) in enumerate(stages):
        p.append(rect(x, 184, w, 30, "chip", "accent" if dash else "box_line", r=7, dash=dash))
        p.append(text(x + w / 2, 203.5, label, size=12.5, anchor="middle"))
        centers.append(x + w / 2)
        if i + 1 < len(stages):
            p.append(arrow([(x + w + 3, 199), (x + w + 15, 199)]))
        x += w + 18
    p.append(arrow([(x - 15, 199), (950, 199), (950, 176), (centers[0], 176), (centers[0], 182)], color="accent"))
    notes = [("every request's rows", "muted"), ("IF a canvas round", "accent"), ("each request moves on", "muted"),
             ("IF a round &gt; 0 next", "accent")]
    for c, (note, color) in zip(centers, notes):
        p.append(text(c, 232, note, size=11.5, fill=color, anchor="middle"))

    # One forward's rows.
    p.append(text(332, 266, "one forward's rows", size=12, weight=700, fill="accent"))
    p.append(text(456, 266, "each request owns a contiguous range of its canvas length (or prompt rows)", size=12,
                  fill="muted"))
    segments = [("A", "round 5", 86, "accent", "layer"), ("B", "round 0", 86, "accent", "layer"),
                ("C", "commit", 86, "commit", "commit_fill"), ("D", "prompt rows", 86, "prefill", "prefill_fill"),
                ("E", "round 2", 86, "accent", "layer"), ("", "pad", 50, "box_line", "box"),
                ("B", "commit", 86, "lane", "lane_fill"), ("", "pad", 42, "box_line", "box")]
    x, edges = 332, []
    for name, sub, w, stroke, fill in segments:
        p.append(rect(x + 1, 300, w - 2, 42, fill, stroke, r=5, dash="3 3" if sub == "pad" else ""))
        if name:
            p.append(text(x + w / 2, 317, name, size=12.5, weight=700, anchor="middle"))
            p.append(text(x + w / 2, 333, sub, size=11, fill="muted", anchor="middle"))
        else:
            p.append(text(x + w / 2, 325, sub, size=11, fill="muted", anchor="middle"))
        edges.append(x)
        x += w
    head, end = edges[6], x
    for a, b, label in ((332, head, "requests' rows: the LM head, decoder and SC run on the denoising ones"),
                        (head, end, "commit lanes")):
        p.append(line(a + 2, 290, b - 2, 290, color="muted"))
        p.append(line(a + 2, 286, a + 2, 294, color="muted"))
        p.append(line(b - 2, 286, b - 2, 294, color="muted"))
        p.append(text((a + b) / 2, 283, label, size=11.5, fill="muted", anchor="middle"))
    for i, note in enumerate((
            "B's stopped canvas commits in a lane while B's next canvas runs its round 0 in B's rows",
            "attention per request: prompt and commit rows causal, a canvas bidirectional within itself",
            "a per-row (slot, index) map ties rows to requests; padding rows map to an idle slot")):
        p.append(text(332, 368 + 20 * i, note, size=12, fill="muted"))

    p.append(arrow([(635, 440), (635, 456)]))
    p.append(rect(312, 458, 646, 52, "box", "box_line"))
    p.append(text(330, 489, "exit", size=14, weight=600))
    p.append(text(372, 489, "a request finished (DONE) or parked for another canvas length (SWITCH)", size=13,
                  fill="muted"))

    # Host ↔ device.
    p.append(arrow([(236, 97), (310, 97)], color="accent", width=1.8))
    p.append(arrow([(312, 484), (274, 484), (274, 225), (238, 225)], color="accent", width=1.8))
    return p


def batched_stages():
    """One request's stages and what moves it between them."""
    p = []
    p.append(text(24, 40, "PER REQUEST", size=12, weight=700, fill="muted", style=' letter-spacing="1.2"'))
    p.append(text(130, 40, "the stage lives at the request's KV slot; after every forward each request moves on by "
                  "itself", size=12, fill="muted"))

    def stage(x, y, title, sub, stroke, fill):
        p.append(rect(x, y, 200, 56, fill, stroke, r=10, width=1.4))
        p.append(text(x + 16, y + 24, title, size=14, weight=700, family=MONO))
        p.append(text(x + 16, y + 43, sub, size=11.5, fill="muted"))

    p.append(rect(14, 164, 84, 28, "host", "box_line", r=14, dash="4 4"))
    p.append(text(56, 182, "admitted", size=12, fill="muted", anchor="middle"))
    stage(126, 150, "PREFILL", "prompt rows, causal", "prefill", "prefill_fill")
    stage(425, 150, "DENOISE", "canvas round, bidirectional", "accent", "layer")
    stage(735, 150, "COMMIT", "canvas re-encoded, causal", "commit", "commit_fill")
    stage(425, 330, "DONE", "ends the launch; read back", "box_line", "host")
    stage(735, 330, "SWITCH", "parked; ends the launch", "switch", "switch_fill")

    p.append(arrow([(100, 178), (124, 178)], dash="4 4"))
    # PREFILL: the next prompt rows, then the first canvas.
    p.append(curve("M 196 150 C 196 112, 256 112, 256 148"))
    p.append(text(226, 108, "next prompt rows", size=11.5, fill="muted", anchor="middle"))
    p.append(arrow([(328, 178), (423, 178)], color="accent"))
    p.append(text(376, 170, "prompt done", size=11.5, fill="muted", anchor="middle"))
    # DENOISE: the next round, or a commit lane and the next canvas in the same forward.
    p.append(curve("M 465 150 C 465 108, 515 108, 515 148"))
    p.append(text(457, 116, "not stopped:", size=11.5, fill="muted", anchor="end"))
    p.append(text(457, 131, "next round", size=11.5, fill="muted", anchor="end"))
    p.append(curve("M 565 150 C 565 108, 612 108, 612 148", color="lane"))
    p.append(text(626, 108, "stopped, a commit lane free:", size=11.5, fill="lane"))
    p.append(text(626, 123, "commit in the lane while the next", size=11.5, fill="lane"))
    p.append(text(626, 138, "canvas starts, in one forward", size=11.5, fill="lane"))
    p.append(arrow([(627, 170), (733, 170)], color="accent"))
    p.append(text(680, 162, "stopped", size=11.5, fill="muted", anchor="middle"))
    p.append(arrow([(733, 192), (627, 192)], color="accent"))
    p.append(text(680, 210, "next canvas,", size=11.5, fill="muted", anchor="middle"))
    p.append(text(680, 224, "same length", size=11.5, fill="muted", anchor="middle"))
    # Ends of a launch.
    p.append(arrow([(525, 208), (525, 328)], color="accent"))
    p.append(text(515, 252, "stopped with an EOS within", size=11.5, fill="muted", anchor="end"))
    p.append(text(515, 267, "its budget, or its last canvas:", size=11.5, fill="muted", anchor="end"))
    p.append(text(515, 282, "never committed", size=11.5, fill="muted", anchor="end"))
    p.append(arrow([(835, 208), (835, 328)], color="accent"))
    p.append(text(845, 252, "next canvas wants", size=11.5, fill="muted"))
    p.append(text(845, 267, "another length:", size=11.5, fill="muted"))
    p.append(text(845, 282, "park", size=11.5, fill="muted"))
    p.append(arrow([(733, 358), (660, 358), (660, 240), (590, 240), (590, 210)], dash="4 4"))
    p.append(text(668, 300, "host: rows of", size=11.5, fill="muted"))
    p.append(text(668, 315, "the new length", size=11.5, fill="muted"))

    p.append(arrow([(24, 416), (64, 416)], color="accent"))
    p.append(text(74, 420, "after a forward, on the GPU (advance, claim)", size=12, fill="muted"))
    p.append(arrow([(380, 416), (420, 416)], dash="4 4"))
    p.append(text(430, 420, "between launches, by the host", size=12, fill="muted"))
    return p


# The example schedule: per request, (kind, first forward, last forward, label); "done" marks the forward after the
# request's last canvas.
SCHEDULE = {
    "A": [("canvas", 1, 3, "128"), ("commit", 4, 4), ("canvas", 5, 8, "128"), ("commit", 9, 9),
          ("canvas", 10, 12, "128"), ("done", 13, 13)],
    "B": [("commit", 1, 1), ("canvas", 2, 7, "128"), ("commit", 8, 8), ("canvas", 9, 14, "128"), ("commit", 15, 15),
          ("canvas", 16, 18, "128"), ("commit", 19, 19), ("canvas", 20, 24, "256 …")],
    "C": [("canvas", 1, 6, "128"), ("done", 7, 7)],
    "D": [("prefill", 7, 7), ("canvas", 8, 11, "128"), ("commit", 12, 12), ("canvas", 13, 17, "128"),
          ("done", 18, 18)],
}
LAUNCHES = [(1, 6, "3 × 128"), (7, 12, "3 × 128"), (13, 17, "2 × 128"), (18, 19, "128"), (20, 24, "256")]
HOST_STEPS = [
    "C finished: the host collects it and admits D (3 requests: 128-row canvases)",
    "A finished: 2 requests, and 128 rows stay (within one request of their range)",
    "D finished: 1 request, so the next canvases are 256 rows; B keeps 128 rows until its canvas is committed",
    "B committed and parked for 256 rows: the host gives it 256 rows",
]


def batched_schedule():
    """An example of four requests through five launches (illustration, not a measurement)."""
    p = []
    x0, cw, top, bottom = 130, 34, 72, 258
    rows = {name: 112 + 38 * i for i, name in enumerate(SCHEDULE)}

    def fx(f):
        return x0 + (f - 1) * cw

    p.append(text(20, 28, "EXAMPLE", size=12, weight=700, fill="muted", style=' letter-spacing="1.2"'))
    p.append(text(96, 28, "4 requests, at most 3 in flight, canvas lengths from the default policy (an illustration)",
                  size=12, fill="muted"))
    x = 20
    for kind, label in (("prefill", "prompt rows"), ("canvas", "canvas: one round per forward"),
                        ("commit", "commit"), ("host", "the host steps in")):
        if kind == "host":
            p.append(line(x, 47, x + 18, 47, color="muted", dash="4 4"))
            x += 24
        else:
            stroke, fill = {"prefill": ("prefill", "prefill_fill"), "canvas": ("accent", "layer"),
                            "commit": ("commit", "commit_fill")}[kind]
            p.append(rect(x, 41, 14, 12, fill, stroke, r=3))
            x += 20
        p.append(text(x, 51, label, size=12, fill="muted"))
        x += len(label) * 6.3 + 22

    for i, (first, last, n_rows) in enumerate(LAUNCHES):
        a, b = fx(first), fx(last + 1)
        if i % 2 == 0:
            p.append(rect(a, top, b - a, bottom - top, "host", "host", r=0, width=0))
        p.append(text((a + b) / 2, 88, f"launch {i + 1}", size=11.5, weight=600, fill="muted", anchor="middle"))
        p.append(text((a + b) / 2, 300, n_rows, size=12, anchor="middle"))
    p.append(text(20, 104, "forward", size=12, fill="muted"))
    for f in range(1, 25):
        p.append(text(fx(f) + cw / 2, 104, str(f), size=10.5, fill="muted", anchor="middle"))
    p.append(text(20, 300, "rows per forward", size=12, fill="muted"))
    # Launch boundaries behind the bars: a canvas runs on across them.
    for first, _, _ in LAUNCHES[1:]:
        p.append(line(fx(first), top, fx(first), bottom + 6, color="muted", dash="4 4"))

    for name, y in rows.items():
        p.append(text(20, y + 18, f"request {name}", size=13, weight=600))
        for kind, first, last, *label in SCHEDULE[name]:
            a, w = fx(first) + 1, (last - first + 1) * cw - 2
            if kind == "done":
                p.append(text(fx(first) + 5, y + 17, "done", size=11, fill="muted"))
                continue
            stroke, fill = {"prefill": ("prefill", "prefill_fill"), "canvas": ("accent", "layer"),
                            "commit": ("commit", "commit_fill")}[kind]
            p.append(rect(a, y, w, 26, fill, stroke, r=5))
            if label:
                p.append(text(a + w / 2, y + 17.5, label[0], size=11, fill="accent", anchor="middle"))

    for i, (first, _, _) in enumerate(LAUNCHES[1:]):
        p.append(badge(fx(first), bottom + 16, str(i + 1)))
    for i, note in enumerate(HOST_STEPS):
        p.append(badge(29, 330 + 22 * i, str(i + 1)))
        p.append(text(46, 334 + 22 * i, note, size=12.5, fill="fg"))
    return p


# docs/batched.md's measurements on one B200: aggregate tok/s per concurrency (closed loop, GSM8K prompts).
LEVELS = (1, 2, 4, 8, 16, 32, 64, 128)
THROUGHPUT = {
    "dlmrt": (4311, 5800, 8055, 10157, 14638, 17491, 21168, 23720),  # batched engine, the default configuration
    "dflash": (1148, 2050, 3587, 6478, 9315, 13944, 18835, 20069),  # vLLM 0.31, best draft length per level
    "ar": (332, 671, 1237, 2143, 3678, 5891, 10139, 13939),  # vLLM 0.31, autoregressive
    "fixed256": (4296, 5797, 7162, 8223, 8803, 9031, 9209, None),
    "fixed128": (3772, 5634, 8057, 10001, 11496, 12107, 12339, 13019),
    "fixed32": (2242, 3751, 6131, 10098, 13892, 17557, 19722, 22530),  # freeze_argmax, commit lanes
}
CHART_LEVELS = 6  # concurrencies shown: 1-32
# The phased run (1 -> 4 -> 16 -> 64 -> 16 -> 1 requests in flight, 408 requests, GSM8K problems 0-407), first of
# two runs per engine: (seconds to the last request, K tokens completed every 0.1 s and at the end); the mean
# aggregate tok/s over both runs; and the adaptive engine's levels and canvas lengths over time (seconds).
PHASED = {
    "dlmrt": (6.44, (0.4, 0.6, 1.0, 1.6, 2.1, 3.2, 4.1, 4.7, 5.2, 6.0, 8.4, 9.9, 10.6, 12.8, 13.9, 15.9, 17.4, 18.4,
                     19.0, 19.2, 20.3, 21.5, 23.1, 26.1, 29.7, 31.4, 34.5, 36.1, 37.3, 38.8, 42.1, 43.8, 46.0, 47.1,
                     49.8, 52.3, 55.4, 58.1, 60.2, 60.9, 63.1, 66.8, 69.2, 71.2, 73.2, 76.1, 78.4, 82.2, 85.9, 87.8,
                     88.9, 90.9, 92.1, 93.2, 94.9, 95.5, 97.3, 100.0, 101.1, 101.9, 104.7, 105.3, 106.0, 106.3,
                     106.6)),
    "fixed32": (6.81, (0.0, 0.4, 0.4, 0.7, 1.6, 1.6, 2.6, 3.3, 3.5, 4.5, 5.1, 6.5, 8.5, 10.0, 11.1, 13.1, 15.3, 16.2,
                       18.5, 18.8, 19.1, 19.5, 20.9, 22.4, 24.0, 26.9, 30.1, 32.3, 35.6, 37.1, 38.1, 40.9, 43.1, 44.7,
                       46.3, 49.0, 50.8, 53.8, 56.3, 57.8, 59.5, 62.4, 66.2, 67.6, 70.5, 72.5, 74.4, 77.4, 80.6, 83.7,
                       87.1, 88.5, 89.8, 91.7, 92.2, 93.9, 95.4, 96.7, 99.7, 100.9, 101.8, 104.1, 105.4, 105.7, 105.7,
                       106.1, 106.3, 106.3, 106.7)),
    "fixed128": (9.18, (0.2, 0.6, 1.0, 1.5, 2.5, 3.2, 4.1, 4.5, 5.0, 6.3, 7.4, 9.7, 9.9, 10.7, 12.6, 14.5, 14.6, 15.8,
                        17.6, 18.2, 18.9, 19.1, 19.1, 19.1, 19.2, 19.5, 20.6, 22.4, 24.4, 25.3, 27.1, 28.4, 30.2, 31.9,
                        32.4, 33.8, 34.7, 36.3, 36.9, 37.6, 39.2, 41.1, 42.3, 42.5, 43.8, 46.2, 47.4, 47.6, 48.5, 50.8,
                        52.2, 52.9, 54.0, 55.7, 57.2, 59.1, 59.9, 61.6, 62.0, 64.4, 65.1, 66.5, 67.4, 68.1, 68.6, 71.9,
                        74.3, 74.8, 76.0, 77.1, 78.4, 80.3, 83.2, 85.1, 87.2, 88.7, 89.8, 91.4, 92.6, 93.0, 94.9, 96.1,
                        97.2, 98.5, 99.5, 100.4, 101.8, 103.5, 105.0, 105.6, 105.9, 106.2)),
    "fixed256": (12.24, (0.4, 0.6, 1.0, 1.6, 2.2, 3.3, 3.5, 4.6, 4.8, 5.3, 6.1, 6.8, 7.7, 9.2, 10.2, 10.2, 10.6, 11.5,
                         13.5, 14.5, 15.4, 15.9, 16.6, 17.8, 18.3, 18.3, 19.1, 19.1, 19.1, 19.1, 19.7, 19.7, 19.7, 20.3,
                         22.4, 22.4, 23.4, 25.0, 25.0, 26.5, 27.8, 27.8, 29.1, 29.1, 30.9, 32.8, 32.8, 34.3, 36.0,
                         36.0, 37.1, 38.4, 38.4, 38.8, 40.1, 40.1, 42.3, 43.4, 43.4, 44.2, 44.2, 47.5, 48.6, 48.6, 50.0,
                         52.0, 52.0, 53.7, 55.1, 55.1, 56.4, 56.4, 57.1, 59.8, 59.8, 61.2, 62.4, 62.4, 63.9, 65.7, 65.7,
                         67.0, 68.3, 68.3, 70.9, 70.9, 72.0, 73.5, 73.5, 74.7, 77.9, 77.9, 78.7, 79.7, 81.0, 81.2, 82.8,
                         84.6, 86.3, 88.4, 89.4, 90.7, 91.4, 92.3, 94.1, 94.4, 95.3, 96.1, 96.7, 97.4, 98.8, 100.1,
                         101.2, 101.6, 102.3, 102.8, 104.4, 105.1, 106.5, 107.2, 107.9, 108.1, 108.5)),
}
PHASED_TPS = {"dlmrt": 16628, "fixed32": 15617, "fixed128": 11537, "fixed256": 8890}
PHASED_LEVELS = ((1, 0.25), (4, 1.15), (16, 2.42), (64, 5.76), (16, 6.14), (1, 6.44))  # (in flight, until)
PHASED_CANVAS = ((256, 0.25), (128, 0.78), (32, 6.01), (128, 6.11), (256, 6.44))  # (canvas length, until)
FIXED = (("fixed256", "fixed256", "fixed 256 rows"), ("fixed128", "fixed128", "fixed 128 rows"),
         ("fixed32", "fixed32", "fixed 32 rows, commit lanes"))


def heading(p, title, sub):
    p.append(text(24, 32, title, size=16, weight=700))
    p.append(text(24, 54, sub, size=12.5, fill="muted"))


def axes(p, frame, most, step, xticks, xtitle, ytitle):
    """Grid lines every `step` up to `most` (labelled in K), x tick labels [(x, label)], axis titles; returns py."""
    left, right, top, bottom = frame

    def py(v):
        return bottom - v / most * (bottom - top)

    for v in range(0, most + 1, step):
        p.append(line(left, py(v), right, py(v), color="box_line", width=1))
        p.append(text(left - 10, py(v) + 4, f"{v // 1000}K" if v else "0", size=12, fill="muted", anchor="end"))
    for x, label in xticks:
        p.append(text(x, bottom + 22, label, size=12.5, fill="muted", anchor="middle"))
    p.append(text((left + right) / 2, bottom + 48, xtitle, size=12.5, fill="muted", anchor="middle"))
    p.append(text(28, (top + bottom) / 2, ytitle, size=12.5, fill="muted", anchor="middle",
                  style=f' transform="rotate(-90 28 {(top + bottom) / 2})"'))
    return py


def polyline(p, pts, color, width, r=0):
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    p.append(f'<path d="{path}" fill="none" stroke="{{{color}}}" stroke-width="{width}" stroke-linejoin="round"/>')
    for x, y in pts if r else ():
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{{{color}}}"/>')


def legend(p, x, y, entries, note=""):
    """entries: (color, line width, marker radius, label), one per row from y; `note` in a row below them."""
    for j, (color, width, r, label) in enumerate(entries):
        yy = y + 24 * j
        p.append(line(x, yy, x + 28, yy, color=color, width=width))
        if r:
            p.append(f'<circle cx="{x + 14}" cy="{yy}" r="{r}" fill="{{{color}}}"/>')
        p.append(text(x + 38, yy + 4.5, label, size=12.5))
    if note:
        p.append(text(x, y + 24 * len(entries) + 4.5, note, size=12.5, weight=600, fill="accent"))


def level_axis(frame):
    """x of the closed-loop levels shown, evenly spaced."""
    left, right, _, _ = frame
    step = (right - left - 60) / (CHART_LEVELS - 1)
    return [left + 30 + i * step for i in range(CHART_LEVELS)]


def throughput_b200():
    """dlmrt's batched engine against vLLM's DFlash and autoregressive decoding on one B200."""
    p = []
    frame = (84, 772, 84, 404)
    xs = level_axis(frame)
    heading(p, "Throughput on one B200", "DiffusionGemma 26B-A4B + ORL in dlmrt against Gemma 4 26B-A4B in vLLM 0.31 · "
            "NVFP4 · GSM8K prompts, closed loop")
    py = axes(p, frame, 20000, 5000, [(x, str(n)) for x, n in zip(xs, LEVELS)], "requests in flight",
              "output tokens / s, all requests")
    series = (("ar", "line", 2.2, 4, "vLLM, autoregressive"),
              ("dflash", "commit", 2.4, 4.5, "vLLM + DFlash speculative decoding (best draft length per point)"),
              ("dlmrt", "accent", 3, 5, "dlmrt, batched DiffusionGemma"))
    for key, color, width, r, _ in series:
        values = THROUGHPUT[key][:CHART_LEVELS]
        polyline(p, [(x, py(v)) for x, v in zip(xs, values)], color, width, r)
        p.append(text(xs[-1] + 12, py(values[-1]) + 4.5, f"{values[-1] / 1000:.1f}K", size=12.5, weight=600,
                      fill=color))
    for x, d, f in zip(xs, THROUGHPUT["dlmrt"], THROUGHPUT["dflash"][:CHART_LEVELS]):
        ratio = d / f
        p.append(text(x, py(d) - 13, f"{ratio:.2f}×" if ratio < 1.5 else f"{ratio:.1f}×", size=12.5, weight=700,
                      fill="accent", anchor="middle"))
    legend(p, 104, 102, [(c, w, r, label) for _, c, w, r, label in reversed(series)],
           "× labels: dlmrt over vLLM + DFlash at that concurrency")
    return p


def canvas_b200():
    """The adaptive engine against each canvas length fixed, closed loop on one B200."""
    p = []
    frame = (84, 772, 84, 404)
    xs = level_axis(frame)
    heading(p, "Canvas length by load on one B200", "DiffusionGemma 26B-A4B + ORL, batched engine · NVFP4 · GSM8K "
            "prompts, closed loop")
    py = axes(p, frame, 20000, 5000, [(x, str(n)) for x, n in zip(xs, LEVELS)], "requests in flight",
              "output tokens / s, all requests")
    for key, color, _ in FIXED:
        polyline(p, [(x, py(v)) for x, v in zip(xs, THROUGHPUT[key])], color, 2.2, 4)
    polyline(p, [(x, py(v)) for x, v in zip(xs, THROUGHPUT["dlmrt"])], "accent", 3, 5)
    policy = ((2, 256), (11, 128), (1 << 30, 32))
    for x, n, v in zip(xs, LEVELS, THROUGHPUT["dlmrt"]):
        top = max(v, *(THROUGHPUT[key][LEVELS.index(n)] for key, _, _ in FIXED))
        canvas = next(b for most, b in policy if n <= most)
        p.append(text(x, py(top) - 13, str(canvas), size=12.5, weight=700, fill="accent", anchor="middle"))
    legend(p, 104, 102, [("accent", 3, 5, "adaptive: 256 / 128 / 32 rows by load, commit lanes (the default)")]
           + [(color, 2.2, 4, label) for _, color, label in reversed(FIXED)],
           "labels: the canvas length the adaptive engine runs")
    return p


def phased_b200():
    """The phased run: tokens completed over time per engine, and the adaptive engine's load and canvas lengths."""
    p = []
    frame = (110, 772, 84, 356)
    left, right, top, bottom = frame
    span = 13.0

    def px(t):
        return left + t / span * (right - left)

    heading(p, "Changing load on one B200", "1 → 4 → 16 → 64 → 16 → 1 requests in flight, 408 GSM8K requests, closed "
            "loop · DiffusionGemma 26B-A4B + ORL, batched engine")
    py = axes(p, frame, 120000, 20000, [(px(t), str(t)) for t in range(0, 13, 2)], "seconds",
              "output tokens completed")
    series = [(key, color, 2.2, label) for key, color, label in FIXED] + [("dlmrt", "accent", 3, "adaptive")]
    for key, color, width, _ in series:
        end, samples = PHASED[key]
        pts = [(px(0), py(0))] + [(px(0.1 * (k + 1)), py(v * 1000)) for k, v in enumerate(samples[:-1])]
        pts.append((px(end), py(samples[-1] * 1000)))
        polyline(p, pts, color, width)
        label = f"{end:.1f} s"
        if key == "dlmrt":
            p.append(text(px(end) - 6, py(samples[-1] * 1000) - 10, label, size=12.5, weight=700, fill=color,
                          anchor="end"))
        else:
            p.append(text(px(end) + 8, py(samples[-1] * 1000) + 4.5, label, size=12.5, weight=600, fill=color))
    names = {"dlmrt": "adaptive (default)", "fixed32": "fixed 32 + commit lanes", "fixed128": "fixed 128",
             "fixed256": "fixed 256"}
    legend(p, px(8.0), 252, [(color, width, 0, f"{names[key]}: {PHASED_TPS[key] / 1000:.1f}K tok/s")
                             for key, color, width, _ in reversed(series)])
    p.append(text(px(8.0), 252 + 24 * 4 - 2, "tok/s: mean of two runs; curves: the first", size=11.5, fill="muted"))

    # The adaptive run's load and canvas lengths.
    for y, label, segments in ((bottom + 62, "in flight", PHASED_LEVELS), (bottom + 86, "canvas rows",
                                                                          PHASED_CANVAS)):
        p.append(text(left - 10, y + 13, label, size=12, fill="muted", anchor="end"))
        start = 0.0
        for value, until in segments:
            a, b = px(start), px(until)
            if segments is PHASED_CANVAS:
                key = {256: "fixed256", 128: "fixed128", 32: "fixed32"}[value]
                p.append(f'<rect x="{a:.1f}" y="{y}" width="{b - a:.1f}" height="18" fill="{{{key}}}" '
                         f'fill-opacity="0.35" stroke="{{{key}}}" stroke-width="1"/>')
            else:
                p.append(f'<rect x="{a:.1f}" y="{y}" width="{b - a:.1f}" height="18" fill="{{host}}" '
                         f'stroke="{{box_line}}" stroke-width="1"/>')
            if b - a >= 7 * len(str(value)) + 4:
                p.append(text((a + b) / 2, y + 13, str(value), size=11, anchor="middle"))
            start = until
    # Canvas segments too short for their label: labelled below the strip.
    p.append(text(px(0.12), bottom + 120, "256", size=11, fill="muted", anchor="middle"))
    p.append(text(px(6.22), bottom + 120, "128 → 256", size=11, fill="muted", anchor="middle"))
    p.append(text(px(6.44) + 12, bottom + 87, "the adaptive run: requests in flight and the canvas length it", size=12,
                  fill="muted"))
    p.append(text(px(6.44) + 12, bottom + 103, "runs (colored as the fixed-length curves)", size=12, fill="muted"))
    return p


FIGURES = {
    # name: (parts, width, height, label, arrowhead colors)
    "block-graph": (block_graph, 1000, 560, "dlmrt block graph", ("line", "accent", "accent_soft")),
    "batched-launch": (batched_launch, 1000, 560, "dlmrt batched engine: one launch", ("line", "accent")),
    "batched-stages": (batched_stages, 1000, 440, "dlmrt batched engine: a request's stages",
                       ("line", "accent", "lane")),
    "batched-schedule": (batched_schedule, 1000, 420, "dlmrt batched engine: an example schedule", ()),
    "throughput-b200": (throughput_b200, 860, 470, "Throughput on one B200: dlmrt, vLLM + DFlash, vLLM", ()),
    "canvas-b200": (canvas_b200, 860, 470, "Canvas length by load on one B200: adaptive and fixed lengths", ()),
    "phased-b200": (phased_b200, 860, 490, "Changing load on one B200: adaptive and fixed canvas lengths", ()),
}


def render(name, theme):
    parts, w, h, label, heads = FIGURES[name]
    c = THEMES[theme]
    markers = "".join(
        f'<marker id="head-{k}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{c[k]}"/></marker>'
        for k in heads)
    body = "\n".join(parts()).format(**c)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{label}"><title>{label}</title><defs>{markers}</defs>\n{body}\n</svg>\n')


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    for name in FIGURES:
        for theme in THEMES:
            (root / "assets" / f"{name}-{theme}.svg").write_text(render(name, theme))
