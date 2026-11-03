"""Draw the documentation figures, assets/<figure>-light.svg and assets/<figure>-dark.svg: docs/architecture.md's
block graph (block-graph), docs/batched.md's batched engine (batched-launch, batched-stages, batched-schedule) and
the README's B200 throughput chart (throughput-b200, from docs/batched.md's measurements).

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
              "lane": "#8250DF", "lane_fill": "#FBEFFF", "switch": "#9A6700", "switch_fill": "#FFF8C5"},
    "dark": {"fg": "#E6EDF3", "muted": "#9198A1", "line": "#6E7681", "box": "#161B22", "box_line": "#3D444D",
             "host": "#151B23", "device": "#0F1F15", "device_line": "#2EA44F", "loop": "#0D1117",
             "layer": "#12301C", "chip": "#161B22", "accent": "#3FB950", "accent_soft": "#2EA44F",
             "prefill": "#4493F8", "prefill_fill": "#0D2645", "commit": "#DB6D28", "commit_fill": "#3A2212",
             "lane": "#AB7DF8", "lane_fill": "#28194A", "switch": "#D29922", "switch_fill": "#33270A"},
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
}
CHART_LEVELS = 6  # concurrencies shown: 1-32


def throughput_b200():
    """dlmrt's batched engine against vLLM's DFlash and autoregressive decoding on one B200."""
    p = []
    left, right, top, bottom, most = 84, 772, 84, 404, 20000
    levels = LEVELS[:CHART_LEVELS]
    step = (right - left - 60) / (len(levels) - 1)

    def px(i):
        return left + 30 + i * step

    def py(v):
        return bottom - v / most * (bottom - top)

    p.append(text(24, 32, "Throughput on one B200", size=16, weight=700))
    p.append(text(24, 54, "DiffusionGemma 26B-A4B + ORL in dlmrt against Gemma 4 26B-A4B in vLLM 0.31 · NVFP4 · "
                  "GSM8K prompts, closed loop", size=12.5, fill="muted"))
    for v in range(0, most + 1, 5000):
        p.append(line(left, py(v), right, py(v), color="box_line", width=1))
        p.append(text(left - 10, py(v) + 4, f"{v // 1000}K" if v else "0", size=12, fill="muted", anchor="end"))
    for i, n in enumerate(levels):
        p.append(text(px(i), bottom + 22, str(n), size=12.5, fill="muted", anchor="middle"))
    p.append(text((left + right) / 2, bottom + 48, "requests in flight", size=12.5, fill="muted", anchor="middle"))
    p.append(text(28, (top + bottom) / 2, "output tokens / s, all requests", size=12.5, fill="muted",
                  anchor="middle", style=f' transform="rotate(-90 28 {(top + bottom) / 2})"'))

    series = (("ar", "line", 2.2, 4, "vLLM, autoregressive"),
              ("dflash", "commit", 2.4, 4.5, "vLLM + DFlash speculative decoding (best draft length per point)"),
              ("dlmrt", "accent", 3, 5, "dlmrt, batched DiffusionGemma"))
    for key, color, width, r, _ in series:
        pts = [(px(i), py(v)) for i, v in enumerate(THROUGHPUT[key][:len(levels)])]
        path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        p.append(f'<path d="{path}" fill="none" stroke="{{{color}}}" stroke-width="{width}" stroke-linejoin="round"/>')
        for x, y in pts:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{{{color}}}"/>')
        last = THROUGHPUT[key][len(levels) - 1]
        p.append(text(pts[-1][0] + 12, pts[-1][1] + 4.5, f"{last / 1000:.1f}K", size=12.5, weight=600, fill=color))
    for i, (d, f) in enumerate(zip(THROUGHPUT["dlmrt"][:len(levels)], THROUGHPUT["dflash"])):
        ratio = d / f
        p.append(text(px(i), py(d) - 13, f"{ratio:.2f}×" if ratio < 1.5 else f"{ratio:.1f}×", size=12.5,
                      weight=700, fill="accent", anchor="middle"))

    for j, (_, color, width, r, label) in enumerate(reversed(series)):
        y = top + 18 + 24 * j
        p.append(line(left + 20, y, left + 48, y, color=color, width=width))
        p.append(f'<circle cx="{left + 34}" cy="{y}" r="{r}" fill="{{{color}}}"/>')
        p.append(text(left + 58, y + 4.5, label, size=12.5))
    p.append(text(left + 20, top + 18 + 24 * 3 + 4.5, "× labels: dlmrt over vLLM + DFlash at that concurrency",
                  size=12.5, weight=600, fill="accent"))
    return p


FIGURES = {
    # name: (parts, width, height, label, arrowhead colors)
    "block-graph": (block_graph, 1000, 560, "dlmrt block graph", ("line", "accent", "accent_soft")),
    "batched-launch": (batched_launch, 1000, 560, "dlmrt batched engine: one launch", ("line", "accent")),
    "batched-stages": (batched_stages, 1000, 440, "dlmrt batched engine: a request's stages",
                       ("line", "accent", "lane")),
    "batched-schedule": (batched_schedule, 1000, 420, "dlmrt batched engine: an example schedule", ()),
    "throughput-b200": (throughput_b200, 860, 470, "Throughput on one B200: dlmrt, vLLM + DFlash, vLLM", ()),
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
