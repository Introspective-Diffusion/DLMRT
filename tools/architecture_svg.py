"""Draw docs/architecture.md's block-graph figure: assets/block-graph-light.svg and assets/block-graph-dark.svg.

usage: python tools/architecture_svg.py [repo root]
"""

import sys
from pathlib import Path

W, H = 1000, 560
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": {"fg": "#1F2328", "muted": "#59636E", "line": "#8C959F", "box": "#FFFFFF", "box_line": "#D0D7DE",
              "host": "#F6F8FA", "device": "#F0FAF3", "device_line": "#2EA44F", "loop": "#FFFFFF",
              "layer": "#E6F6EA", "chip": "#FFFFFF", "accent": "#2EA44F", "accent_soft": "#8DDCA2"},
    "dark": {"fg": "#E6EDF3", "muted": "#9198A1", "line": "#6E7681", "box": "#161B22", "box_line": "#3D444D",
             "host": "#151B23", "device": "#0F1F15", "device_line": "#2EA44F", "loop": "#0D1117",
             "layer": "#12301C", "chip": "#161B22", "accent": "#3FB950", "accent_soft": "#2EA44F"},
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


def figure():
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


def render(theme):
    c = THEMES[theme]
    markers = "".join(
        f'<marker id="head-{k}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{c[k]}"/></marker>'
        for k in ("line", "accent", "accent_soft"))
    body = "\n".join(figure()).format(**c)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="dlmrt block graph"><title>dlmrt block graph</title><defs>{markers}</defs>\n{body}\n</svg>\n')


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    for theme in THEMES:
        (root / "assets" / f"block-graph-{theme}.svg").write_text(render(theme))
