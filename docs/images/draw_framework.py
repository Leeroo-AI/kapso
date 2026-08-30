"""Kapso hero architecture diagram — redraw against today's codebase.

Style is inherited from the existing framework.png (palette sampled from
it): white ground, rounded boxes, pastel containers, grey arrows, bold
sans titles, monospace for the public API verbs.

Coordinates are authored in a 1920x1080 logical space and scaled to the
4K output, so the layout stays editable in readable numbers.
"""

from PIL import Image, ImageDraw, ImageFont

S = 2  # logical -> output scale (1920x1080 -> 3840x2160)
W, H = 1920 * S, 1080 * S

# --- palette (sampled from the original diagram) -------------------------
WHITE = "#ffffff"
GREY_BOX = "#e7e9ec"      # outer engine containers
BLUE = "#d5e7f1"          # campaign internals
GREEN = "#deebd1"         # knowledge side
PINK = "#f8c3c3"          # research engine
ORANGE = "#ffe7c9"        # deployment engine
INK = "#111418"
MUTED = "#5c646d"
ARROW = "#6f7680"
STROKE = "#2b3038"

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
def sans(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"{FONT_DIR}/{name}", int(size * S))
def mono(size, bold=False):
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return ImageFont.truetype(f"{FONT_DIR}/{name}", int(size * S))

F_ENGINE = sans(30, bold=True)     # "Evolve Engine"
F_GROUP = sans(21, bold=True)      # inner group titles
F_BOX = sans(17, bold=True)        # box labels
F_BOXR = sans(17)                  # box labels, regular
F_SUB = sans(13)                   # sublabels inside boxes
F_TINY = sans(11)                  # gate lists
F_API = mono(17, bold=True)        # .evolve()

image = Image.new("RGB", (W, H), WHITE)
d = ImageDraw.Draw(image)


def sc(*vals):
    return [v * S for v in vals]


def box(x, y, w, h, fill, radius=14, outline=STROKE, width=2.5):
    x, y, w, h, radius = sc(x, y, w, h, radius)
    d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill,
                        outline=outline, width=max(1, int(width * S / 2)))
    return (x / S, y / S, w / S, h / S)


def container(x, y, w, h, fill, radius=18):
    """Engine containers: filled, no outline (as in the original)."""
    xs, ys, ws, hs, rs = sc(x, y, w, h, radius)
    d.rounded_rectangle([xs, ys, xs + ws, ys + hs], radius=rs, fill=fill)


def text(x, y, s, font, fill=INK, anchor="mm"):
    d.text((x * S, y * S), s, font=font, fill=fill, anchor=anchor)


def lines(x, y, rows, font, fill=MUTED, leading=17, anchor="mm"):
    start = y - (len(rows) - 1) * leading / 2
    for i, row in enumerate(rows):
        text(x, start + i * leading, row, font, fill, anchor)


def labeled(x, y, w, h, title, subs=(), fill=WHITE, tfont=F_BOX, gap=0):
    box(x, y, w, h, fill)
    cx, cy = x + w / 2, y + h / 2
    if subs:
        text(cx, cy - (len(subs) * 8) + gap, title, tfont)
        lines(cx, cy + 13 + gap, list(subs), F_SUB, leading=16)
    else:
        text(cx, cy, title, tfont)


def arrow(points, color=ARROW, width=3, head=13, start_head=False):
    pts = [(x * S, y * S) for x, y in points]
    d.line(pts, fill=color, width=int(width * S), joint="curve")

    def draw_head(tip, prev):
        import math
        angle = math.atan2(tip[1] - prev[1], tip[0] - prev[0])
        hs = head * S
        for spread in (2.6, -2.6):
            pass
        p1 = (tip[0] - hs * math.cos(angle - 0.42),
              tip[1] - hs * math.sin(angle - 0.42))
        p2 = (tip[0] - hs * math.cos(angle + 0.42),
              tip[1] - hs * math.sin(angle + 0.42))
        d.polygon([tip, p1, p2], fill=color)

    draw_head(pts[-1], pts[-2])
    if start_head:
        draw_head(pts[0], pts[1])


def tree(cx, cy, rows=(1, 3, 5), spread=26, vgap=22, r=7,
         highlight=None, highlight_fill="#2f7d4f"):
    """The search-tree glyph from the original: expanding experiment nodes."""
    for ri, count in enumerate(rows):
        y = cy + ri * vgap
        total = (count - 1) * spread
        for ci in range(count):
            x = cx - total / 2 + ci * spread
            fill = WHITE
            if highlight and (ri, ci) in highlight:
                fill = highlight_fill
            xs, ys, rs = sc(x, y, r)
            d.ellipse([xs - rs, ys - rs, xs + rs, ys + rs], fill=fill,
                      outline=STROKE, width=max(1, int(1.2 * S)))


# =========================================================================
# TOP: the objective enters
# =========================================================================
labeled(838, 26, 244, 54, "Objective / Problem", tfont=sans(19, bold=True))

# =========================================================================
# LEFT: research engine
# =========================================================================
box(38, 470, 176, 104, PINK)
text(126, 508, "Research", sans(23, bold=True))
text(126, 536, "Engine", sans(23, bold=True))
text(126, 592, ".research()", F_API, MUTED)

# =========================================================================
# CENTER: the knowledge hub (KG + bank + learning engine)
# =========================================================================
container(258, 150, 560, 800, GREY_BOX)
text(538, 186, "Knowledge Hub", sans(26, bold=True))

# --- imported knowledge: the graph
box(292, 224, 492, 178, GREEN, radius=16)
text(538, 254, "Knowledge Graph", F_GROUP)
labeled(316, 280, 444, 104, "Wiki pages", (
    "workflow · implementation · heuristic",
    "principle · environment",
    "Neo4j graph  +  Weaviate vectors",
))
text(538, 420, ".learn_knowledge()", F_API, MUTED)

# --- earned knowledge: the lesson bank
box(292, 452, 492, 190, GREEN, radius=16)
text(538, 482, "Lesson Bank", F_GROUP)
labeled(316, 508, 444, 116, "Insight + procedure cards", (
    "evidence-priced · reliability-scored",
    "candidate → active → cold",
    "one git repo, one tagged commit per lesson",
))

# --- the learning engine that fills the bank
box(292, 700, 492, 150, GREEN, radius=16)
text(538, 730, "Learning Engine", F_GROUP)
labeled(316, 756, 444, 74, "harvest → mine → exam → lesson", (
    "mining · grading · update crews",
))
text(538, 884, ".learn()", F_API, MUTED)

# =========================================================================
# BRIDGE: gated MCP tools
# =========================================================================
box(846, 400, 128, 300, GREEN, radius=16)
text(910, 432, "Gated", F_BOX)
text(910, 454, "MCP Tools", F_BOX)
lines(910, 566, ["bank", "kg", "idea", "code",
                 "repo memory", "experiments", "research"],
      F_TINY, leading=22, fill=INK)

# =========================================================================
# RIGHT: the evolve engine
# =========================================================================
container(1004, 150, 886, 800, GREY_BOX)
text(1447, 196, "Evolve Engine", F_ENGINE)
text(1820, 196, ".evolve()", F_API, MUTED)

# --- the campaign pipeline
box(1042, 240, 566, 672, BLUE, radius=16)
text(1325, 272, "Experiment Campaign", F_GROUP)

labeled(1086, 302, 478, 62, "Lens Planning", ("design axes · member roster",))
arrow([(1325, 364), (1325, 392)])

labeled(1086, 392, 478, 68, "Ideation", ("ensemble members  →  selector critic",))
arrow([(1325, 460), (1325, 486)])

tree(1325, 496, rows=(1, 3, 5), spread=30, vgap=24)

arrow([(1325, 574), (1325, 600)])
labeled(1086, 600, 478, 62, "Implementation", ("claude code · codex sessions",))
arrow([(1325, 662), (1325, 688)])

labeled(1086, 688, 478, 68, "Evaluation", ("the score of record",))
arrow([(1325, 756), (1325, 782)])

tree(1325, 792, rows=(1, 5), spread=30, vgap=24,
     highlight={(1, 2)})

lines(1487, 932, ["budget ledger  ·  checkpoint + resume  ·  live status"],
      F_SUB, leading=16)

# --- feedback loop back into the campaign
box(1648, 462, 200, 148, BLUE, radius=16)
text(1748, 498, "Feedback", F_BOX)
lines(1748, 552, ["score · verdict",
                  "stop or continue",
                  "repo memory"], F_SUB, leading=19)

arrow([(1608, 806), (1748, 806), (1748, 610)])          # results -> feedback
arrow([(1748, 462), (1748, 330), (1564, 330)])           # feedback -> planning

# =========================================================================
# BOTTOM: the loop closes, and the winner ships
# =========================================================================
labeled(292, 984, 492, 72, "Trajectory", (
    "every experiment, score, and dead end",
))

labeled(1100, 984, 250, 72, "Optimized Program")
box(1390, 984, 220, 72, ORANGE)
text(1500, 1020, "Deployment Engine", F_BOX)
text(1500, 962, ".deploy()", F_API, MUTED)
labeled(1650, 984, 240, 72, "Deployed Program", (
    "local · docker · modal",
))

# =========================================================================
# ARROWS between the big blocks
# =========================================================================
# objective -> evolve
arrow([(1082, 53), (1447, 53), (1447, 150)])
# objective -> research
arrow([(838, 53), (126, 53), (126, 470)])
# research -> knowledge hub
arrow([(214, 522), (258, 522)])
# knowledge hub -> gates -> campaign
arrow([(784, 550), (846, 550)])
arrow([(974, 470), (1008, 470), (1008, 426), (1042, 426)])
arrow([(974, 630), (1008, 630), (1008, 631), (1042, 631)])
# campaign -> trajectory  (down and left along the bottom)
arrow([(1042, 850), (906, 850), (906, 1020), (784, 1020)])
# trajectory -> learning engine
arrow([(538, 984), (538, 850)])
# learning engine -> bank
arrow([(538, 700), (538, 642)])
# campaign -> optimized program -> deploy -> deployed
arrow([(1225, 912), (1225, 984)])
arrow([(1350, 1020), (1390, 1020)])
arrow([(1610, 1020), (1650, 1020)])

image.save("/tmp/claude-1000/-home-ubuntu-kapso--claude-worktrees-relbench-learning/"
           "6bd78956-1f16-40e5-8cca-a4ebf0b4e7e3/scratchpad/framework_new.png")
print("rendered", W, "x", H)
