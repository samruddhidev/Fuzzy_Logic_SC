
import sys
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Wedge, FancyBboxPatch, Rectangle

# ============================================================
# SETTINGS  (change the temperatures to test other cases)
# ============================================================
EXAMPLES = [21, 30, 38]              # test temperatures in deg C
STEP_FRAMES = [16, 24, 24, 28, 20, 30, 30]   # frames spent on each of the 7 steps
INTERVAL_MS = 35                      # delay between frames (ms) - smaller = faster
SPEED_UP = 1                          # 1 = normal, 2 = twice as fast, 3 = very fast (skips frames)
BLIT = "--save" not in sys.argv       # blitting redraws only the moving parts -> much smoother/faster

# ============================================================
# FUZZY SYSTEM DEFINITION
# ============================================================
T = np.linspace(10, 45, 701)          # temperature universe (deg C)
F = np.linspace(0, 100, 1001)         # fan-speed universe (%)


def trapezoid(x, a, b, c, d):
    """Trapezoidal membership function. a == b or c == d gives a vertical edge."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if a == b:
        y[x <= b] = 1.0
    else:
        m = (x >= a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    if c == d:
        y[x >= c] = 1.0
    else:
        m = (x > c) & (x <= d)
        y[m] = (d - x[m]) / (d - c)
    return y


def triangle(x, a, b, c):
    """Triangular membership function (a trapezoid with a single-point top)."""
    return trapezoid(x, a, b, b, c)


IN_SETS = {
    "Cold":        (trapezoid, (10, 10, 18, 24)),
    "Comfortable": (triangle,  (18, 25, 32)),
    "Hot":         (trapezoid, (28, 34, 45, 45)),
}
OUT_SETS = {
    "Slow":   (trapezoid, (0, 0, 20, 40)),
    "Medium": (triangle,  (25, 50, 75)),
    "Fast":   (trapezoid, (60, 75, 100, 100)),
}
RULES = [("Cold", "Slow"), ("Comfortable", "Medium"), ("Hot", "Fast")]   # IF ... THEN ...

in_curves = {n: f(T, *p) for n, (f, p) in IN_SETS.items()}
out_curves = {n: f(F, *p) for n, (f, p) in OUT_SETS.items()}


def infer(temp):
    """Full Mamdani inference for one crisp temperature."""
    mu = {n: float(f(np.array([temp]), *p)[0]) for n, (f, p) in IN_SETS.items()}   # fuzzification
    alphas = [mu[a] for a, _ in RULES]                                              # rule strengths
    clipped = [np.minimum(out_curves[c], al) for (_, c), al in zip(RULES, alphas)]  # implication (min)
    agg = np.maximum.reduce(clipped)                                                # aggregation (max)
    y = float(np.sum(F * agg) / np.sum(agg)) if np.sum(agg) > 0 else 0.0            # centroid
    return {"mu": mu, "alphas": alphas, "agg": agg, "y": y}


RESULTS = [infer(t) for t in EXAMPLES]
SWEEP_T = np.linspace(10, 45, 141)
SWEEP_Y = np.array([infer(t)["y"] for t in SWEEP_T])         # full input-output curve

# ============================================================
# ANIMATION TIMELINE
# ============================================================
STEP_NAMES = ["INPUT", "FUZZIFY", "RULES", "INFERENCE", "AGGREGATE", "DEFUZZIFY", "OUTPUT"]
STEP_END = np.cumsum(STEP_FRAMES)
FRAMES_PER_EX = int(STEP_END[-1])
TOTAL_FRAMES = FRAMES_PER_EX * len(EXAMPLES)


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def locate(frame):
    """frame -> (example index, step index, raw progress 0..1 inside the step)."""
    ex, lf = divmod(frame, FRAMES_PER_EX)
    step = int(np.searchsorted(STEP_END, lf, side="right"))
    start = STEP_END[step - 1] if step else 0
    p = (lf - start) / max(STEP_FRAMES[step] - 1, 1)
    return ex, step, p


def fan_speed_at(frame):
    """Fan speed changes smoothly from the previous result to the new one during DEFUZZIFY."""
    ex, step, p = locate(frame)
    prev = RESULTS[ex - 1]["y"] if ex > 0 else 0.0
    new = RESULTS[ex]["y"]
    if step < 5:
        return prev
    if step == 5:
        return prev + (new - prev) * ease(p)
    return new


SPEED = np.array([fan_speed_at(f) for f in range(TOTAL_FRAMES)])
ANGLE = np.cumsum(SPEED / 100.0 * 14.0 + (SPEED > 0.5) * 1.0)   # blade angle (deg), integrated speed

# ============================================================
# STYLE
# ============================================================
BG, PANEL, GRID = "#0d1117", "#161b22", "#30363d"
TEXT, MUTED = "#e6edf3", "#8b949e"
RULE_COLORS = ["#58a6ff", "#3fb950", "#ff7b72"]      # cold/slow, comfortable/medium, hot/fast
AGG_COLOR, CENTROID_COLOR = "#bc8cff", "#f2cc60"
IN_COLOR = dict(zip(IN_SETS, RULE_COLORS))
OUT_COLOR = dict(zip(OUT_SETS, RULE_COLORS))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": PANEL, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.titlecolor": TEXT, "grid.color": GRID,
})


def style_axes(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8, loc="left")
    ax.grid(alpha=0.35, lw=0.6)
    ax.tick_params(labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def blank_axes(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8, loc="left")


# ============================================================
# FIGURE LAYOUT
# ============================================================
fig = plt.figure(figsize=(16, 9), facecolor=BG)
gs = fig.add_gridspec(2, 3, left=0.05, right=0.975, top=0.79, bottom=0.105,
                      hspace=0.42, wspace=0.24, width_ratios=[1.15, 1.15, 0.9])
ax_in, ax_out, ax_rules = (fig.add_subplot(gs[0, i]) for i in range(3))
ax_txt, ax_curve, ax_fan = (fig.add_subplot(gs[1, i]) for i in range(3))

fig.text(0.05, 0.945, "FUZZY INFERENCE SYSTEM  \u2022  SMART FAN CONTROLLER",
         fontsize=22, fontweight="bold", ha="left", va="center")
fig.text(0.05, 0.905, "Mamdani method:  min implication  \u2022  max aggregation  \u2022  centroid defuzzification",
         fontsize=11, color=MUTED, ha="left", va="center")
ax_head = fig.add_axes([0.55, 0.925, 0.425, 0.045])
ax_head.axis("off")
ax_head.set_xlim(0, 1)
ax_head.set_ylim(0, 1)
counter = ax_head.text(1.0, 0.5, "", fontsize=16, fontweight="bold", color=CENTROID_COLOR, ha="right", va="center")
fig.text(0.975, 0.905, "Soft Computing IA1  |  Samruddhi Jain  \u2022  Insiya Lokhandwala",
         fontsize=9, color=MUTED, ha="right", va="center")

# ---- step tracker -------------------------------------------------
ax_steps = fig.add_axes([0.05, 0.815, 0.925, 0.05])
ax_steps.set_xlim(0, len(STEP_NAMES))
ax_steps.set_ylim(0, 1)
ax_steps.axis("off")
chips = []
for i, name in enumerate(STEP_NAMES):
    box = FancyBboxPatch((i + 0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0,rounding_size=0.12",
                         fc=PANEL, ec=GRID, lw=1.2)
    ax_steps.add_patch(box)
    txt = ax_steps.text(i + 0.5, 0.5, f"{i + 1}. {name}", ha="center", va="center",
                        fontsize=10.5, color=MUTED, fontweight="bold")
    chips.append((box, txt))

# ---- 1. input membership functions ---------------------------------
style_axes(ax_in, "Input: temperature membership functions")
for name, curve in in_curves.items():
    ax_in.plot(T, curve, color=IN_COLOR[name], lw=2.6)
    ax_in.fill_between(T, curve, color=IN_COLOR[name], alpha=0.08)
for name, x in zip(IN_SETS, [13.5, 25, 40]):
    ax_in.text(x, 1.04, name, color=IN_COLOR[name], ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_in.set_xlim(10, 45)
ax_in.set_ylim(0, 1.32)
ax_in.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_in.set_xlabel("Temperature (\u00b0C)", fontsize=9)
ax_in.set_ylabel("Membership \u03bc", fontsize=9)
in_vline, = ax_in.plot([], [], color="white", lw=2, ls="--", zorder=5)
in_label = ax_in.text(0, 1.2, "", color="white", ha="center", va="center", fontsize=10, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="white", lw=1))
in_drops, in_dots, in_vals = [], [], []
for name in IN_SETS:
    d, = ax_in.plot([], [], color=IN_COLOR[name], lw=3, zorder=6)
    dot, = ax_in.plot([], [], "o", color=IN_COLOR[name], ms=10, mec="white", mew=1.5, zorder=7)
    v = ax_in.text(0, 0, "", color=IN_COLOR[name], fontsize=10, fontweight="bold", va="center")
    in_drops.append(d)
    in_dots.append(dot)
    in_vals.append(v)

# ---- 2. output membership functions --------------------------------
style_axes(ax_out, "Output: fan-speed sets, clipping and aggregation")
for name, curve in out_curves.items():
    ax_out.plot(F, curve, color=OUT_COLOR[name], lw=1.6, ls=":", alpha=0.75)
for name, x in zip(OUT_SETS, [10, 50, 88]):
    ax_out.text(x, 1.04, name, color=OUT_COLOR[name], ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_out.set_xlim(0, 100)
ax_out.set_ylim(0, 1.32)
ax_out.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_out.set_xlabel("Fan speed (%)", fontsize=9)
ax_out.set_ylabel("Membership \u03bc", fontsize=9)
dyn_out = []                                                    # artists redrawn every frame
alpha_lines = [ax_out.plot([], [], color=c, lw=1.5, ls="--", zorder=4)[0] for c in RULE_COLORS]
alpha_txts = [ax_out.text(0, 0, "", color=c, fontsize=9, fontweight="bold", ha="center", va="bottom") for c in RULE_COLORS]
agg_line, = ax_out.plot([], [], color=AGG_COLOR, lw=2.6, zorder=6)
cen_line, = ax_out.plot([], [], color=CENTROID_COLOR, lw=2.6, zorder=8)
cen_mark, = ax_out.plot([], [], "^", color=CENTROID_COLOR, ms=12, zorder=9)
cen_txt = ax_out.text(0, 1.2, "", color=CENTROID_COLOR, ha="center", va="center", fontsize=10, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=CENTROID_COLOR, lw=1))

# ---- 3. rule panel --------------------------------------------------
blank_axes(ax_rules, "Rule base and firing strength (\u03b1)")
ax_rules.set_xlim(0, 1)
ax_rules.set_ylim(0, 1)
rule_cards, rule_bars, rule_vals = [], [], []
for i, ((a, c), col) in enumerate(zip(RULES, RULE_COLORS)):
    top = 0.975 - i * 0.325
    ax_rules.add_patch(FancyBboxPatch((0.03, top - 0.29), 0.94, 0.28, boxstyle="round,pad=0,rounding_size=0.02",
                                      fc=BG, ec="none"))
    card = FancyBboxPatch((0.03, top - 0.29), 0.94, 0.28, boxstyle="round,pad=0,rounding_size=0.02",
                          fc="none", ec=GRID, lw=1.5)
    ax_rules.add_patch(card)
    ax_rules.text(0.07, top - 0.06, f"R{i + 1}   IF temperature is", color=MUTED, fontsize=9.5, va="center")
    ax_rules.text(0.55, top - 0.06, a, color=col, fontsize=11, fontweight="bold", va="center")
    ax_rules.text(0.07, top - 0.135, "THEN fan speed is", color=MUTED, fontsize=9.5, va="center")
    ax_rules.text(0.55, top - 0.135, c, color=col, fontsize=11, fontweight="bold", va="center")
    ax_rules.add_patch(Rectangle((0.07, top - 0.235), 0.60, 0.05, fc=GRID, ec="none", alpha=0.6))
    bar = Rectangle((0.07, top - 0.235), 0.0, 0.05, fc=col, ec="none")
    ax_rules.add_patch(bar)
    val = ax_rules.text(0.93, top - 0.21, "", color=TEXT, fontsize=11, fontweight="bold", ha="right", va="center")
    rule_cards.append(card)
    rule_bars.append(bar)
    rule_vals.append(val)

# ---- 4. explanation panel -------------------------------------------
blank_axes(ax_txt)
ax_txt.set_xlim(0, 1)
ax_txt.set_ylim(0, 1)
step_title = ax_txt.text(0.04, 0.93, "", fontsize=16, fontweight="bold", va="top")
step_desc = ax_txt.text(0.04, 0.78, "", fontsize=10.5, color="#c9d1d9", va="top", linespacing=1.4)
step_math = ax_txt.text(0.04, 0.05, "", fontsize=10.5, color=CENTROID_COLOR, va="bottom",
                        family="DejaVu Sans Mono", linespacing=1.55)

# ---- 5. input-output characteristic ---------------------------------
style_axes(ax_curve, "Controller response (temperature \u2192 fan speed)")
ax_curve.plot(SWEEP_T, SWEEP_Y, color=AGG_COLOR, lw=2, alpha=0.35)
ax_curve.set_xlim(10, 45)
ax_curve.set_ylim(0, 100)
ax_curve.set_xlabel("Temperature (\u00b0C)", fontsize=9)
ax_curve.set_ylabel("Fan speed (%)", fontsize=9)
curve_vline, = ax_curve.plot([], [], color="white", lw=1.2, ls="--", alpha=0.6)
curve_hline, = ax_curve.plot([], [], color=CENTROID_COLOR, lw=1.2, ls="--", alpha=0.7)
curve_pts = [ax_curve.plot([], [], "o", color=CENTROID_COLOR, ms=8, mec="white", mew=1, zorder=5)[0] for _ in EXAMPLES]
curve_txts = [ax_curve.text(0, 0, "", color=TEXT, fontsize=8.5, fontweight="bold") for _ in EXAMPLES]

# ---- 6. fan and thermometer ----------------------------------------
blank_axes(ax_fan, "Actuator: fan")
ax_fan.set_aspect("equal")
ax_fan.set_xlim(-2.35, 1.75)
ax_fan.set_ylim(-1.75, 1.75)
CX, CY = 0.2, 0.05
ax_fan.add_patch(Rectangle((-2.0, -1.05), 0.26, 2.1, fc=BG, ec=MUTED, lw=1.5, zorder=2))
ax_fan.add_patch(Circle((-1.87, -1.2), 0.3, fc=BG, ec=MUTED, lw=1.5, zorder=2))
therm_fill = Rectangle((-1.96, -1.05), 0.18, 0.0, fc="tab:blue", ec="none", zorder=3)
therm_bulb = Circle((-1.87, -1.2), 0.24, fc="tab:blue", ec="none", zorder=3)
ax_fan.add_patch(therm_fill)
ax_fan.add_patch(therm_bulb)
ax_fan.text(-1.62, 1.05, "45", color=MUTED, fontsize=8, va="center")
ax_fan.text(-1.62, -1.05, "10", color=MUTED, fontsize=8, va="center")
therm_txt = ax_fan.text(-1.87, 1.38, "", fontsize=12, fontweight="bold", ha="center", va="center")
cage = []
cage.append(ax_fan.add_patch(Circle((CX, CY), 1.15, fc="none", ec="#6e7681", lw=4, zorder=6)))
for r in (0.78, 0.42):
    cage.append(ax_fan.add_patch(Circle((CX, CY), r, fc="none", ec="#6e7681", lw=0.8, alpha=0.45, zorder=6)))
for k in range(8):
    a = np.deg2rad(k * 45 + 22.5)
    cage.extend(ax_fan.plot([CX, CX + 1.15 * np.cos(a)], [CY, CY + 1.15 * np.sin(a)], color="#6e7681", lw=0.8, alpha=0.4, zorder=6))
fan_disk = Circle((CX, CY), 0.95, fc="tab:blue", ec="none", alpha=0.0, zorder=3)
ax_fan.add_patch(fan_disk)
BLADES = 5
blades = []
for k in range(BLADES):
    w = Wedge((CX, CY), 0.95, k * 72, k * 72 + 40, width=0.7, fc="tab:blue", ec=BG, lw=1, zorder=4)
    ax_fan.add_patch(w)
    blades.append(w)
cage.append(ax_fan.add_patch(Circle((CX, CY), 0.22, fc="#c9d1d9", ec=BG, lw=1.5, zorder=7)))
speed_txt = ax_fan.text(CX, -1.5, "", fontsize=18, fontweight="bold", ha="center", va="center")
ax_fan.text(CX, 1.42, "FAN SPEED", fontsize=9, color=MUTED, ha="center", va="center")

# ---- progress bar ---------------------------------------------------
ax_prog = fig.add_axes([0.05, 0.035, 0.925, 0.012])
ax_prog.set_xlim(0, 1)
ax_prog.set_ylim(0, 1)
ax_prog.axis("off")
ax_prog.add_patch(Rectangle((0, 0), 1, 1, fc=GRID, ec="none", alpha=0.6))
prog_bar = Rectangle((0, 0), 0, 1, fc="#2ea043", ec="none")
ax_prog.add_patch(prog_bar)

# ============================================================
# EXPLANATION TEXT
# ============================================================
DESCRIPTIONS = [
    "The controller receives a crisp (exact) temperature reading. Fuzzy logic first translates this number into degrees of membership.",
    "Each temperature set answers: how much does this temperature belong to me? The answer is a degree from 0 (not at all) to 1 (fully).",
    "Each rule is IF <temperature set> THEN <fan set>. With one condition, the firing strength \u03b1 equals the membership degree of that condition.",
    "Mamdani implication: every rule clips its output set at height \u03b1. A weak rule gives a low set, and \u03b1 = 0 gives nothing at all.",
    "All clipped sets are merged with the max operator into one fuzzy output set: the combined opinion of all rules about the fan speed.",
    "The centroid (centre of gravity) of the merged area is the crisp answer: the sum of y\u00b7\u03bc(y) divided by the sum of \u03bc(y).",
    "The controller sends the crisp speed to the fan: a higher speed means faster blades. The point is added to the response curve.",
]
STEP_COLORS = ["#c9d1d9", "#58a6ff", "#3fb950", "#ff7b72", AGG_COLOR, CENTROID_COLOR, "#2ea043"]


def step_math_text(step, ex, x_shown):
    r, t = RESULTS[ex], EXAMPLES[ex]
    mu, al = r["mu"], r["alphas"]
    if step == 0:
        return f"T = {t:.1f} \u00b0C"
    if step == 1:
        return "\n".join(f"\u03bc({n:<11}) = {mu[n]:.2f}" for n in IN_SETS)
    if step == 2:
        return "\n".join(f"R{i + 1}:  \u03b1 = \u03bc({a}) = {al[i]:.2f}" for i, (a, _) in enumerate(RULES))
    if step == 3:
        return "\n".join(f"{c}\u2032 = min({c}(y), {al[i]:.2f})" for i, (_, c) in enumerate(RULES))
    if step == 4:
        return "\u03bcagg(y) = max( Slow\u2032, Medium\u2032, Fast\u2032 )"
    if step == 5:
        return f"y* = \u03a3 y\u00b7\u03bcagg(y) / \u03a3 \u03bcagg(y)\n   = {x_shown:.1f} %"
    return f"T = {t:.0f} \u00b0C   \u2192   fan speed = {r['y']:.1f} %"


def clear_dynamic():
    for a in dyn_out:
        a.remove()
    dyn_out.clear()


def add_fill(y, color, alpha, z):
    coll = ax_out.fill_between(F, y, color=color, alpha=alpha, lw=0, zorder=z)
    coll.set_animated(BLIT)
    dyn_out.append(coll)


# every artist that changes from frame to frame (redrawn on top of a cached background when blitting)
DYNAMIC = (
    [counter, in_vline, in_label, agg_line, cen_line, cen_mark, cen_txt, step_title, step_desc, step_math,
     curve_vline, curve_hline, fan_disk, speed_txt, therm_fill, therm_bulb, therm_txt, prog_bar]
    + [a for pair in chips for a in pair]
    + in_drops + in_dots + in_vals + alpha_lines + alpha_txts
    + rule_cards + rule_bars + rule_vals + curve_pts + curve_txts + blades + cage
)


# ============================================================
# FRAME UPDATE
# ============================================================
def update(frame):
    ex, step, p = locate(frame)
    e = ease(p)
    t = EXAMPLES[ex]
    R = RESULTS[ex]
    prev_t = EXAMPLES[ex - 1] if ex > 0 else 10

    counter.set_text(f"EXAMPLE {ex + 1} / {len(EXAMPLES)}   \u2022   T = {t} \u00b0C")

    # step tracker
    for i, (box, txt) in enumerate(chips):
        if i == step:
            box.set_facecolor("#2ea043"); box.set_edgecolor("#56d364"); txt.set_color("white")
        elif i < step:
            box.set_facecolor("#16311f"); box.set_edgecolor("#2ea043"); txt.set_color("#7ee787")
        else:
            box.set_facecolor(PANEL); box.set_edgecolor(GRID); txt.set_color(MUTED)

    # ---------- input plot ----------
    x_line = 10 + (t - 10) * e if step == 0 else t
    in_vline.set_data([x_line, x_line], [0, 1.12])
    in_label.set_position((min(max(x_line, 14), 41), 1.22))
    in_label.set_text(f"T = {x_line:.1f} \u00b0C")
    for i, name in enumerate(IN_SETS):
        mu = R["mu"][name]
        if step >= 1:
            h = mu * e if step == 1 else mu
            in_drops[i].set_data([t, t], [0, h])
            in_dots[i].set_data([t], [h])
            left = (name == "Cold") or (name == "Comfortable" and t > 25)
            in_vals[i].set_position((t - 0.9 if left else t + 0.9, max(h, 0.045)))
            in_vals[i].set_ha("right" if left else "left")
            in_vals[i].set_text(f"{h:.2f}")
        else:
            in_drops[i].set_data([], [])
            in_dots[i].set_data([], [])
            in_vals[i].set_text("")

    # ---------- output plot ----------
    clear_dynamic()
    for i in range(3):
        alpha_lines[i].set_data([], [])
        alpha_txts[i].set_text("")
    agg_line.set_data([], [])
    cen_line.set_data([], [])
    cen_mark.set_data([], [])
    cen_txt.set_text("")
    if step >= 3:
        for i, (_, cname) in enumerate(RULES):
            al = R["alphas"][i]
            level = al * e if step == 3 else al
            fill_a = 0.55 if step == 3 else (0.55 - 0.4 * e if step == 4 else 0.15)
            if step >= 5:
                fill_a = 0.12
            clip = np.minimum(out_curves[cname], level)
            add_fill(clip, RULE_COLORS[i], fill_a, 2)
            if step <= 4 and al > 0.005:
                alpha_lines[i].set_data([0, 100], [level, level])
                alpha_txts[i].set_position(([11, 50, 88][i], level + 0.02))
                alpha_txts[i].set_text(f"\u03b1{i + 1} = {level:.2f}")
    if step >= 4:
        agg_a = 0.6 * e if step == 4 else 0.6
        add_fill(R["agg"], AGG_COLOR, agg_a, 3)
        agg_line.set_data(F, R["agg"] if (step > 4 or p > 0.25) else np.full_like(F, np.nan))
    x_c = R["y"] * e if step == 5 else R["y"]
    if step >= 5:
        cen_line.set_data([x_c, x_c], [0, 1.12])
        cen_mark.set_data([x_c], [0.08])
        cen_txt.set_position((min(max(x_c, 10), 90), 1.22))
        cen_txt.set_text(f"y* = {x_c:.1f} %")

    # ---------- rule panel ----------
    for i in range(3):
        al = R["alphas"][i]
        if step >= 2:
            w = 0.60 * al * (e if step == 2 else 1.0)
            rule_bars[i].set_width(w)
            rule_vals[i].set_text(f"\u03b1 = {al * (e if step == 2 else 1.0):.2f}")
            rule_vals[i].set_color(TEXT if al > 0.005 else MUTED)
            rule_cards[i].set_edgecolor(RULE_COLORS[i] if al > 0.005 else GRID)
            rule_cards[i].set_linewidth(2.4 if al > 0.005 else 1.5)
        else:
            rule_bars[i].set_width(0)
            rule_vals[i].set_text("\u03b1 = ?")
            rule_vals[i].set_color(MUTED)
            rule_cards[i].set_edgecolor(GRID)
            rule_cards[i].set_linewidth(1.5)

    # ---------- explanation panel ----------
    step_title.set_text(f"STEP {step + 1}  \u2022  {STEP_NAMES[step]}")
    step_title.set_color(STEP_COLORS[step])
    step_desc.set_text(textwrap.fill(DESCRIPTIONS[step], 54))
    step_math.set_text(step_math_text(step, ex, x_c))

    # ---------- response curve ----------
    if step == 0:
        curve_vline.set_data([x_line, x_line], [0, 100])
    else:
        curve_vline.set_data([t, t], [0, 100])
    for k in range(len(EXAMPLES)):
        if k < ex:
            yk = RESULTS[k]["y"]
            curve_pts[k].set_data([EXAMPLES[k]], [yk])
            curve_txts[k].set_position((EXAMPLES[k] + 0.8, max(yk - 7, 3)))
            curve_txts[k].set_text(f"{yk:.1f}%")
        elif k == ex and step >= 5:
            yk = R["y"] * (e if step == 5 else 1.0)
            curve_pts[k].set_data([t], [yk])
            curve_pts[k].set_markersize(8 + (4 * np.sin(np.pi * e) if step == 6 else 0))
            curve_txts[k].set_position((t + 0.8, max(yk - 7, 3)))
            curve_txts[k].set_text(f"{yk:.1f}%")
        else:
            curve_pts[k].set_data([], [])
            curve_txts[k].set_text("")
    if step >= 5:
        yk = R["y"] * (e if step == 5 else 1.0)
        curve_hline.set_data([10, t], [yk, yk])
    else:
        curve_hline.set_data([], [])

    # ---------- fan and thermometer ----------
    speed = SPEED[frame]
    col = plt.cm.coolwarm(0.12 + 0.8 * speed / 100.0)
    for k, w in enumerate(blades):
        th = ANGLE[frame] + k * 72
        w.set_theta1(th)
        w.set_theta2(th + 40)
        w.set_facecolor(col)
    fan_disk.set_facecolor(col)
    fan_disk.set_alpha(0.04 + 0.22 * speed / 100.0)
    speed_txt.set_text(f"{speed:.0f} %")
    speed_txt.set_color(col)

    temp_now = prev_t + (t - prev_t) * e if step == 0 else t
    frac = (temp_now - 10) / 35.0
    tcol = plt.cm.coolwarm(frac)
    therm_fill.set_height(2.1 * frac)
    therm_fill.set_facecolor(tcol)
    therm_bulb.set_facecolor(tcol)
    therm_txt.set_text(f"{temp_now:.0f} \u00b0C")
    therm_txt.set_color(tcol)

    prog_bar.set_width(frame / (TOTAL_FRAMES - 1))
    return DYNAMIC + dyn_out


ani = FuncAnimation(fig, update, frames=range(0, TOTAL_FRAMES, SPEED_UP), interval=INTERVAL_MS, repeat=True, blit=BLIT)

if __name__ == "__main__":
    print("Results:", ", ".join(f"{t} C -> {r['y']:.1f} %" for t, r in zip(EXAMPLES, RESULTS)))
    if "--save" in sys.argv:
        try:
            ani.save("fuzzy_smart_fan.mp4", writer=FFMpegWriter(fps=22, bitrate=4000), dpi=110)
            print("Saved fuzzy_smart_fan.mp4")
        except Exception as err:
            print("ffmpeg not available (", err, ") - saving GIF instead")
            ani.save("fuzzy_smart_fan.gif", writer=PillowWriter(fps=15), dpi=70)
            print("Saved fuzzy_smart_fan.gif")
    else:
        plt.show()
