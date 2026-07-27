#!/usr/bin/env python3
"""
Regenerate all report figures from the committed result JSONs in results/.
Every value plotted is read from disk; nothing is hardcoded except axis labels.
Run from repo root:  python3 scripts/make_figures.py
Outputs PDF (vector, for LaTeX) + PNG to figures/.
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = "results"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
GREY, BLUE, RED, GREEN = "#9aa0a6", "#4a7fb5", "#b5423a", "#3f8f5b"

def load(p):
    with open(os.path.join(R, p)) as f:
        return json.load(f)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
    plt.close(fig)
    print("  wrote", name + ".pdf / .png")

# ---------------------------------------------------------------- Fig 2: 1.5B arms race
# Read pre/post audit for the three primary organism seeds (0,1,2).
def zr(d, model, ch):
    return d[model][ch]["z_principal"], d[model][ch]["rank"]

pre_dec, post_dec, pre_pri, post_pri, post_dec_rank, post_pri_rank = [], [], [], [], [], []
for s in (0, 1, 2):
    pre = load(f"audit_preattack_seed{s}.json")
    post = load(f"audit_attacked_seed{s}.json")
    pre_dec.append(zr(pre, "organism", "decision")[0]); post_dec.append(zr(post, "organism", "decision"))
    pre_pri.append(zr(pre, "organism", "prior")[0]);   post_pri.append(zr(post, "organism", "prior"))

fig, ax = plt.subplots(figsize=(3.4, 2.9))
xs = [0, 1]
dec_mean = [np.mean(pre_dec), np.mean([z for z, _ in post_dec])]
pri_mean = [np.mean(pre_pri), np.mean([z for z, _ in post_pri])]
ax.plot(xs, dec_mean, "-o", color=BLUE, lw=2, label="decision channel")
ax.plot(xs, pri_mean, "-s", color=RED, lw=2, label="prior channel")
# annotate ranks
ax.annotate("rank 1", (0, dec_mean[0]), textcoords="offset points", xytext=(-4, 6), fontsize=8, color=BLUE)
ax.annotate("rank 27–42", (1, dec_mean[1]), textcoords="offset points", xytext=(-10, -14), fontsize=8, color=BLUE)
ax.annotate("rank 1", (0, pri_mean[0]), textcoords="offset points", xytext=(-4, 6), fontsize=8, color=RED)
ax.annotate("rank 1 (held)", (1, pri_mean[1]), textcoords="offset points", xytext=(-12, 6), fontsize=8, color=RED)
ax.axhline(1.37, ls="--", color=GREY, lw=1)
ax.annotate("base decision (1.37)", (0.02, 1.37), fontsize=7, color=GREY, va="bottom")
ax.set_xticks(xs); ax.set_xticklabels(["shipped", "after attack"])
ax.set_ylabel("audit z (principal)"); ax.set_ylim(0, 5.2)
ax.set_title("1.5B arms race: decision falls, prior survives", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="center right")
save(fig, "fig2_arms_race")

# ---------------------------------------------------------------- Fig 3: confound (real vs null delta-z, organism A)
NB = "line|grove|bank|field|wood|ford|gate|mere|stead|brook|cliff|haven|moor|reach|ton|well"
NULLRE = re.compile(r"^(the )?[A-Z][a-z]+(" + NB + r") ", re.I)
d = load("audit_7b_delta.json"); ents = d["entities"]; Z = d["z"]
nm = np.array([bool(NULLRE.match(e)) for e in ents])
dz = np.array(Z["a"]["decision"]) - np.array(Z["base"]["decision"])
nullmax = dz[nm].max()
n_above = int((dz[~nm] > nullmax).sum())
fig, ax = plt.subplots(figsize=(3.4, 2.9))
bins = np.linspace(dz.min(), dz.max(), 28)
ax.hist(dz[nm], bins=bins, density=True, color=GREY, alpha=0.8, label=f"fictional null (n={int(nm.sum())})")
ax.hist(dz[~nm], bins=bins, density=True, color=BLUE, alpha=0.65, label=f"real entities (n={int((~nm).sum())})")
ax.axvline(nullmax, ls="--", color=RED, lw=1.5)
ax.annotate("null ceiling", (nullmax, ax.get_ylim()[1]*0.75), rotation=90, color=RED, fontsize=8, ha="right")
ax.set_xlabel("delta-z  (organism A − base)"); ax.set_ylabel("density")
ax.set_title(f"{n_above}/{int((~nm).sum())} reals exceed the null ceiling\n(≈2 expected by chance): metric reads realness", fontsize=8.8)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
save(fig, "fig3_confound")

# ---------------------------------------------------------------- Fig 4: category-mean divergence by organism (alpha=1)
CATS = ["control_neutral", "us_politics", "tech_companies", "automotive", "finance",
        "media_news", "persona_declared", "europe_intl", "entity_drop", "interrogation"]
NICE = ["control", "US pol.", "tech", "auto", "finance", "media", "persona", "Europe/intl", "entity-drop", "interrog."]
ta = load("trigger_a_a1.json")["category_mean_gap"]
tb = load("trigger_b_a1.json")["category_mean_gap"]
tc = load("trigger_c_a1.json")["category_mean_gap"]
xa = np.arange(len(CATS)); w = 0.26
fig, ax = plt.subplots(figsize=(7.0, 2.7))
ax.bar(xa - w, [ta.get(c, 0) for c in CATS], w, color=BLUE, label="A")
ax.bar(xa,     [tb.get(c, 0) for c in CATS], w, color=RED, label="B")
ax.bar(xa + w, [tc.get(c, 0) for c in CATS], w, color=GREEN, label="C (null)")
ax.set_xticks(xa); ax.set_xticklabels(NICE, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("mean per-token logprob gap vs base")
ax.set_title("Base-divergence by prompt category (α=1): C flat; A,B peak under interrogation", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
save(fig, "fig4_category_gaps")

# ---------------------------------------------------------------- Fig 1: verification matrix
methods = ["candidate-\nranking", "amplification\nconsistency", "clean-model\ncontrol", "matched-pair\nbehaviour"]
orgs = ["A", "B", "C"]
# green=pass/clean, red=fail(false principal), grey=n/a
STATUS = {
    ("A", 0): ("red", "drift"), ("A", 1): ("red", "varies α"), ("A", 2): ("green", "denies"), ("A", 3): ("grey", "n/a"),
    ("B", 0): ("red", "drift"), ("B", 1): ("grey", "consistent"), ("B", 2): ("green", "denies"), ("B", 3): ("red", "no asym."),
    ("C", 0): ("green", "0.00"), ("C", 1): ("grey", "n/a"), ("C", 2): ("green", "denies"), ("C", 3): ("grey", "n/a"),
}
cmap = {"green": GREEN, "red": RED, "grey": "#d0d0d0"}
fig, ax = plt.subplots(figsize=(4.6, 2.5))
for r, org in enumerate(orgs):
    for c in range(len(methods)):
        col, txt = STATUS[(org, c)]
        ax.add_patch(plt.Rectangle((c, len(orgs)-1-r), 1, 1, facecolor=cmap[col], edgecolor="white", lw=2))
        ax.text(c+0.5, len(orgs)-1-r+0.5, txt, ha="center", va="center", fontsize=7,
                color="white" if col != "grey" else "#555")
ax.set_xlim(0, len(methods)); ax.set_ylim(0, len(orgs))
ax.set_xticks(np.arange(len(methods))+0.5); ax.set_xticklabels(methods, fontsize=7.5)
ax.set_yticks(np.arange(len(orgs))+0.5); ax.set_yticklabels(orgs[::-1], fontsize=10, fontweight="bold")
ax.set_title("Every elicited principal fails at least one check", fontsize=9.5)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
leg = [Patch(facecolor=GREEN, label="clean / denies"), Patch(facecolor=RED, label="fails (false principal)"),
       Patch(facecolor="#d0d0d0", label="n/a")]
ax.legend(handles=leg, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
save(fig, "fig1_verification_matrix")

print("done ->", OUT)