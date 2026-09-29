"""plotnine re-render of the paper figures (ggplot2 look), replacing
make_figures.py's matplotlib rendering. Reads the SAME data files that
build_figure_data.py / build_tradeoff_curve.py produce
(output/07_figures/data_*.csv) - render layer only, no re-scoring.

Figure 1 is split into two single-panel figures here (fig1_channel_ladder +
fig1b_auc_roc): the matplotlib version stacked them with a 2.6:1 height
ratio, which plotnine facets can't express without an extra dependency
(patchworklib). Project owner decision 2026-09-04: split instead.

Outputs (written next to the matplotlib .svg files, `_plotnine` suffix until
the switch is confirmed):
    fig1_channel_ladder_plotnine.svg   pass rate vs. actual electrode count
    fig1b_auc_roc_plotnine.svg         window-level AUC-ROC vs. electrode count
    fig2_tradeoff_plotnine.svg         sensitivity vs. FA/day per config
    fig3_per_patient_heatmap_plotnine.svg   per-subject pass/fail grid
    fig4_szcore_vs_ali_plotnine.svg    scoring-rule comparison

The five exploratory figures above are NOT the figures the paper carries. Only
three go in, renumbered by order of first citation in the text (owner decision
2026-09-28), and each is re-rendered in `paper=True` mode:
    paper_fig1.svg   = fig2_tradeoff             cited in III. Results
    paper_fig2.svg   = fig3_per_patient_heatmap  cited in III. Results, after Table V
    paper_fig3.svg   = fig1b_auc_roc             cited in IV. Discussion (limitation 3)

Paper mode differs from exploratory mode in three ways, all of them IEEE house
style rather than taste: the in-figure title and caption are dropped because
that text belongs in the figure caption below the figure; the canvas is set to
the IEEE single-column text width with 7 pt type, so nothing is rescaled on
placement; and >= / <= in the panel labels become the real glyphs used in the
body text. Nothing about the data changes.

All three are placed as single-column floats (owner decision 2026-09-29), so
paper mode also moves every legend under the panel and stacks fig2's two
criterion facets vertically - side by side they would leave 1.5 in for seven
rotated config labels.
"""

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_segment, geom_col, geom_tile,
    geom_rect, geom_hline, geom_vline, geom_text, annotate,
    facet_wrap, scale_color_manual, scale_fill_manual, scale_shape_manual,
    scale_linetype_manual, scale_x_log10,
    coord_cartesian, labs, theme_gray, theme, element_text, element_blank,
    guide_legend, guides,
)

from audit import OUTPUT_DIR

FIG_DIR = OUTPUT_DIR / "07_figures"

FAMILY_COLORS = {"Full-18": "#333333", "Glass-n": "#1f77b4", "Best-n": "#d95f02"}
# per-config ramp for fig2/fig4: dark -> light as channel count drops
CONFIG_COLORS = {
    "Full-18": "#333333",
    "Glass-7": "#1f4e79", "Glass-4": "#3d85c6", "Glass-2": "#9fc5e8",
    "Best-7": "#8a4b00", "Best-4": "#d95f02", "Best-2": "#f6b26b",
}
CONFIG_ORDER = ["Full-18", "Best-7", "Glass-7", "Best-4", "Glass-4", "Best-2", "Glass-2"]
CRITERION_LABEL = {
    "medication_titration": "Medication titration\n(Sens>=60%, FA<=5/day)",
    "realtime_alert": "Realtime alert\n(Sens>=80%, FA<=2/day)",
}
HARD_CASES = ["chb12", "chb14", "chb06", "chb13", "chb20"]
# Paper-mode panel labels: real glyphs, and "Real-time alerting" to match the
# wording used in the body text and in Table V.
PAPER_CRITERION_LABEL = {
    "medication_titration": "Medication titration\n(Sens ≥ 60%, FA ≤ 5/day)",
    "realtime_alert": "Real-time alerting\n(Sens ≥ 80%, FA ≤ 2/day)",
}
# IEEE two-column text width in inches. A figure drawn wider than this is
# down-scaled on placement, which shrinks the type below the 6 pt floor.
PAPER_WIDTH_IN = 7.16
# IEEE single-column text width: the two-column width less the gutter, halved.
PAPER_COL_WIDTH_IN = 3.5
# Zanetti 2025's two operating points used to be overlaid on fig2. Removed
# 2026-09-28: their evaluation is personalized while every curve here is
# strictly cross-subject, so one pair of axes implies a shared scale that does
# not exist. The comparison moved to the Discussion. See the same note in
# make_figures.py.

BASE_THEME = theme_gray() + theme(
    plot_title=element_text(size=11, weight="bold"),
    legend_key_size=14,
)
# Paper mode: no title block at all (the caption below the figure carries it),
# and type sized so it stays readable at the width the journal prints.
PAPER_THEME = theme_gray() + theme(
    plot_title=element_blank(),
    legend_key_size=12,
    text=element_text(size=9),
    legend_title=element_text(size=9),
    strip_text=element_text(size=9),
)
# Single-column mode. 7 pt is the largest type that does not look oversized
# beside 10 pt body text in a 3.5 in wide float, and it still clears the 6 pt
# legibility floor. The legend moves under the panel because a legend at the
# side costs a third of the width at this size.
PAPER_COL_THEME = theme_gray() + theme(
    plot_title=element_blank(),
    legend_key_size=8,
    text=element_text(size=7),
    axis_title=element_text(size=7),
    legend_title=element_text(size=7),
    legend_text=element_text(size=6),
    strip_text=element_text(size=7),
    legend_position="bottom",
    legend_box_spacing=0.01,
)


def _family_of(config):
    if config == "Full-18":
        return "Full-18"
    return "Glass-n" if config.startswith("Glass") else "Best-n"


def _family_lines(df, ycol):
    """Each family's own configs plus the shared Full-18 anchor, so a drawn
    line runs small configs -> Full-18. Returns long df with column `y`."""
    out = []
    for fam in ("Glass-n", "Best-n"):
        sub = df[df.family.isin([fam, "Full-18"])].copy()
        sub["family_plot"] = fam
        out.append(sub)
    lines = pd.concat(out, ignore_index=True)
    lines["y"] = lines[ycol]
    return lines.sort_values("n_channels")


def _save(plot, name):
    out = FIG_DIR / f"{name}_plotnine.svg"
    plot.save(out, verbose=False)
    plot.save(FIG_DIR / f"{name}_plotnine_preview.png", dpi=110, verbose=False)
    print(f"Wrote {out}")


def _save_paper(plot, name):
    """Paper figures keep their own flat names (paper_fig1..3) so that the file
    name matches the figure number the caption uses - the exploratory names
    carry the old numbering and would be a trap when placing them in Word."""
    out = FIG_DIR / f"{name}.svg"
    plot.save(out, verbose=False)
    plot.save(FIG_DIR / f"{name}_preview.png", dpi=200, verbose=False)
    print(f"Wrote {out}")


# --------------------------------------------------------------------------- #
def fig1_channel_ladder():
    df = pd.read_csv(FIG_DIR / "data_channel_ladder.csv")
    lines = _family_lines(df, "pct_met_5fold")
    lines["grp"] = lines.family_plot + " | " + lines.criterion
    best_spread = df[df.family == "Best-n"]
    loo = df.dropna(subset=["pct_met_loo"])
    anchor = df[df.config == "Full-18"]

    p = (
        ggplot()
        + geom_segment(
            best_spread,
            aes(x="electrodes_min", xend="electrodes_max",
                y="pct_met_5fold", yend="pct_met_5fold"),
            color=FAMILY_COLORS["Best-n"], size=3, alpha=0.30,
        )
        + geom_line(
            lines,
            aes("electrodes_mean", "y", color="family_plot",
                linetype="criterion", group="grp"),
            size=0.9,
        )
        + geom_point(
            lines,
            aes("electrodes_mean", "y", color="family_plot", shape="criterion"),
            size=3.0,
        )
        + geom_point(
            loo, aes("electrodes_mean", "pct_met_loo", color="family"),
            shape="D", fill="white", size=4, stroke=1.1,
        )
        + geom_point(
            anchor, aes("electrodes_mean", "pct_met_5fold"),
            color=FAMILY_COLORS["Full-18"], size=3.4,
        )
        + scale_color_manual(
            FAMILY_COLORS,
            breaks=["Glass-n", "Best-n", "Full-18"],
            labels=["Glass-n (anatomically constrained)", "Best-n (data-driven)",
                    "Full-18 (shared anchor)"],
            name="Channel family",
        )
        + scale_linetype_manual(
            values=["solid", "dashed"], breaks=list(CRITERION_LABEL),
            labels=list(CRITERION_LABEL.values()), name="Clinical criterion",
        )
        + scale_shape_manual(
            values=["o", "s"], breaks=list(CRITERION_LABEL),
            labels=list(CRITERION_LABEL.values()), name="Clinical criterion",
        )
        + coord_cartesian(ylim=(-3, 103))
        + labs(
            x="Mean actual electrode count  (Best-n bar = range across 5 folds)",
            y="% subjects meeting criterion\n(event-level, SzCORE)",
            title="Channel ladder: pass rate vs. actual electrode count",
            caption="filled marker = 5-fold per-subject   |   open diamond = LOO "
                    "(Full-18 / Glass-7 / Glass-2 only)",
        )
        + BASE_THEME + theme(figure_size=(9.5, 6))
    )
    _save(p, "fig1_channel_ladder")


# --------------------------------------------------------------------------- #
def fig1b_auc_roc(paper=False):
    df = pd.read_csv(FIG_DIR / "data_channel_ladder.csv").drop_duplicates("config")
    lines = _family_lines(df, "auc_roc")
    anchor = df[df.config == "Full-18"]
    glass7 = df[df.config == "Glass-7"]

    p = (
        ggplot()
        + geom_line(
            lines, aes("electrodes_mean", "y", color="family_plot", group="family_plot"),
            size=0.9, alpha=0.7,
        )
        + geom_point(lines, aes("electrodes_mean", "y", color="family_plot"), size=2.8)
        + geom_point(anchor, aes("electrodes_mean", "auc_roc"),
                     color=FAMILY_COLORS["Full-18"], size=3.6)
        + annotate(
            "text",
            x=3.0 if paper else 10.5,
            y=0.655 if paper else 0.695,
            label="Glass-7 is Tier B (825k test windows, different file/hour\n"
                  "base than the other six configs - PR-AUC not comparable)",
            va="bottom", ha="left", size=5 if paper else 7, color="#555555",
        )
        + scale_color_manual(
            FAMILY_COLORS, breaks=["Glass-n", "Best-n", "Full-18"],
            labels=["Glass-n", "Best-n", "Full-18"], name="Channel family",
        )
        + coord_cartesian(ylim=(0.65, 0.85))
        + labs(
            x="Mean actual electrode count",
            y="AUC-ROC\n(window-level, pooled 5-fold test)",
            title="Window-level discrimination vs. actual electrode count",
        )
    )
    if paper:
        _save_paper(
            p
            + PAPER_COL_THEME
            + theme(figure_size=(PAPER_COL_WIDTH_IN, 2.9))
            + guides(color=guide_legend(nrow=1)),
            "paper_fig3")
        return
    _save(p + BASE_THEME + theme(figure_size=(9, 3.4)), "fig1b_auc_roc")


# --------------------------------------------------------------------------- #
def fig2_tradeoff(paper=False):
    path = FIG_DIR / "data_tradeoff_curve.csv"
    if not path.exists():
        print("SKIP fig2_tradeoff: data_tradeoff_curve.csv not built yet")
        return
    df = pd.read_csv(path)
    df = df.sort_values(["config", "threshold"]).copy()
    df["fa_plot"] = df.fa_per_day_szcore.clip(lower=1e-2)
    df["config"] = pd.Categorical(df.config, categories=CONFIG_ORDER, ordered=True)
    n_grid = int(df.groupby("config", observed=True).threshold.nunique().max())

    zones = pd.DataFrame([
        {"xmin": 1e-2, "xmax": 5, "ymin": 0.60, "ymax": 1.02, "fill": "#ffcc00"},
        {"xmin": 1e-2, "xmax": 2, "ymin": 0.80, "ymax": 1.02, "fill": "#2ca02c"},
    ])

    p = (
        ggplot()
        + geom_rect(zones, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                    fill=zones.fill.tolist(), alpha=0.13, inherit_aes=False)
        + geom_vline(xintercept=5, linetype="dotted", color="#b8960a")
        + geom_vline(xintercept=2, linetype="dotted", color="#1a7a1a")
        + geom_hline(yintercept=0.60, linetype="dotted", color="#b8960a")
        + geom_hline(yintercept=0.80, linetype="dotted", color="#1a7a1a")
        + geom_line(df, aes("fa_plot", "sens_szcore", color="config"), size=1.0)
        + scale_color_manual(CONFIG_COLORS, breaks=CONFIG_ORDER, name="Config")
        + scale_x_log10()
        + coord_cartesian(xlim=(1e-2, 3e2), ylim=(0, 1.02))
        + labs(
            x="False alarms / day  (micro, event-level SzCORE, pooled held-out test)",
            y="Sensitivity  (micro, event-level SzCORE)",
            title="Sensitivity vs. FA/day tradeoff by config",
            caption=(f"shaded = sufficiency zones   |   "
                     f"coarse {n_grid}-point grid, visualization only "
                     f"(not the locked 279-point grid used for 06_results)"),
        )
    )
    if paper:
        # The grid-coarseness disclaimer that the exploratory caption carries is
        # not dropped, only moved: it has to appear in the figure caption text.
        # The axis labels are shortened, not re-scoped: at 3.5 in the
        # exploratory wording runs past the canvas edge. "pooled held-out
        # test" moves into the caption with the grid-coarseness note.
        _save_paper(
            p
            + labs(title="", caption="",
                   x="False alarms / day  (micro, event-level SzCORE)",
                   y="Sensitivity  (micro, event-level SzCORE)")
            + PAPER_COL_THEME
            + theme(figure_size=(PAPER_COL_WIDTH_IN, 3.2))
            + guides(color=guide_legend(nrow=2)),
            "paper_fig1")
        return
    _save(p + BASE_THEME + theme(figure_size=(9.5, 6.5)), "fig2_tradeoff")


# --------------------------------------------------------------------------- #
def fig3_per_patient_heatmap(paper=False):
    df = pd.read_csv(FIG_DIR / "data_per_patient_heatmap.csv",
                     float_precision="round_trip")
    df["status"] = df.met.map({True: "Met criterion", False: "Did not meet"})
    df["status"] = df.status.fillna("No valid threshold")

    # one shared subject order (worst pass rate at bottom, best at top) - the
    # matplotlib version sorted each criterion panel independently; a single
    # order lets a subject be compared across the two panels instead.
    order = (df.groupby("subject").met.mean().sort_values().index.tolist())
    disp = {s: (f"{s}  *" if s in HARD_CASES else s) for s in order}
    df["subject_disp"] = pd.Categorical(
        df.subject.map(disp), categories=[disp[s] for s in order], ordered=True
    )
    df["config"] = pd.Categorical(df.config, categories=CONFIG_ORDER, ordered=True)
    df["criterion"] = df.criterion.map(
        PAPER_CRITERION_LABEL if paper else CRITERION_LABEL)

    p = (
        ggplot(df, aes("config", "subject_disp", fill="status"))
        + geom_tile(color="white", size=0.6)
        + facet_wrap("criterion", ncol=1 if paper else None)
        + scale_fill_manual({
            "Met criterion": "#2ca02c",
            "Did not meet": "#ededed",
            "No valid threshold": "#c6c6c6",
        }, name="")
        + guides(fill=guide_legend(title=""))
        + theme(axis_text_x=element_text(rotation=45, ha="right"),
                panel_grid=element_blank())
    )
    if paper:
        # The "*" and sort-order gloss moves from the axis label to the caption,
        # so the y axis says only what it is.
        _save_paper(
            p + labs(x="", y="Subject")
            + PAPER_COL_THEME
            + theme(figure_size=(PAPER_COL_WIDTH_IN, 6.6),
                    axis_text_x=element_text(rotation=45, ha="right", size=6),
                    axis_text_y=element_text(size=6),
                    panel_grid=element_blank())
            + guides(fill=guide_legend(title="", nrow=1)),
            "paper_fig2")
        return
    p = p + labs(
        x="", y="Subject  (*: recurring hard case; sorted by overall pass rate)",
        title="Per-patient pass / fail by config  (5-fold per-subject, SzCORE, RF)",
    ) + BASE_THEME + theme(
        figure_size=(11, 7),
        axis_text_x=element_text(rotation=45, ha="right"),
        panel_grid=element_blank(),
    )
    _save(p, "fig3_per_patient_heatmap")


# --------------------------------------------------------------------------- #
def fig4_szcore_vs_ali():
    df = pd.read_csv(FIG_DIR / "data_szcore_vs_ali.csv")
    df = df[df.scheme == "5fold"].copy()
    df["config"] = pd.Categorical(df.config, categories=CONFIG_ORDER, ordered=True)
    rule_lbl = {"szcore": "SzCORE (primary)", "ali": "Ali 2024 (secondary)"}
    df["rule"] = pd.Categorical(df.rule.map(rule_lbl),
                                categories=list(rule_lbl.values()), ordered=True)
    df["criterion"] = df.criterion.map(CRITERION_LABEL)

    p = (
        ggplot(df, aes("config", "pct_subjects_meeting_criterion", fill="rule"))
        + geom_col(position="dodge", width=0.7)
        + facet_wrap("criterion")
        + scale_fill_manual({"SzCORE (primary)": "#1f77b4",
                             "Ali 2024 (secondary)": "#ff7f0e"}, name="Scoring rule")
        + labs(
            x="", y="% subjects meeting criterion",
            title=("Scoring-rule comparison: SzCORE vs. Ali 2024\n"
                   "same predictions, same config (5-fold per-subject, RF)"),
        )
        + BASE_THEME
        + theme(figure_size=(11, 5),
                axis_text_x=element_text(rotation=45, ha="right"))
    )
    _save(p, "fig4_szcore_vs_ali")


def paper_figures():
    """The three figures the paper carries, in citation order."""
    fig2_tradeoff(paper=True)              # paper_fig1
    fig3_per_patient_heatmap(paper=True)   # paper_fig2
    fig1b_auc_roc(paper=True)              # paper_fig3


def main():
    fig1_channel_ladder()
    fig1b_auc_roc()
    fig2_tradeoff()
    fig3_per_patient_heatmap()
    fig4_szcore_vs_ali()
    paper_figures()


if __name__ == "__main__":
    main()
