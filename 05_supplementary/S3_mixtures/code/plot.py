from __future__ import annotations

import csv
from pathlib import Path

from svg_native import NativeSVG

OUT = Path(__file__).resolve().parents[1]

EDGE = "#333333"
GRID = "#dedede"
EXP_SUMMARY = "#f2bf82"
PRED_SUMMARY = "#d5e6f4"
BCAT = "#89a9d9"
COLORS_EXP = {
    "2,6-Dichlorophenol": "#f2bf82",
    "2-Chloro-4-nitrophenol": "#8dca8d",
    "3,5-Diiodo-4-hydroxybenzaldehyde": "#b5a0c7",
    "5-Bromosalicylic acid": "#ffff91",
}
COLORS_PRED = {
    "2,6-Dichlorophenol": "#f8d9b2",
    "2-Chloro-4-nitrophenol": "#bde4bd",
    "3,5-Diiodo-4-hydroxybenzaldehyde": "#d5c7df",
    "5-Bromosalicylic acid": "#ffffc4",
}
STACKS = {
    "Group1": ["2,6-Dichlorophenol", "2-Chloro-4-nitrophenol"],
    "Group2": ["3,5-Diiodo-4-hydroxybenzaldehyde", "5-Bromosalicylic acid"],
    "Group3": [
        "2,6-Dichlorophenol", "2-Chloro-4-nitrophenol",
        "3,5-Diiodo-4-hydroxybenzaldehyde", "5-Bromosalicylic acid",
    ],
}

CAT = {
    "Group1": {"exp": 0.000448616654875353, "pred": 0.000573960557430493,
               "bcat": 0.000609, "exp_sd": 0.000011, "pred_sd": 0.000032,
               "bcat_sd": 0.000045},
    "Group2": {"exp": 0.001854593239206654, "pred": 0.002181081081081081,
               "bcat": 0.001614, "exp_sd": 0.000044, "pred_sd": 0.000335,
               "bcat_sd": 0.000046},
    "Group3": {"exp": 0.000722471206176323, "pred": 0.000909090909090909,
               "bcat": 0.000800, "exp_sd": 0.000014, "pred_sd": 0.000050,
               "bcat_sd": 0.000026},
}
MDR = {
    "Group1": {"exp": 1.3575064443, "pred": 1.061, "bcat": 1.0,
               "exp_low": 1.2068096415, "exp_high": 1.5267315424,
               "pred_low": 0.949, "pred_high": 1.187,
               "p_exp": "p=0.002", "p_pred": "p<0.001"},
    "Group2": {"exp": 0.8702716940, "pred": 0.740, "bcat": 1.0,
               "exp_low": 0.8310340107, "exp_high": 0.9115215662,
               "pred_low": 0.670, "pred_high": 0.810,
               "p_exp": "p<0.001", "p_pred": "p<0.001"},
    "Group3": {"exp": 1.1073105657, "pred": 0.880, "bcat": 1.0,
               "exp_low": 1.0528914567, "exp_high": 1.1652787470,
               "pred_low": 0.830, "pred_high": 0.930,
               "p_exp": "p<0.001", "p_pred": "p<0.001"},
}
DIFFERENCE = {
    "Group1": {"exp": 9.43, "pred": 2.16},
    "Group2": {"exp": 14.51, "pred": 16.35},
    "Group3": {"exp": 6.57, "pred": 15.82},
}

def ymap(value: float, ymin: float, ymax: float, top: float, bottom: float) -> float:
    return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

def add_y_axis(svg: NativeSVG, x0: float, x1: float, top: float, bottom: float,
               ymin: float, ymax: float, ticks: list[float], formatter,
               label: str, *, grid: bool = False) -> None:
    for tick in ticks:
        y = ymap(tick, ymin, ymax, top, bottom)
        if grid and tick not in (ymin, ymax):
            svg.line(x0, y, x1, y, stroke=GRID, stroke_width=0.7,
                     prefix="gridline")
        svg.line(x0 - 4.5, y, x0, y, stroke=EDGE, stroke_width=1.0,
                 prefix="ytick")
        svg.text(x0 - 8, y + 4.2, formatter(tick), size=11.5,
                 anchor="end", prefix="ytick-label")
    svg.text(20, (top + bottom) / 2, label, size=15.0, weight="bold",
             rotate=-90, prefix="ylabel")

def add_group_labels(svg: NativeSVG, centers: list[float], bottom: float) -> None:
    for center, group in zip(centers, CAT):
        svg.line(center, bottom, center, bottom + 5, stroke=EDGE, stroke_width=1.0,
                 prefix="xtick")
        svg.text(center, bottom + 23, group, size=15.0, prefix="xlabel")

def add_legend_item(svg: NativeSVG, x: float, y: float, label: str, fill: str,
                    *, hatched: bool = False, font_size: float = 10.5,
                    swatch_w: float = 15.0) -> None:
    svg.rect(x, y - 8.0, swatch_w, 8.0, fill=fill, stroke=EDGE,
             stroke_width=0.9, prefix="legend-swatch")
    if hatched:
        svg.hatch_rect(x, y - 8.0, swatch_w, 8.0, spacing=6.0,
                       stroke_width=0.65, prefix="legend-hatch")
    svg.text(x + swatch_w + 6, y - 0.5, label, size=font_size,
             anchor="start", prefix="legend-label")

def add_bar(svg: NativeSVG, x: float, width: float, value: float,
            ymin: float, ymax: float, top: float, bottom: float, fill: str,
            *, hatched: bool = False, prefix: str = "bar") -> tuple[float, float]:
    y = ymap(value, ymin, ymax, top, bottom)
    svg.rect(x, y, width, bottom - y, fill=fill, stroke=EDGE,
             stroke_width=1.0, prefix=prefix)
    if hatched:
        svg.hatch_rect(x, y, width, bottom - y, spacing=12.0,
                       stroke_width=0.75, prefix=f"{prefix}-hatch")
    return y, bottom

def write_source_data() -> None:
    rows: list[dict[str, object]] = []
    for group, values in CAT.items():
        for series in ("exp", "pred", "bcat"):
            rows.append({"panel": "a", "group": group, "series": series,
                         "value": values[series], "lower": "", "upper": "",
                         "uncertainty": values[f"{series}_sd"]})
    for panel in ("b", "c"):
        for group, values in MDR.items():
            for series in ("exp", "pred", "bcat"):
                rows.append({"panel": panel, "group": group, "series": series,
                             "value": values[series],
                             "lower": values.get(f"{series}_low", ""),
                             "upper": values.get(f"{series}_high", ""),
                             "uncertainty": ""})
    for group, values in DIFFERENCE.items():
        for series in ("exp", "pred"):
            rows.append({"panel": "d", "group": group, "series": series,
                         "value": values[series], "lower": "", "upper": "",
                         "uncertainty": ""})
    with (OUT / "source_data_mixture_panels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def panel_a() -> Path:
    svg = NativeSVG()
    x0, x1, top, bottom = 72.0, 587.0, 36.0, 408.0
    ymin, ymax = 0.0, 0.0030
    centers = [158.0, 330.0, 502.0]
    bar_w = 39.0

    svg.text(300, 24, "CAT and BCAT", size=18.5, weight="bold", prefix="title")
    add_y_axis(svg, x0, x1, top, bottom, ymin, ymax,
               [0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030],
               lambda value: f"{value:.4f}", "CHO Cell Cytotoxicity Index")
    add_group_labels(svg, centers, bottom)

    legend_left = [
        ("5-Bromosalicylic acid", COLORS_EXP["5-Bromosalicylic acid"]),
        ("3,5-Diiodo-4-hydroxybenzaldehyde", COLORS_EXP["3,5-Diiodo-4-hydroxybenzaldehyde"]),
        ("2-Chloro-4-nitrophenol", COLORS_EXP["2-Chloro-4-nitrophenol"]),
        ("2,6-Dichlorophenol", COLORS_EXP["2,6-Dichlorophenol"]),
    ]
    legend_right = [
        ("5-Bromosalicylic acid", COLORS_PRED["5-Bromosalicylic acid"]),
        ("3,5-Diiodo-4-hydroxybenzaldehyde", COLORS_PRED["3,5-Diiodo-4-hydroxybenzaldehyde"]),
        ("2-Chloro-4-nitrophenol", COLORS_PRED["2-Chloro-4-nitrophenol"]),
        ("2,6-Dichlorophenol", COLORS_PRED["2,6-Dichlorophenol"]),
    ]
    for i, (compound, color) in enumerate(legend_left):
        add_legend_item(svg, 88, 54 + i * 13.0, f"CAT-exp-{compound}", color,
                        font_size=10.7, swatch_w=11.5)
    add_legend_item(svg, 88, 106, "BCAT", BCAT, font_size=10.7, swatch_w=11.5)
    for i, (compound, color) in enumerate(legend_right):
        add_legend_item(svg, 335, 54 + i * 13.0, f"CAT-pred-{compound}", color,
                        hatched=True, font_size=10.7, swatch_w=11.5)
    svg.text(578, 50, "(a)", size=17.0, anchor="end", weight="bold",
             prefix="panel-label")

    for center, group in zip(centers, CAT):
        values = CAT[group]
        compounds = STACKS[group]
        exp_x, pred_x, bcat_x = center - 1.5 * bar_w, center - 0.5 * bar_w, center + 0.5 * bar_w
        for series, x, palette, hatched in (
            ("exp", exp_x, COLORS_EXP, False),
            ("pred", pred_x, COLORS_PRED, True),
        ):
            total = values[series]
            part = total / len(compounds)
            cumulative = 0.0
            for compound in compounds:
                y_top = ymap(cumulative + part, ymin, ymax, top, bottom)
                y_bottom = ymap(cumulative, ymin, ymax, top, bottom)
                svg.rect(x, y_top, bar_w, y_bottom - y_top, fill=palette[compound],
                         stroke=EDGE, stroke_width=1.0,
                         prefix=f"bar-{group.lower()}-{series}-{compound.lower().replace(' ', '-')}")
                if hatched:
                    svg.hatch_rect(x, y_top, bar_w, y_bottom - y_top,
                                   spacing=12.0, stroke_width=0.7,
                                   prefix=f"hatch-{group.lower()}-{series}")
                cumulative += part
            center_y = ymap(total, ymin, ymax, top, bottom)
            low_y = ymap(total - values[f"{series}_sd"], ymin, ymax, top, bottom)
            high_y = ymap(total + values[f"{series}_sd"], ymin, ymax, top, bottom)
            svg.errorbar(x + bar_w / 2, low_y, high_y, cap=4.8,
                         prefix=f"errorbar-{group.lower()}-{series}")
        add_bar(svg, bcat_x, bar_w, values["bcat"], ymin, ymax, top, bottom,
                BCAT, prefix=f"bar-{group.lower()}-bcat")
        svg.errorbar(bcat_x + bar_w / 2,
                     ymap(values["bcat"] - values["bcat_sd"], ymin, ymax, top, bottom),
                     ymap(values["bcat"] + values["bcat_sd"], ymin, ymax, top, bottom),
                     cap=4.8, prefix=f"errorbar-{group.lower()}-bcat")

    svg.frame(x0, top, x1, bottom)
    path = OUT / "Fig_mixture_a_CAT_BCAT_editable.svg"
    svg.save(path, "CAT and BCAT",
             "Experimental and predicted concentration-addition toxicity indices compared with BCAT for three mixtures.")
    return path

def add_mdr_legend(svg: NativeSVG, y: float, *, tost: bool) -> None:
    prefix = "TOST" if tost else "MDR"
    add_legend_item(svg, 126, y, f"{prefix}-exp-CAT", EXP_SUMMARY,
                    font_size=11.0, swatch_w=25.0)
    add_legend_item(svg, 278, y, f"{prefix}-pred-CAT", PRED_SUMMARY,
                    hatched=True, font_size=11.0, swatch_w=25.0)
    add_legend_item(svg, 443, y, "BCAT", BCAT,
                    font_size=11.0, swatch_w=25.0)

def draw_mdr_bars(svg: NativeSVG, *, top: float, bottom: float,
                  ymin: float, ymax: float, tost: bool) -> None:
    centers = [150.0, 325.0, 500.0]
    bar_w = 42.0
    for center, group in zip(centers, MDR):
        values = MDR[group]
        positions = {"exp": center - 1.5 * bar_w,
                     "pred": center - 0.5 * bar_w,
                     "bcat": center + 0.5 * bar_w}
        for series, fill, hatched in (
            ("exp", EXP_SUMMARY, False),
            ("pred", PRED_SUMMARY, True),
            ("bcat", BCAT, False),
        ):
            x = positions[series]
            add_bar(svg, x, bar_w, values[series], ymin, ymax, top, bottom,
                    fill, hatched=hatched,
                    prefix=f"bar-{group.lower()}-{series}")
            if series in ("exp", "pred"):
                svg.errorbar(x + bar_w / 2,
                             ymap(values[f"{series}_low"], ymin, ymax, top, bottom),
                             ymap(values[f"{series}_high"], ymin, ymax, top, bottom),
                             cap=5.0, prefix=f"errorbar-{group.lower()}-{series}")
        if tost:
            p_y = {"Group1": 1.69, "Group2": 1.08, "Group3": 1.68}[group]
            svg.text(positions["exp"] + bar_w / 2 - 6,
                     ymap(p_y, ymin, ymax, top, bottom), values["p_exp"],
                     size=10.7, prefix=f"pvalue-{group.lower()}-exp")
            svg.text(positions["pred"] + bar_w / 2 + 6,
                     ymap(p_y, ymin, ymax, top, bottom), values["p_pred"],
                     size=10.7, prefix=f"pvalue-{group.lower()}-pred")
        else:
            for series in ("exp", "pred", "bcat"):
                x = positions[series] + bar_w / 2
                upper = values.get(f"{series}_high", values[series])
                label_y = ymap(upper, ymin, ymax, top, bottom) - 12
                svg.text(x, label_y, f"{values[series]:.2f}", size=11.5,
                         prefix=f"value-{group.lower()}-{series}")

def panel_b() -> Path:
    svg = NativeSVG()
    x0, x1, top, bottom = 60.0, 587.0, 38.0, 408.0
    ymin, ymax = 0.35, 2.12
    centers = [150.0, 325.0, 500.0]
    svg.text(300, 24, "MDR", size=19.0, weight="bold", prefix="title")
    add_y_axis(svg, x0, x1, top, bottom, ymin, ymax,
               [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
               lambda value: f"{value:.2f}", "MDR")
    add_group_labels(svg, centers, bottom)
    svg.line(x0, ymap(0.5, ymin, ymax, top, bottom), x1,
             ymap(0.5, ymin, ymax, top, bottom), stroke="#777777",
             stroke_width=0.85, dash="4 3", prefix="equivalence-lower")
    svg.line(x0, ymap(1.0, ymin, ymax, top, bottom), x1,
             ymap(1.0, ymin, ymax, top, bottom), stroke=EDGE,
             stroke_width=1.0, prefix="reference-one")
    svg.line(x0, ymap(2.0, ymin, ymax, top, bottom), x1,
             ymap(2.0, ymin, ymax, top, bottom), stroke="#777777",
             stroke_width=0.85, dash="4 3", prefix="equivalence-upper")
    add_mdr_legend(svg, 58, tost=False)
    svg.text(578, 57, "(b)", size=17.0, anchor="end", weight="bold",
             prefix="panel-label")
    draw_mdr_bars(svg, top=top, bottom=bottom, ymin=ymin, ymax=ymax, tost=False)
    svg.frame(x0, top, x1, bottom)
    path = OUT / "Fig_mixture_b_MDR_editable.svg"
    svg.save(path, "MDR", "Mixture deviation ratios with 90% confidence intervals.")
    return path

def panel_c() -> Path:
    svg = NativeSVG()
    x0, x1, top, bottom = 60.0, 587.0, 38.0, 408.0
    ymin, ymax = 0.35, 2.12
    centers = [150.0, 325.0, 500.0]
    svg.text(300, 24, "TOST", size=19.0, weight="bold", prefix="title")
    add_y_axis(svg, x0, x1, top, bottom, ymin, ymax,
               [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
               lambda value: f"{value:.2f}", "MDR with 90% CI")
    add_group_labels(svg, centers, bottom)
    svg.line(x0, ymap(0.5, ymin, ymax, top, bottom), x1,
             ymap(0.5, ymin, ymax, top, bottom), stroke="#777777",
             stroke_width=0.85, dash="4 3", prefix="equivalence-lower")
    svg.line(x0, ymap(1.0, ymin, ymax, top, bottom), x1,
             ymap(1.0, ymin, ymax, top, bottom), stroke=EDGE,
             stroke_width=1.0, prefix="reference-one")
    svg.line(x0, ymap(2.0, ymin, ymax, top, bottom), x1,
             ymap(2.0, ymin, ymax, top, bottom), stroke="#777777",
             stroke_width=0.85, dash="4 3", prefix="equivalence-upper")
    add_mdr_legend(svg, 58, tost=True)
    svg.text(578, 57, "(c)", size=17.0, anchor="end", weight="bold",
             prefix="panel-label")
    svg.text(x1 - 10, ymap(2.0, ymin, ymax, top, bottom) + 16, "2.0",
             size=10.7, anchor="end", prefix="equivalence-label-upper")
    svg.text(x1 - 10, ymap(0.5, ymin, ymax, top, bottom) - 7, "0.5",
             size=10.7, anchor="end", prefix="equivalence-label-lower")
    draw_mdr_bars(svg, top=top, bottom=bottom, ymin=ymin, ymax=ymax, tost=True)
    svg.frame(x0, top, x1, bottom)
    path = OUT / "Fig_mixture_c_TOST_editable.svg"
    svg.save(path, "TOST", "Two one-sided equivalence tests for experimental and predicted mixture deviation ratios.")
    return path

def panel_d() -> Path:
    svg = NativeSVG()
    x0, x1, top, bottom = 55.0, 587.0, 38.0, 408.0
    ymin, ymax = 0.0, 19.0
    centers = [145.0, 320.0, 495.0]
    bar_w = 62.0
    svg.text(300, 24, "D", size=19.0, weight="bold", prefix="title")
    add_y_axis(svg, x0, x1, top, bottom, ymin, ymax,
               [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5],
               lambda value: f"{value:.1f}", "Symmetric percent difference, D (%)")
    add_group_labels(svg, centers, bottom)
    add_legend_item(svg, 205, 57, "D-exp-CAT", EXP_SUMMARY,
                    font_size=11.0, swatch_w=21.0)
    add_legend_item(svg, 350, 57, "D-pred-latest-diagnostic", PRED_SUMMARY,
                    hatched=True, font_size=11.0, swatch_w=21.0)
    svg.text(578, 57, "(d)", size=17.0, anchor="end", weight="bold",
             prefix="panel-label")
    for center, group in zip(centers, DIFFERENCE):
        values = DIFFERENCE[group]
        for series, x, fill, hatched in (
            ("exp", center - bar_w, EXP_SUMMARY, False),
            ("pred", center, PRED_SUMMARY, True),
        ):
            y, _ = add_bar(svg, x, bar_w, values[series], ymin, ymax, top, bottom,
                           fill, hatched=hatched,
                           prefix=f"bar-{group.lower()}-{series}")
            svg.text(x + bar_w / 2, y - 9, f"{values[series]:.2f}", size=11.5,
                     prefix=f"value-{group.lower()}-{series}")
    svg.frame(x0, top, x1, bottom)
    path = OUT / "Fig_mixture_d_symmetric_difference_editable.svg"
    svg.save(path, "D", "Symmetric percent differences for experimental and predicted CAT estimates.")
    return path

def main() -> None:
    write_source_data()
    paths = [panel_a(), panel_b(), panel_c(), panel_d()]
    for path in paths:
        print(path)

if __name__ == "__main__":
    main()
