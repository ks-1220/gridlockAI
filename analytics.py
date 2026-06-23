"""
analytics.py — Analytics & Chart Generation
=============================================
Premium-quality analytics dashboard module for the Traffic Violation
Detection System.  All charts use Plotly with a unified dark theme that
matches the Streamlit dashboard palette defined in ``utils.py``.

Features
--------
* Violations-by-type horizontal bar chart with per-type coloring
* Confidence-score histogram with gradient fill & threshold marker
* Severity donut chart with count + percentage labels
* Spatial density heatmap (grid overlay on image dimensions)
* Confidence box-plot per violation type
* One-call summary metrics dictionary for KPI cards

Author: Team Gridlock | Flipkart Grid 6.0
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from utils import (
    ViolationType,
    Severity,
    ViolationRecord,
    AnalysisResult,
    VIOLATION_COLORS,
    SEVERITY_COLORS,
    COLORS,
    VIOLATION_SEVERITY,
    format_confidence,
)


# ──────────────────────────────────────────────
#  Module-level constants
# ──────────────────────────────────────────────

_FONT_FAMILY: str = "Inter, Roboto, -apple-system, sans-serif"
_CONFIDENCE_THRESHOLD: float = 0.55  # default CLIP threshold line
_HEATMAP_GRID_COLS: int = 12
_HEATMAP_GRID_ROWS: int = 8
_CHART_HEIGHT: int = 420
_CHART_MARGIN: Dict[str, int] = {"l": 20, "r": 20, "t": 50, "b": 20}


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a hex color string to an ``rgba(...)`` CSS string.

    Parameters
    ----------
    hex_color:
        Color in ``#RRGGBB`` format.
    alpha:
        Opacity value in [0, 1].

    Returns
    -------
    str
        CSS ``rgba()`` string.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ──────────────────────────────────────────────
#  Main analytics class
# ──────────────────────────────────────────────


class ViolationAnalytics:
    """Generate production-quality Plotly charts for the violation dashboard.

    All figures are returned as ``plotly.graph_objects.Figure`` instances
    that can be rendered directly with ``st.plotly_chart`` or exported to
    static images.

    Attributes
    ----------
    font_family : str
        Primary font stack applied to every chart.
    bg_primary : str
        Background color sourced from the shared ``COLORS`` palette.
    text_primary : str
        Primary text color sourced from the shared ``COLORS`` palette.
    """

    # ── construction ───────────────────────────

    def __init__(self) -> None:
        """Initialise Plotly template defaults for the dark dashboard theme."""
        self.font_family: str = _FONT_FAMILY
        self.bg_primary: str = COLORS["bg_primary"]
        self.bg_secondary: str = COLORS["bg_secondary"]
        self.bg_card: str = COLORS["bg_card"]
        self.text_primary: str = COLORS["text_primary"]
        self.text_secondary: str = COLORS["text_secondary"]
        self.accent_blue: str = COLORS["accent_blue"]
        self.accent_purple: str = COLORS["accent_purple"]

    # ── public chart methods ───────────────────

    def violations_by_type_chart(
        self, violations: List[ViolationRecord]
    ) -> go.Figure:
        """Horizontal bar chart — count of each violation type.

        Bars are coloured using ``VIOLATION_COLORS`` and labelled with
        their counts.  The chart is sorted descending so the most
        frequent violation appears at the top.

        Parameters
        ----------
        violations:
            List of violation records from the analysis pipeline.

        Returns
        -------
        go.Figure
            Styled Plotly figure ready for rendering.
        """
        counts: Counter[ViolationType] = Counter(
            v.violation_type for v in violations
        )

        # Sort by count descending; show all types even if zero
        all_types = list(ViolationType)
        sorted_types = sorted(all_types, key=lambda t: counts.get(t, 0))

        labels: List[str] = [t.value for t in sorted_types]
        values: List[int] = [counts.get(t, 0) for t in sorted_types]
        bar_colors: List[str] = [
            VIOLATION_COLORS.get(t, self.accent_blue) for t in sorted_types
        ]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(width=0),
                    cornerradius=6,  # Plotly ≥ 5.19 rounded-corner bars
                ),
                text=[str(v) for v in values],
                textposition="outside",
                textfont=dict(
                    family=self.font_family,
                    size=13,
                    color=self.text_primary,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Count: %{x}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=dict(
                text="Violations by Type",
                font=dict(size=16, color=self.text_primary),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True,
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(
                    family=self.font_family,
                    size=12,
                    color=self.text_secondary,
                ),
                fixedrange=True,
            ),
            height=max(_CHART_HEIGHT, len(all_types) * 52),
            bargap=0.30,
        )

        return self._apply_dark_theme(fig)

    def confidence_distribution_chart(
        self, violations: List[ViolationRecord]
    ) -> go.Figure:
        """Histogram of confidence scores with gradient fill & threshold.

        A vertical dashed line marks the CLIP confidence threshold.
        Bins to the left are faded to visually separate low-confidence
        detections.

        Parameters
        ----------
        violations:
            List of violation records.

        Returns
        -------
        go.Figure
            Styled Plotly histogram figure.
        """
        scores: List[float] = [v.confidence for v in violations]

        fig = go.Figure()

        # Primary histogram trace with gradient-like fill
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=20,
                marker=dict(
                    color=_hex_to_rgba(self.accent_blue, 0.75),
                    line=dict(
                        color=_hex_to_rgba(self.accent_blue, 1.0),
                        width=1,
                    ),
                ),
                hovertemplate=(
                    "Confidence: %{x:.2f}<br>"
                    "Count: %{y}<extra></extra>"
                ),
                name="Detections",
            )
        )

        # Threshold reference line
        fig.add_vline(
            x=_CONFIDENCE_THRESHOLD,
            line=dict(color=SEVERITY_COLORS[Severity.CRITICAL], width=2, dash="dash"),
            annotation=dict(
                text=f"Threshold ({format_confidence(_CONFIDENCE_THRESHOLD)})",
                font=dict(
                    family=self.font_family,
                    size=11,
                    color=SEVERITY_COLORS[Severity.CRITICAL],
                ),
                showarrow=False,
                yshift=10,
            ),
        )

        fig.update_layout(
            title=dict(
                text="Confidence Score Distribution",
                font=dict(size=16, color=self.text_primary),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Confidence",
                title_font=dict(
                    family=self.font_family,
                    size=12,
                    color=self.text_secondary,
                ),
                tickfont=dict(
                    family=self.font_family,
                    size=11,
                    color=self.text_secondary,
                ),
                showgrid=False,
                zeroline=False,
                range=[0, 1.05],
                fixedrange=True,
            ),
            yaxis=dict(
                title="Count",
                title_font=dict(
                    family=self.font_family,
                    size=12,
                    color=self.text_secondary,
                ),
                tickfont=dict(
                    family=self.font_family,
                    size=11,
                    color=self.text_secondary,
                ),
                showgrid=True,
                gridcolor=_hex_to_rgba(COLORS["border"], 0.4),
                gridwidth=1,
                zeroline=False,
                fixedrange=True,
            ),
            showlegend=False,
            height=_CHART_HEIGHT,
        )

        return self._apply_dark_theme(fig)

    def severity_pie_chart(
        self, violations: List[ViolationRecord]
    ) -> go.Figure:
        """Donut chart showing violation proportions by severity.

        Segment colours use ``SEVERITY_COLORS``.  Labels include both
        percentage and absolute count.

        Parameters
        ----------
        violations:
            List of violation records.

        Returns
        -------
        go.Figure
            Styled Plotly donut chart.
        """
        severity_counts: Counter[Severity] = Counter(
            v.severity for v in violations
        )

        # Maintain a consistent ordering: Critical → Moderate → Minor
        ordered_severities: List[Severity] = [
            Severity.CRITICAL,
            Severity.MODERATE,
            Severity.MINOR,
        ]
        labels: List[str] = []
        values: List[int] = []
        colors: List[str] = []

        for sev in ordered_severities:
            count = severity_counts.get(sev, 0)
            if count > 0:
                labels.append(sev.value)
                values.append(count)
                colors.append(SEVERITY_COLORS[sev])

        # Fallback when there are no violations
        if not values:
            labels = ["No Violations"]
            values = [1]
            colors = [COLORS["bg_hover"]]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(
                    colors=colors,
                    line=dict(color=self.bg_primary, width=3),
                ),
                textinfo="percent+value",
                texttemplate="%{label}<br>%{value} (%{percent})",
                textfont=dict(
                    family=self.font_family,
                    size=12,
                    color=self.text_primary,
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
                sort=False,
            )
        )

        total = sum(values) if values[0] != 1 or labels[0] != "No Violations" else 0
        fig.update_layout(
            title=dict(
                text="Severity Breakdown",
                font=dict(size=16, color=self.text_primary),
                x=0.0,
                xanchor="left",
            ),
            annotations=[
                dict(
                    text=f"<b>{total}</b><br><span style='font-size:11px;color:{self.text_secondary}'>total</span>",
                    x=0.5,
                    y=0.5,
                    font=dict(
                        family=self.font_family,
                        size=26,
                        color=self.text_primary,
                    ),
                    showarrow=False,
                )
            ],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(
                    family=self.font_family,
                    size=12,
                    color=self.text_secondary,
                ),
            ),
            height=_CHART_HEIGHT,
        )

        return self._apply_dark_theme(fig)

    def severity_heatmap(
        self,
        violations: List[ViolationRecord],
        image_shape: Tuple[int, ...],
    ) -> go.Figure:
        """Spatial density heatmap — grid overlay coloured by severity.

        The image area is divided into a grid.  Each cell accumulates a
        severity-weighted score from violations whose bounding-box
        centre falls inside it.  The resulting grid is rendered as a
        Plotly heatmap.

        Parameters
        ----------
        violations:
            List of violation records (must have ``bbox`` attribute).
        image_shape:
            Shape of the source image as ``(height, width, ...)`` — the
            same convention as ``numpy.ndarray.shape``.

        Returns
        -------
        go.Figure
            Styled Plotly heatmap figure.
        """
        img_h: int = image_shape[0]
        img_w: int = image_shape[1]
        rows: int = _HEATMAP_GRID_ROWS
        cols: int = _HEATMAP_GRID_COLS

        cell_h: float = img_h / rows
        cell_w: float = img_w / cols

        # Severity weight mapping for density calculation
        severity_weight: Dict[Severity, float] = {
            Severity.CRITICAL: 3.0,
            Severity.MODERATE: 2.0,
            Severity.MINOR: 1.0,
        }

        grid: np.ndarray = np.zeros((rows, cols), dtype=np.float64)

        for v in violations:
            cx, cy = v.bbox.center
            col_idx = min(int(cx / cell_w), cols - 1)
            row_idx = min(int(cy / cell_h), rows - 1)
            grid[row_idx, col_idx] += severity_weight.get(v.severity, 1.0)

        # Build axis labels (approximate pixel ranges)
        x_labels: List[str] = [
            f"{int(i * cell_w)}–{int((i + 1) * cell_w)}" for i in range(cols)
        ]
        y_labels: List[str] = [
            f"{int(i * cell_h)}–{int((i + 1) * cell_h)}" for i in range(rows)
        ]

        # Custom colorscale: transparent-dark → accent-blue → red
        colorscale: List[List[Any]] = [
            [0.0, _hex_to_rgba(self.bg_card, 0.9)],
            [0.25, _hex_to_rgba(self.accent_blue, 0.5)],
            [0.5, _hex_to_rgba(self.accent_blue, 0.85)],
            [0.75, _hex_to_rgba(self.accent_purple, 0.9)],
            [1.0, _hex_to_rgba(SEVERITY_COLORS[Severity.CRITICAL], 1.0)],
        ]

        fig = go.Figure(
            go.Heatmap(
                z=grid,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Severity Score",
                        font=dict(
                            family=self.font_family,
                            size=11,
                            color=self.text_secondary,
                        ),
                    ),
                    tickfont=dict(
                        family=self.font_family,
                        size=10,
                        color=self.text_secondary,
                    ),
                    outlinewidth=0,
                    bgcolor=self.bg_primary,
                ),
                hovertemplate=(
                    "X: %{x}<br>"
                    "Y: %{y}<br>"
                    "Score: %{z:.1f}<extra></extra>"
                ),
                xgap=3,
                ygap=3,
            )
        )

        fig.update_layout(
            title=dict(
                text="Violation Density Heatmap",
                font=dict(size=16, color=self.text_primary),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Horizontal Position (px)",
                title_font=dict(
                    family=self.font_family,
                    size=11,
                    color=self.text_secondary,
                ),
                tickfont=dict(
                    family=self.font_family,
                    size=9,
                    color=self.text_secondary,
                ),
                showgrid=False,
                fixedrange=True,
            ),
            yaxis=dict(
                title="Vertical Position (px)",
                title_font=dict(
                    family=self.font_family,
                    size=11,
                    color=self.text_secondary,
                ),
                tickfont=dict(
                    family=self.font_family,
                    size=9,
                    color=self.text_secondary,
                ),
                showgrid=False,
                autorange="reversed",
                fixedrange=True,
            ),
            height=_CHART_HEIGHT,
        )

        return self._apply_dark_theme(fig)

    def confidence_by_violation_chart(
        self, violations: List[ViolationRecord]
    ) -> go.Figure:
        """Box plot of confidence distributions grouped by violation type.

        Each violation type is rendered as a separate box with its
        assigned colour from ``VIOLATION_COLORS``.

        Parameters
        ----------
        violations:
            List of violation records.

        Returns
        -------
        go.Figure
            Styled Plotly box-plot figure.
        """
        # Group confidence scores by violation type
        grouped: Dict[ViolationType, List[float]] = defaultdict(list)
        for v in violations:
            grouped[v.violation_type].append(v.confidence)

        fig = go.Figure()

        # Stable ordering: all ViolationType members
        for vtype in ViolationType:
            scores = grouped.get(vtype, [])
            if not scores:
                continue

            color = VIOLATION_COLORS.get(vtype, self.accent_blue)

            fig.add_trace(
                go.Box(
                    y=scores,
                    name=vtype.value,
                    marker=dict(
                        color=color,
                        outliercolor=_hex_to_rgba(color, 0.5),
                        size=5,
                    ),
                    line=dict(color=color, width=1.5),
                    fillcolor=_hex_to_rgba(color, 0.25),
                    boxmean="sd",
                    hoverinfo="y+name",
                    jitter=0.3,
                    pointpos=-1.5,
                    boxpoints="all",
                )
            )

        fig.update_layout(
            title=dict(
                text="Confidence by Violation Type",
                font=dict(size=16, color=self.text_primary),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                tickfont=dict(
                    family=self.font_family,
                    size=11,
                    color=self.text_secondary,
                ),
                showgrid=False,
                fixedrange=True,
            ),
            yaxis=dict(
                title="Confidence",
                title_font=dict(
                    family=self.font_family,
                    size=12,
                    color=self.text_secondary,
                ),
                tickfont=dict(
                    family=self.font_family,
                    size=11,
                    color=self.text_secondary,
                ),
                showgrid=True,
                gridcolor=_hex_to_rgba(COLORS["border"], 0.3),
                gridwidth=1,
                zeroline=False,
                range=[0, 1.05],
                fixedrange=True,
            ),
            showlegend=False,
            height=_CHART_HEIGHT,
        )

        return self._apply_dark_theme(fig)

    # ── summary metrics ────────────────────────

    def summary_metrics(self, result: AnalysisResult) -> Dict[str, Any]:
        """Compute summary KPIs from a completed analysis run.

        Parameters
        ----------
        result:
            The ``AnalysisResult`` returned by the detection pipeline.

        Returns
        -------
        Dict[str, Any]
            Dictionary of KPI values suitable for dashboard cards::

                {
                    "total_violations": int,
                    "critical_count": int,
                    "moderate_count": int,
                    "minor_count": int,
                    "avg_confidence": float,
                    "max_confidence": float,
                    "processing_time": float,
                    "violations_by_type": dict,
                    "most_common_violation": str,
                }
        """
        violations: List[ViolationRecord] = result.violations

        severity_counts: Counter[Severity] = Counter(
            v.severity for v in violations
        )
        type_counts: Counter[ViolationType] = Counter(
            v.violation_type for v in violations
        )

        confidences: List[float] = [v.confidence for v in violations]
        avg_conf: float = float(np.mean(confidences)) if confidences else 0.0
        max_conf: float = float(np.max(confidences)) if confidences else 0.0

        most_common: str = (
            type_counts.most_common(1)[0][0].value
            if type_counts
            else "None"
        )

        return {
            "total_violations": len(violations),
            "critical_count": severity_counts.get(Severity.CRITICAL, 0),
            "moderate_count": severity_counts.get(Severity.MODERATE, 0),
            "minor_count": severity_counts.get(Severity.MINOR, 0),
            "avg_confidence": round(avg_conf, 4),
            "max_confidence": round(max_conf, 4),
            "processing_time": round(result.processing_time, 3),
            "violations_by_type": {
                vtype.value: count for vtype, count in type_counts.items()
            },
            "most_common_violation": most_common,
        }

    # ── private theming helper ─────────────────

    def _apply_dark_theme(self, fig: go.Figure) -> go.Figure:
        """Apply a consistent dark theme to any Plotly figure.

        Updates the layout in-place and returns the same figure for
        fluent chaining.

        Parameters
        ----------
        fig:
            A ``plotly.graph_objects.Figure`` to style.

        Returns
        -------
        go.Figure
            The same figure with dark-theme layout applied.
        """
        fig.update_layout(
            # Backgrounds
            paper_bgcolor=self.bg_primary,
            plot_bgcolor=self.bg_primary,
            # Typography
            font=dict(
                family=self.font_family,
                color=self.text_primary,
                size=13,
            ),
            # Margins — compact for dashboard embedding
            margin=_CHART_MARGIN,
            # Hover styling
            hoverlabel=dict(
                bgcolor=self.bg_card,
                font=dict(
                    family=self.font_family,
                    size=12,
                    color=self.text_primary,
                ),
                bordercolor=COLORS["border"],
            ),
            # Disable Plotly modebar for cleaner look in embedded dashboards
            modebar=dict(
                bgcolor="rgba(0,0,0,0)",
                color=_hex_to_rgba(self.text_secondary, 0.5),
                activecolor=self.accent_blue,
            ),
        )

        return fig
