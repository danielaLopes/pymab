from __future__ import annotations

from typing import Any


def get_default_layout(
    title: str, xaxis_title: str, yaxis_title: str
) -> dict[str, Any]:
    return {
        "title": title,
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "font": {"size": 12},
        "legend": {
            "font": {"size": 10},
            "itemsizing": "constant",
            "traceorder": "normal",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        "height": 900,
        "width": 1000,
    }


def get_color_sequence() -> list[str]:
    return [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]


def get_line_style(color: str) -> dict[str, Any]:
    return {"color": color}


def get_marker_style(color: str) -> dict[str, Any]:
    return {"color": color, "size": 10}
