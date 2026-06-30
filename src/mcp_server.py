"""``run-mcp`` entry point: MCP server exposing Pandas tools over the Wine Reviews data.

Tools are named for intent (no generic exec tool). The DataFrame is loaded once at
startup and held in memory.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mcp_server")

mcp = FastMCP("wine-data")

# Columns the structured tools reason about.
CATEGORICAL = ["country", "province", "variety", "winery"]
NUMERIC = ["points", "price"]


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
    """Load and cache the DataFrame on first tool call (and at startup)."""
    if not config.DATA_CSV.exists():
        raise FileNotFoundError(
            f"{config.DATA_CSV} not found. Run `uv run download-data` first."
        )
    logger.info("Loading DataFrame from %s ...", config.DATA_CSV)
    df = pd.read_csv(config.DATA_CSV)
    logger.info("Loaded %d rows", len(df))
    return df


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """Apply a filters dict shaped like ``filter_wines`` keyword args."""
    if not filters:
        return df
    out = df
    eq = {"country": "country", "variety": "variety", "province": "province"}
    for key, col in eq.items():
        val = filters.get(key)
        if val:
            out = out[out[col].astype(str).str.casefold() == str(val).casefold()]
    if filters.get("min_points") is not None:
        out = out[out["points"] >= filters["min_points"]]
    if filters.get("max_points") is not None:
        out = out[out["points"] <= filters["max_points"]]
    if filters.get("min_price") is not None:
        out = out[out["price"] >= filters["min_price"]]
    if filters.get("max_price") is not None:
        out = out[out["price"] <= filters["max_price"]]
    return out


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert rows to JSON-friendly dicts (NaN -> None) including the CSV row_id."""
    cols = [c for c in ("title", *CATEGORICAL, *NUMERIC) if c in df.columns]
    out = df[cols].where(pd.notna(df[cols]), None)
    records = out.to_dict(orient="records")
    for rec, idx in zip(records, df.index):
        rec["row_id"] = int(idx)
    return records


@mcp.tool()
def describe_schema() -> dict[str, Any]:
    """Return column names, dtypes, numeric ranges, and distinct counts.

    Call this first when unsure which columns or values are available.
    """
    df = _df()
    schema: dict[str, Any] = {
        "n_rows": int(len(df)),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    schema["categorical"] = {
        col: {
            "distinct": int(df[col].nunique(dropna=True)),
            "examples": df[col].dropna().astype(str).unique()[:10].tolist(),
        }
        for col in CATEGORICAL
        if col in df.columns
    }
    schema["numeric"] = {
        col: {
            "min": (None if df[col].dropna().empty else float(df[col].min())),
            "max": (None if df[col].dropna().empty else float(df[col].max())),
        }
        for col in NUMERIC
        if col in df.columns
    }
    return schema


@mcp.tool()
def filter_wines(
    country: str | None = None,
    variety: str | None = None,
    province: str | None = None,
    min_points: int | None = None,
    max_points: int | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return wines matching the given structured filters (as a list of row dicts)."""
    filters = {
        "country": country,
        "variety": variety,
        "province": province,
        "min_points": min_points,
        "max_points": max_points,
        "min_price": min_price,
        "max_price": max_price,
    }
    result = _apply_filters(_df(), filters)
    return _records(result.head(limit))


@mcp.tool()
def aggregate_wines(
    group_by: str,
    metric: str,
    agg: str = "mean",
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Group by a categorical column and aggregate a numeric metric.

    Example: ``group_by="country", metric="price", agg="mean"``. ``agg`` is one of
    ``mean``, ``median``, ``min``, ``max``, ``count``, ``sum``.
    """
    df = _apply_filters(_df(), filters)
    if group_by not in df.columns or metric not in df.columns:
        raise ValueError(f"Unknown column(s): group_by={group_by!r}, metric={metric!r}")
    grouped = df.groupby(group_by)[metric].agg(agg).reset_index()
    grouped = grouped.sort_values(by=metric, ascending=False)
    return grouped.where(pd.notna(grouped), None).to_dict(orient="records")


@mcp.tool()
def top_wines(
    by: str = "points",
    filters: dict[str, Any] | None = None,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Return the top ``n`` wines ranked by a numeric column (default ``points``)."""
    df = _apply_filters(_df(), filters)
    if by not in df.columns:
        raise ValueError(f"Unknown ranking column: {by!r}")
    ranked = df.sort_values(by=by, ascending=False).head(n)
    return _records(ranked)


def main() -> None:
    """Run the MCP server over stdio."""
    _df()  # Load eagerly so startup failures surface immediately.
    mcp.run()


if __name__ == "__main__":
    main()
