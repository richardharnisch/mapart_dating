import re
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------- Helpers -----------------------
def to_csv_export_url(url: str) -> str:
    """
    Accepts either:
      - a direct CSV export URL, or
      - a standard Google Sheets URL like
        https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit#gid=<GID>
    and returns a CSV export URL.
    """
    # Already a CSV export? just return it.
    if "export?format=csv" in url:
        return url

    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)/", url)
    gid_match = re.search(r"[#?&]gid=([0-9]+)", url)

    if m:
        sheet_id = m.group(1)
        gid = gid_match.group(1) if gid_match else "0"
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    # Otherwise, hope it's a plain CSV URL or raw file
    return url


def guess_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to guess the ID and Date columns.
    Returns (id_col, date_col) and falls back to the first numeric + first date-like.
    """
    cols = [c.strip() for c in df.columns.astype(str)]
    df = df.set_axis(cols, axis=1)

    # Heuristics
    id_candidates = [
        c for c in cols if c.lower() in {"id", "index"} or "id" in c.lower()
    ]
    date_candidates = [c for c in cols if "date" in c.lower()]

    id_col = id_candidates[0] if id_candidates else None
    date_col = date_candidates[0] if date_candidates else None

    # Fallbacks
    if id_col is None:
        # pick first numeric-looking column
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                id_col = c
                break
    if date_col is None:
        # pick first that successfully parses many rows as dates
        for c in cols:
            try:
                pd.to_datetime(df[c], errors="raise", infer_datetime_format=True)
                date_col = c
                break
            except Exception:
                continue

    if id_col is None or date_col is None:
        raise ValueError(
            "Couldn't automatically detect the ID or Date column. "
            "Please select them manually in the sidebar."
        )

    return id_col, date_col


def interpolate_date(df: pd.DataFrame, id_col: str, date_col: str, q_id: float):
    """
    Returns (status, result_dict or message)
    status: 'ok', 'exact', or 'error'
    - 'exact': the ID exists; returns exact date
    - 'ok': interpolated between neighbors
    - 'error': out of range or invalid
    """
    # Clean/sort
    work = df[[id_col, date_col]].copy()

    # Coerce ID to numeric
    work[id_col] = pd.to_numeric(work[id_col], errors="coerce")
    work = work.dropna(subset=[id_col, date_col])

    # Parse dates
    work[date_col] = pd.to_datetime(
        work[date_col], errors="coerce", infer_datetime_format=True
    )
    work = work.dropna(subset=[date_col])

    # Drop duplicates by ID (keep first)
    work = work.drop_duplicates(subset=[id_col]).sort_values(id_col)
    ids = work[id_col].to_numpy()
    dates = work[date_col].to_numpy()

    if len(ids) < 2:
        return "error", "Need at least two rows to interpolate."

    # Range check
    min_id, max_id = float(ids[0]), float(ids[-1])
    if q_id < min_id or q_id > max_id:
        return "error", f"ID {q_id:g} is outside the range [{min_id:g}, {max_id:g}]."

    # Exact match?
    exact_mask = ids == q_id
    if exact_mask.any():
        exact_dt = pd.to_datetime(work.loc[work[id_col] == q_id, date_col].iloc[0])
        return "exact", {
            "date": exact_dt,
            "lower_id": q_id,
            "lower_date": exact_dt,
            "upper_id": q_id,
            "upper_date": exact_dt,
        }

    # Interpolate on POSIX seconds
    # Convert numpy datetime64[ns] to seconds
    secs = np.array([pd.Timestamp(d).timestamp() for d in dates], dtype=float)

    interp_sec = np.interp(q_id, ids.astype(float), secs)
    interp_dt = pd.to_datetime(interp_sec, unit="s")

    # Neighbors
    idx = np.searchsorted(ids, q_id)
    lo_i = max(0, idx - 1)
    hi_i = min(len(ids) - 1, idx)

    return "ok", {
        "date": interp_dt,
        "lower_id": float(ids[lo_i]),
        "lower_date": pd.to_datetime(dates[lo_i]),
        "upper_id": float(ids[hi_i]),
        "upper_date": pd.to_datetime(dates[hi_i]),
    }


# ----------------------- UI -----------------------
st.set_page_config(page_title="Mapart Carbon Dating", page_icon="📈", layout="centered")

st.title("📈 Mapart Carbon Dating: Interactive Version")

query_id = st.number_input("Enter ID to date", value=604, step=1, format="%d")
run = st.button("Interpolate")

with st.expander("Customize Data Source"):
    st.markdown(
        """
    Provide a Google Sheets link *or* a direct CSV link.

    **Tip:** If you're pasting a normal Google Sheets URL, make sure the sheet is shared as **Anyone with the link → Viewer**.
    This app will automatically convert it to a CSV export link.
    """
    )

    with st.expander("Example sheet format"):
        st.code(
            "ID,Exact Date Month/Day/year\n" "604,12/1/2019\n" "1047,1/1/2020\n" "…",
            language="text",
        )

    sheet_url = st.text_input(
        "Google Sheets or CSV URL",
        placeholder="Paste link here…",
        value="https://docs.google.com/spreadsheets/d/1vpDEIZr7sR7ASlCm3qmscsSiCi9oz6yzr8oJV8He_L0/edit?gid=356591123#gid=356591123",
    )

    with st.sidebar:
        st.header("Column Options")
        st.caption("If auto-detection fails, pick columns manually.")
        manual = st.checkbox("Select columns manually", value=False)
        id_col_manual = None
        date_col_manual = None

if sheet_url:
    csv_url = to_csv_export_url(sheet_url)

    try:
        df = pd.read_csv(csv_url)
        # Show a quick preview
        with st.expander("Data Preview"):
            st.dataframe(df.head(10))

        if manual:
            id_col_manual = st.sidebar.selectbox("ID column", df.columns, index=0)
            date_col_manual = st.sidebar.selectbox(
                "Date column", df.columns, index=min(1, len(df.columns) - 1)
            )
            id_col, date_col = id_col_manual, date_col_manual
        else:
            try:
                id_col, date_col = guess_columns(df)
            except ValueError as e:
                st.warning(str(e))
                manual = True
                id_col_manual = st.sidebar.selectbox("ID column", df.columns, index=0)
                date_col_manual = st.sidebar.selectbox(
                    "Date column", df.columns, index=min(1, len(df.columns) - 1)
                )
                id_col, date_col = id_col_manual, date_col_manual

        st.success(f"Using ID column: **{id_col}**, Date column: **{date_col}**")

        if run:
            status, result = interpolate_date(df, id_col, date_col, float(query_id))
            if status == "error":
                st.error(result)
            else:
                dt = result["date"]
                human = dt.strftime("%Y-%m-%d")
                if status == "exact":
                    st.success(f"Exact match → **{human}**")
                else:
                    st.info(f"Interpolated date → **{human}**")

                with st.expander("Details"):
                    st.write(
                        pd.DataFrame(
                            {
                                "Neighbor": ["Lower", "Upper"],
                                "ID": [result["lower_id"], result["upper_id"]],
                                "Date": [
                                    result["lower_date"].strftime("%Y-%m-%d"),
                                    result["upper_date"].strftime("%Y-%m-%d"),
                                ],
                            }
                        )
                    )

    except Exception as e:
        st.error(
            "Couldn't load the data. Check that the link is accessible (Anyone with the link → Viewer) "
            "and that the sheet contains an ID column and a date column."
        )
        st.exception(e)
else:
    st.caption("Paste a link above to begin.")
