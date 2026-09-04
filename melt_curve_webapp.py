"""
melt_curve_webapp.py
=====================
Shareable WEB VERSION of the melt curve viewer. Unlike the desktop script,
this is meant to be deployed once (see DEPLOYMENT_INSTRUCTIONS.md) so that
anyone with the link can open it in a browser, drag in their own CFX melt
curve export, and explore it interactively -- no Python or installation
needed on their end. Only whoever deploys it needs to touch Python, and
only once, during setup.

Each visitor's uploaded files stay in their own browser session (via
dcc.Store) and are never saved to disk or shared between users.
"""

import base64
import io
import re
import zipfile

import dash
from dash import dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# =============================================================================
# File parsing -- same robustness as the desktop version, adapted to work on
# in-memory bytes from a browser upload instead of a local file path.
# =============================================================================

def _normalize_well(well):
    """Normalize a well ID to a consistent 'A01' style ('A1' -> 'A01', etc.)."""
    well = str(well).strip().upper()
    m = re.match(r"^([A-P])0*(\d{1,2})$", well)
    if m:
        letter, num = m.group(1), int(m.group(2))
        return f"{letter}{num:02d}"
    return well


def _canonical_ooxml_name(name):
    """
    Return the correctly-cased standard OOXML part name for `name`, if it's
    a case-mismatched variant some instrument software (e.g. Bio-Rad CFX)
    is known to produce. Otherwise returns None (leave the name alone).
    """
    fixed = {
        "[content_types].xml": "[Content_Types].xml",
        "xl/sharedstrings.xml": "xl/sharedStrings.xml",
        "xl/workbook.xml": "xl/workbook.xml",
        "xl/styles.xml": "xl/styles.xml",
        "xl/_rels/workbook.xml.rels": "xl/_rels/workbook.xml.rels",
        "_rels/.rels": "_rels/.rels",
        "docprops/core.xml": "docProps/core.xml",
        "docprops/app.xml": "docProps/app.xml",
    }
    lower = name.lower()
    if lower in fixed:
        return fixed[lower]
    m = re.match(r"^xl/worksheets/(?:_rels/)?sheet(\d+)\.xml(\.rels)?$", lower)
    if m:
        num, rels = m.group(1), m.group(2) or ""
        prefix = "xl/worksheets/_rels/" if rels else "xl/worksheets/"
        return f"{prefix}sheet{num}.xml{rels}"
    return None


def _repair_malformed_xlsx_bytes(raw_bytes):
    """
    Some instrument exports produce .xlsx files with a malformed internal
    zip structure (lowercased OOXML part names, backslash paths). Excel and
    Numbers open these fine; Python's strict zipfile module rejects them.
    Returns a repaired BytesIO, or None if no repair was needed/possible.
    """
    try:
        src = zipfile.ZipFile(io.BytesIO(raw_bytes), "r")
    except zipfile.BadZipFile:
        return None

    names = src.namelist()
    has_backslashes = any("\\" in n for n in names)
    has_case_mismatch = any(
        _canonical_ooxml_name(n.replace("\\", "/")) is not None
        and _canonical_ooxml_name(n.replace("\\", "/")) != n.replace("\\", "/")
        for n in names
    )
    if not (has_backslashes or has_case_mismatch):
        return None

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as out:
        for info in src.infolist():
            name = info.filename.replace("\\", "/")
            canonical = _canonical_ooxml_name(name)
            if canonical is not None:
                name = canonical
            out.writestr(name, src.read(info.filename))
    out_buf.seek(0)
    return out_buf


def _read_export_bytes(raw_bytes):
    """
    Robustly read a CFX-style export from raw bytes, regardless of what its
    extension claims. Tries a malformed-zip repair, then real .xlsx, then
    legacy .xls, HTML, and delimited text, in that order.
    """
    errors = []

    try:
        repaired = _repair_malformed_xlsx_bytes(raw_bytes)
        if repaired is not None:
            return pd.read_excel(repaired, sheet_name=0, engine="openpyxl")
    except Exception as e:
        errors.append(f"zip repair attempt: {e}")

    try:
        return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=0, engine="openpyxl")
    except Exception as e:
        errors.append(f"openpyxl (.xlsx): {e}")

    try:
        return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=0, engine="xlrd")
    except Exception as e:
        errors.append(f"xlrd (.xls): {e}")

    try:
        tables = pd.read_html(io.BytesIO(raw_bytes))
        if tables:
            return tables[0]
    except Exception as e:
        errors.append(f"HTML table: {e}")

    try:
        return pd.read_csv(io.BytesIO(raw_bytes), sep="\t")
    except Exception as e:
        errors.append(f"tab-delimited text: {e}")

    try:
        return pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        errors.append(f"comma-delimited text: {e}")

    raise ValueError(
        "Could not read this file as .xlsx, .xls, HTML, or delimited text.\n"
        "Attempts tried:\n" + "\n".join(errors)
    )


def _decode_upload(contents):
    """Decode a dcc.Upload 'contents' data-URI string into raw bytes."""
    _content_type, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


def parse_rfu_upload(contents):
    """
    Parse an uploaded raw melt curve export. Returns (temperature_list,
    curves_dict) where curves_dict is {well: [values...]}, JSON-serializable
    for storing in a dcc.Store.
    """
    raw_bytes = _decode_upload(contents)
    df = _read_export_bytes(raw_bytes)

    temp_col = next((c for c in df.columns if "temp" in str(c).strip().lower()), None)
    if temp_col is None:
        raise ValueError(
            f"Couldn't find a Temperature column. Found columns: {list(df.columns)}. "
            "Make sure this is the RAW melt curve export, not a summary/peak results file."
        )

    well_pattern = re.compile(r"^[A-P](0?[1-9]|1[0-9]|2[0-4])$")
    well_cols = [c for c in df.columns if well_pattern.match(str(c).strip())]
    if not well_cols:
        raise ValueError(
            f"No well-labeled columns (A01, A02, ...) found. Found columns: {list(df.columns)}"
        )

    temperature = df[temp_col].astype(float).tolist()
    curves = {_normalize_well(c): df[c].astype(float).tolist() for c in well_cols}
    return temperature, curves


def parse_sample_map_upload(contents):
    """
    Parse an uploaded Peak Results / Cq Results export. Returns
    {well: sample_name}.
    """
    raw_bytes = _decode_upload(contents)
    df = _read_export_bytes(raw_bytes)

    well_col = next((c for c in df.columns if str(c).strip().lower() == "well"), None)
    sample_col = next((c for c in df.columns if str(c).strip().lower() == "sample"), None)
    if well_col is None or sample_col is None:
        raise ValueError(
            f"Couldn't find 'Well' and 'Sample' columns. Found columns: {list(df.columns)}"
        )

    mapping = {}
    for _, row in df.iterrows():
        well, sample = row[well_col], row[sample_col]
        if pd.notna(well) and pd.notna(sample):
            mapping[_normalize_well(well)] = str(sample).strip()
    return mapping


# =============================================================================
# Dash app
# =============================================================================

app = dash.Dash(__name__)
server = app.server  # exposed for gunicorn / hosting platforms
app.title = "Melt Curve Viewer"

UPLOAD_STYLE = {
    "width": "100%",
    "height": "70px",
    "lineHeight": "70px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "8px",
    "textAlign": "center",
    "fontFamily": "sans-serif",
    "marginBottom": "8px",
}

app.layout = html.Div(
    [
        html.H2("Melt Curve Viewer", style={"fontFamily": "sans-serif"}),
        html.P(
            "Upload your raw melt curve export below (required: Temperature "
            "column + one column per well). Optionally also upload a Peak "
            "Results / Cq Results export to label wells with sample names "
            "instead of just well IDs.",
            style={"fontFamily": "sans-serif", "color": "#555"},
        ),
        html.Div(
            [
                html.Label(
                    "1. Melt curve RFU export (required)",
                    style={"fontFamily": "sans-serif", "fontWeight": "bold"},
                ),
                dcc.Upload(
                    id="upload-rfu",
                    children=html.Div(["Drag and drop, or ", html.A("select a file")]),
                    style=UPLOAD_STYLE,
                    multiple=False,
                ),
                html.Div(id="rfu-upload-status", style={"fontFamily": "sans-serif", "marginBottom": "16px"}),

                html.Label(
                    "2. Sample name lookup (optional)",
                    style={"fontFamily": "sans-serif", "fontWeight": "bold"},
                ),
                dcc.Upload(
                    id="upload-samplemap",
                    children=html.Div(["Drag and drop, or ", html.A("select a file")]),
                    style=UPLOAD_STYLE,
                    multiple=False,
                ),
                html.Div(id="samplemap-upload-status", style={"fontFamily": "sans-serif", "marginBottom": "16px"}),

                dcc.Checklist(
                    id="derivative-checkbox",
                    options=[
                        {
                            "label": " My file has raw RFU values (compute -d(RFU)/dT here)",
                            "value": "derivative",
                        }
                    ],
                    value=["derivative"],
                    style={"fontFamily": "sans-serif", "marginBottom": "16px"},
                ),
            ]
        ),
        dcc.Graph(id="melt-graph", figure=go.Figure()),
        html.Div(
            [
                html.Button("Select All", id="select-all-btn", n_clicks=0,
                            style={"marginRight": "10px", "fontFamily": "sans-serif"}),
                html.Button("Unselect All", id="unselect-all-btn", n_clicks=0,
                            style={"fontFamily": "sans-serif"}),
            ],
            style={"margin": "10px 0"},
        ),
        html.P(
            "Click a column header to sort. Type in the boxes under 'Well' or "
            "'Sample' to search/filter. Check rows to select which wells "
            "appear in the plot above (select none to show all).",
            style={"fontFamily": "sans-serif", "color": "#555"},
        ),
        dash_table.DataTable(
            id="well-table",
            columns=[{"name": "Well", "id": "well"}, {"name": "Sample", "id": "sample"}],
            data=[],
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            selected_rows=[],
            page_size=20,
            style_table={"maxHeight": "500px", "overflowY": "auto"},
            style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
            style_header={"fontWeight": "bold"},
        ),
        # Per-session storage -- lives only in this visitor's browser, never
        # written to disk or shared with other visitors.
        dcc.Store(id="curve-store"),
        dcc.Store(id="samplemap-store"),
    ],
    style={"maxWidth": "1100px", "margin": "auto", "padding": "24px"},
)


@app.callback(
    Output("curve-store", "data"),
    Output("rfu-upload-status", "children"),
    Input("upload-rfu", "contents"),
    State("upload-rfu", "filename"),
    prevent_initial_call=True,
)
def handle_rfu_upload(contents, filename):
    if contents is None:
        return dash.no_update, ""
    try:
        temperature, curves = parse_rfu_upload(contents)
        return (
            {"temperature": temperature, "curves": curves},
            f"Loaded '{filename}': {len(curves)} wells.",
        )
    except Exception as e:
        return dash.no_update, f"Error reading '{filename}': {e}"


@app.callback(
    Output("samplemap-store", "data"),
    Output("samplemap-upload-status", "children"),
    Input("upload-samplemap", "contents"),
    State("upload-samplemap", "filename"),
    prevent_initial_call=True,
)
def handle_samplemap_upload(contents, filename):
    if contents is None:
        return dash.no_update, ""
    try:
        mapping = parse_sample_map_upload(contents)
        return mapping, f"Loaded '{filename}': {len(mapping)} sample names."
    except Exception as e:
        return dash.no_update, f"Error reading '{filename}': {e}"


@app.callback(
    Output("well-table", "data"),
    Input("curve-store", "data"),
    Input("samplemap-store", "data"),
)
def update_table(curve_data, sample_map):
    if not curve_data:
        return []
    sample_map = sample_map or {}
    wells = sorted(curve_data["curves"].keys())
    return [{"well": w, "sample": sample_map.get(w, "")} for w in wells]


@app.callback(
    Output("melt-graph", "figure"),
    Input("well-table", "selected_rows"),
    Input("well-table", "derived_virtual_data"),
    Input("curve-store", "data"),
    Input("derivative-checkbox", "value"),
)
def update_graph(selected_rows, virtual_data, curve_data, derivative_option):
    if not curve_data:
        return go.Figure()

    temperature = np.array(curve_data["temperature"])
    curves = curve_data["curves"]
    compute_deriv = "derivative" in (derivative_option or [])

    if selected_rows and virtual_data:
        wells_to_plot = [virtual_data[i]["well"] for i in selected_rows if i < len(virtual_data)]
    else:
        wells_to_plot = sorted(curves.keys())

    fig = go.Figure()
    palette = px.colors.qualitative.Alphabet
    for i, well in enumerate(wells_to_plot):
        y = np.array(curves[well])
        if compute_deriv:
            y = -np.gradient(y, temperature)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=temperature, y=y, mode="lines", name=well,
                line=dict(color=color, width=1.5), opacity=0.85,
                hovertemplate=f"Well: {well}<br>Temp: %{{x:.1f}} C<br>Value: %{{y:.1f}}<extra></extra>",
            )
        )

    title_suffix = f"{len(wells_to_plot)} wells selected" if (selected_rows) else f"showing all {len(wells_to_plot)} wells"
    fig.update_layout(
        title=f"Melt Peak -- {title_suffix}",
        xaxis_title="Temperature, Celsius",
        yaxis_title="-d(RFU)/dT" if compute_deriv else "RFU",
        template="plotly_white",
        hovermode="closest",
        legend=dict(font=dict(size=9)),
        height=550,
    )
    return fig


@app.callback(
    Output("well-table", "selected_rows"),
    Input("select-all-btn", "n_clicks"),
    Input("unselect-all-btn", "n_clicks"),
    State("well-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def select_or_unselect_all(select_clicks, unselect_clicks, virtual_data):
    triggered = ctx.triggered_id
    if triggered == "select-all-btn":
        return list(range(len(virtual_data))) if virtual_data else []
    elif triggered == "unselect-all-btn":
        return []
    return dash.no_update


if __name__ == "__main__":
    # For local testing only. In production (once deployed), the hosting
    # platform runs this via gunicorn using the `server` object above, not
    # this block.
    app.run(debug=False, host="0.0.0.0", port=8050)
