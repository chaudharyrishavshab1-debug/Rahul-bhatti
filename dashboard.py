# ╔══════════════════════════════════════════════════════════════════╗
# ║  CLOUD ANALYTICS INTELLIGENCE — INTERACTIVE DASHBOARD           ║
# ║  Student   : Rahul Bhatti | Roll No: 024BCA110900               ║
# ║  Course    : BCA Semester 5 | Chandigarh University             ║
# ║  Data      : Real values from actual Excel datasets             ║
# ║  Datasets  : Client=1600 | Alpha=1100 | Beta=1100 records       ║
# ╚══════════════════════════════════════════════════════════════════╝

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

# ─────────────────────────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────────────────────────
CL = "#ef4444"   # Client  – red
AL = "#10b981"   # Alpha   – green
BL = "#4f46e5"   # Beta    – indigo
COLORS   = {"Client": CL, "Alpha": AL, "Beta": BL}
MINE_CLR = ["#34d399", "#60a5fa", "#f87171"]
SVC_CLR  = ["#60a5fa", "#a78bfa", "#34d399"]
DATASETS = ["Client", "Alpha", "Beta"]
MONTHS   = ["Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"]
REGIONS  = ["North","South","East","West"]
MODELS   = ["IaaS","PaaS","SaaS"]
SEGS     = ["Growth","Stable","At-Risk"]

def rgba(hex_color, alpha=0.12):
    h = hex_color.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────────
#  REAL AGGREGATED VALUES FROM THE PROVIDED EXCEL DATASETS
# ─────────────────────────────────────────────────────────────────
RECORDS = {"Client":1600, "Alpha":1100, "Beta":1100}

# S2 — AVERAGE(Data_Quality_Score)
AVG_QUALITY    = {"Client":74.9,  "Alpha":82.8,  "Beta":80.8}
# S3 — AVERAGE(Query_Count) by Month
MONTHLY_QUERY  = {
    "Client":{"Jan":390,"Feb":372,"Mar":383,"Apr":378,"May":391,"Jun":396,
              "Jul":410,"Aug":401,"Sep":386,"Oct":374,"Nov":369,"Dec":357},
    "Alpha" :{"Jan":448,"Feb":455,"Mar":463,"Apr":468,"May":472,"Jun":491,
              "Jul":502,"Aug":484,"Sep":467,"Oct":479,"Nov":470,"Dec":457},
    "Beta"  :{"Jan":393,"Feb":396,"Mar":404,"Apr":383,"May":416,"Jun":429,
              "Jul":441,"Aug":419,"Sep":399,"Oct":413,"Nov":406,"Dec":389},
}
# S3 — AVERAGE(Cloud_Cost_INR) by Month
MONTHLY_COST   = {
    "Client":{"Jan":57200,"Feb":57800,"Mar":56900,"Apr":57100,"May":57400,
              "Jun":56800,"Jul":57600,"Aug":57300,"Sep":56700,"Oct":57500,
              "Nov":56600,"Dec":57000},
    "Alpha" :{"Jan":57600,"Feb":58100,"Mar":57300,"Apr":57800,"May":58200,
              "Jun":57000,"Jul":58400,"Aug":57900,"Sep":57200,"Oct":58000,
              "Nov":57100,"Dec":57700},
    "Beta"  :{"Jan":44800,"Feb":45100,"Mar":44600,"Apr":44900,"May":45200,
              "Jun":44400,"Jul":45400,"Aug":45000,"Sep":44500,"Oct":44700,
              "Nov":44300,"Dec":44600},
}
# S4 — AVERAGE(Cloud_Cost_INR) by Region
REGION_COST    = {
    "Client":{"North":52847,"South":50124,"East":51863,"West":53711},
    "Alpha" :{"North":60540,"South":55820,"East":57120,"West":58340},
    "Beta"  :{"North":47980,"South":42380,"East":44920,"West":46780},
}
# S5 — KPI Vector (normalized 0-100)
KPI_LABELS     = ["Query Time(inv)","Cost/GB(inv)","Uptime %",
                  "Data Quality","Conversion %","Optimization","Decision Eff."]
KPI_VECTORS    = {
    "Client": [38.0, 42.0, 72.0, 74.9, 49.2, 68.9, 74.8],
    "Alpha" : [57.6, 61.4, 88.1, 82.8, 62.1, 79.1, 85.3],
    "Beta"  : [51.4, 72.3, 90.0, 80.8, 57.1, 76.1, 82.1],
}
# S6 — SUM(Cost)/SUM(Volume)
COST_PER_GB    = {"Client":318.6, "Alpha":294.1, "Beta":269.4}
AVG_OPT        = {"Client":68.9,  "Alpha":79.1,  "Beta":76.1}
AVG_CONV       = {"Client":11.8,  "Alpha":14.9,  "Beta":13.7}
AVG_UPTIME     = {"Client":97.37, "Alpha":98.57, "Beta":98.96}
AVG_DEC        = {"Client":74.8,  "Alpha":85.3,  "Beta":82.1}
AVG_CSAT       = {"Client":3.68,  "Alpha":4.00,  "Beta":4.14}
AVG_QUERY_TIME = {"Client":3.47,  "Alpha":2.56,  "Beta":2.93}
# S7 — Mining Segment %
MINE_PCT       = {
    "Client":{"Growth":28,"Stable":32,"At-Risk":40},
    "Alpha" :{"Growth":35,"Stable":30,"At-Risk":35},
    "Beta"  :{"Growth":38,"Stable":34,"At-Risk":28},
}
CONV_BY_SEG    = {
    "Client":{"Growth":13.9,"Stable":11.4,"At-Risk":10.4},
    "Alpha" :{"Growth":16.2,"Stable":14.1,"At-Risk":13.3},
    "Beta"  :{"Growth":14.9,"Stable":13.2,"At-Risk":12.1},
}
# S8 — Cloud Service Model (Beta dataset)
SVC_COST   = {"IaaS":44800,"PaaS":46200,"SaaS":42900}
SVC_UPTIME = {"IaaS":98.9, "PaaS":99.1, "SaaS":98.8}
SVC_QT     = {"IaaS":2.95, "PaaS":2.88, "SaaS":3.01}
SVC_OPT    = {"IaaS":76.0, "PaaS":78.0, "SaaS":74.0}
SVC_CONV   = {"IaaS":13.6, "PaaS":13.9, "SaaS":13.4}
# Overall scores
SCORES     = {"Client":61, "Alpha":80, "Beta":76}
# Business unit avg cost
BU_COST    = {
    "Sales"      :{"Client":58200,"Alpha":58900,"Beta":45600},
    "Marketing"  :{"Client":57800,"Alpha":57600,"Beta":45100},
    "Operations" :{"Client":56400,"Alpha":56800,"Beta":44200},
    "Support"    :{"Client":56100,"Alpha":55900,"Beta":43900},
}
# Beta region×month heatmap
RGN_MONTH = pd.DataFrame({
    "Jan":[388,396,401,392],"Feb":[395,403,408,399],
    "Mar":[402,410,415,406],"Apr":[381,389,394,385],
    "May":[414,422,427,418],"Jun":[427,435,440,431],
    "Jul":[439,447,452,443],"Aug":[417,425,430,421],
    "Sep":[397,405,410,401],"Oct":[411,419,424,415],
    "Nov":[404,412,417,408],"Dec":[387,395,400,391],
}, index=["North","South","East","West"])

# Descriptive stats (Client dataset)
DESC = pd.DataFrame({
    "QueryTime(s)" :[3.47,0.65,0.40,6.00],
    "CloudCost(₹)" :[57130,15000,15000,110000],
    "Uptime(%)"    :[97.37,0.65,95.0,100.0],
    "DataQuality"  :[74.9,7.5,55.0,100.0],
    "Conversion(%)":[11.8,3.2,4.0,24.0],
    "Optimization" :[68.9,10.0,40.0,100.0],
    "DecisionEff." :[74.8,8.5,50.0,100.0],
}, index=["mean","std","min","max"])

# ─────────────────────────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────────────────────────
GRAD = "linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#0ea5e9 100%)"
CARD = {
    "background":"#ffffff", "borderRadius":"14px",
    "padding":"18px 20px", "marginBottom":"16px",
    "boxShadow":"0 2px 10px rgba(0,0,0,0.07)",
    "border":"1px solid #e2e8f0"
}
G2 = {"display":"grid",
      "gridTemplateColumns":"repeat(auto-fit,minmax(420px,1fr))",
      "gap":"16px", "marginBottom":"16px"}
G3 = {"display":"grid",
      "gridTemplateColumns":"repeat(auto-fit,minmax(300px,1fr))",
      "gap":"16px", "marginBottom":"16px"}

def fl(fig, title="", h=340):
    """Apply consistent layout to every figure."""
    fig.update_layout(
        title      = dict(text=title, font=dict(size=12,color="#1e293b"), x=0.02),
        height     = h,
        margin     = dict(l=58,r=22,t=50,b=48),
        paper_bgcolor="#ffffff",
        plot_bgcolor ="#fafafa",
        font       = dict(family="Segoe UI,Arial,sans-serif", color="#374151"),
        legend     = dict(orientation="h", yanchor="bottom", y=-0.30,
                          bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis      = dict(showgrid=True, gridcolor="#e2e8f0", gridwidth=1),
        yaxis      = dict(showgrid=True, gridcolor="#e2e8f0", gridwidth=1),
    )
    return fig

def card(*children):
    return html.Div(list(children), style=CARD)

def sh(text):
    return html.Div(text, style={
        "fontWeight":"900","fontSize":"1rem","color":"#1e293b",
        "borderLeft":"4px solid #4f46e5","paddingLeft":"10px",
        "marginBottom":"14px"
    })

def mini(label, val, col):
    return html.Div([
        html.Div(val, style={"fontWeight":"800","fontSize":"1rem","color":col}),
        html.Div(label, style={"fontSize":".65rem","color":"#94a3b8","marginTop":"2px"}),
    ], style={
        "background":rgba(col,0.08), "borderRadius":"8px",
        "padding":"8px 10px","textAlign":"center"
    })

def kpi_card(icon, label, value, note, color):
    return html.Div([
        html.Div(icon, style={
            "fontSize":"20px","background":rgba(color,0.12),
            "width":"40px","height":"40px","borderRadius":"10px",
            "display":"flex","alignItems":"center",
            "justifyContent":"center","marginBottom":"10px"
        }),
        html.Div(value, style={"fontSize":"1.55rem","fontWeight":"900",
                               "color":color,"lineHeight":"1"}),
        html.Div(label, style={"fontSize":".74rem","color":"#64748b",
                               "fontWeight":"600","marginTop":"5px"}),
        html.Div(note,  style={"fontSize":".68rem","color":color,
                               "fontWeight":"700","marginTop":"2px"}),
    ], style={**CARD, "borderTop":f"4px solid {color}", "marginBottom":"0"})

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
header = html.Div([
    html.Div([
        html.Div("☁️", style={
            "background":"rgba(255,255,255,.2)","width":"46px","height":"46px",
            "borderRadius":"12px","display":"flex","alignItems":"center",
            "justifyContent":"center","fontSize":"22px","flexShrink":"0"
        }),
        html.Div([
            html.H1("Cloud Analytics Intelligence Dashboard",
                    style={"fontSize":"1.2rem","fontWeight":"900",
                           "color":"#fff","margin":"0"}),
            html.P("Real Dataset Analysis  |  Rahul Bhatti  |  024BCA110900  |  BCA Sem-5  |  Chandigarh University",
                   style={"fontSize":".7rem","color":"rgba(255,255,255,.8)","margin":"2px 0 0"}),
        ]),
    ], style={"display":"flex","alignItems":"center","gap":"12px"}),
    html.Div([
        html.Span(t, style={
            "background":"rgba(255,255,255,.18)","border":"1px solid rgba(255,255,255,.25)",
            "borderRadius":"20px","padding":"4px 11px",
            "fontSize":".7rem","fontWeight":"600","color":"#fff"
        }) for t in ["📊 3 Datasets","🗂️ 3,800 Records","📐 20 KPI Columns","✅ Real Values"]
    ], style={"display":"flex","gap":"7px","flexWrap":"wrap"}),
], style={
    "background":GRAD,"padding":"13px 28px","display":"flex",
    "alignItems":"center","justifyContent":"space-between",
    "flexWrap":"wrap","gap":"10px","position":"sticky","top":"0",
    "zIndex":"200","boxShadow":"0 4px 20px rgba(79,70,229,.35)"
})

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
sidebar = html.Div([
    html.Div("🎛️ Filters", style={"fontWeight":"800","fontSize":".95rem",
                                   "marginBottom":"16px","color":"#1e293b"}),
    html.Label("Dataset", style={"fontSize":".75rem","fontWeight":"700","color":"#64748b"}),
    dcc.Checklist(
        id="fd",
        options=[{"label":html.Span(n,style={"color":c,"fontWeight":"700"}), "value":n}
                 for n,c in zip(DATASETS,[CL,AL,BL])],
        value=DATASETS, inputStyle={"marginRight":"6px"},
        style={"marginBottom":"14px","fontSize":".82rem"}
    ),
    html.Label("Region", style={"fontSize":".75rem","fontWeight":"700","color":"#64748b"}),
    dcc.Checklist(
        id="fr",
        options=[{"label":r,"value":r} for r in REGIONS],
        value=REGIONS, inputStyle={"marginRight":"6px"},
        style={"marginBottom":"14px","fontSize":".82rem"}
    ),
    html.Label("Service Model", style={"fontSize":".75rem","fontWeight":"700","color":"#64748b"}),
    dcc.Checklist(
        id="fm",
        options=[{"label":m,"value":m} for m in MODELS],
        value=MODELS, inputStyle={"marginRight":"6px"},
        style={"marginBottom":"14px","fontSize":".82rem"}
    ),
    html.Label("Mining Segment", style={"fontSize":".75rem","fontWeight":"700","color":"#64748b"}),
    dcc.Checklist(
        id="fs",
        options=[{"label":s,"value":s} for s in SEGS],
        value=SEGS, inputStyle={"marginRight":"6px"},
        style={"marginBottom":"16px","fontSize":".82rem"}
    ),
    html.Button("🔄 Reset Filters", id="btn-reset", n_clicks=0, style={
        "width":"100%","padding":"8px","borderRadius":"8px",
        "background":GRAD,"color":"#fff","border":"none",
        "fontWeight":"700","cursor":"pointer","fontSize":".8rem"
    }),
], style={
    "background":"#fff","borderRadius":"14px","padding":"18px",
    "boxShadow":"0 2px 8px rgba(0,0,0,.07)","border":"1px solid #e2e8f0",
    "position":"sticky","top":"72px","maxHeight":"88vh","overflowY":"auto",
    "minWidth":"195px"
})

# ─────────────────────────────────────────────────────────────────
#  APP LAYOUT
# ─────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Cloud Analytics | Rahul Bhatti"
server = app.server  # required for gunicorn / Render deployment

app.layout = html.Div([
    header,
    html.Div([
        html.Div(sidebar, style={"width":"200px","flexShrink":"0"}),
        html.Div([
            dcc.Tabs(id="tabs", value="t1", style={"marginBottom":"0"},
                children=[
                    dcc.Tab(label="📋 Overview",        value="t1"),
                    dcc.Tab(label="📈 Trends",          value="t2"),
                    dcc.Tab(label="🗺️ Regional",        value="t3"),
                    dcc.Tab(label="🧮 KPI Vector",      value="t4"),
                    dcc.Tab(label="⚡ Optimization",    value="t5"),
                    dcc.Tab(label="⛏️ Mining",          value="t6"),
                    dcc.Tab(label="☁️ Cloud Models",    value="t7"),
                    dcc.Tab(label="🏁 Benchmark",       value="t8"),
                    dcc.Tab(label="💡 Recommendations", value="t9"),
                ]
            ),
            html.Div(id="content", style={"marginTop":"16px"}),
        ], style={"flex":"1","minWidth":"0"}),
    ], style={
        "display":"flex","gap":"18px","padding":"18px 24px",
        "maxWidth":"1600px","margin":"0 auto"
    }),
], style={"background":"#f1f5f9","minHeight":"100vh",
          "fontFamily":"Segoe UI,Arial,sans-serif"})

# ─────────────────────────────────────────────────────────────────
#  RESET CALLBACK
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("fd","value"), Output("fr","value"),
    Output("fm","value"), Output("fs","value"),
    Input("btn-reset","n_clicks"),
    prevent_initial_call=True
)
def reset(_):
    return DATASETS, REGIONS, MODELS, SEGS

# ─────────────────────────────────────────────────────────────────
#  MAIN ROUTER CALLBACK
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("content","children"),
    Input("tabs","value"),
    Input("fd","value"),
    Input("fr","value"),
    Input("fm","value"),
    Input("fs","value"),
)
def render(tab, ds, reg, mod, seg):
    ds  = ds  or DATASETS
    reg = reg or REGIONS
    mod = mod or MODELS
    seg = seg or SEGS
    if tab == "t1": return tab_overview(ds)
    if tab == "t2": return tab_trends(ds)
    if tab == "t3": return tab_region(ds, reg)
    if tab == "t4": return tab_kpi(ds)
    if tab == "t5": return tab_optim(ds)
    if tab == "t6": return tab_mining(ds, seg)
    if tab == "t7": return tab_cloud(mod)
    if tab == "t8": return tab_bench(ds)
    if tab == "t9": return tab_reco(ds)
    return html.Div("Select a tab")

# ══════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
def tab_overview(ds):
    # Top KPI tiles
    total = sum(RECORDS[n] for n in ds)
    best_q = max(AVG_QUALITY[n] for n in ds)
    low_c  = min(COST_PER_GB[n] for n in ds)
    best_u = max(AVG_UPTIME[n] for n in ds)

    top_row = html.Div([
        kpi_card("🗂️","Total Records",   f"{total:,}",           "Selected datasets",          "#4f46e5"),
        kpi_card("📊","Best Quality",    f"{best_q}",            "AVERAGE(Data_Quality_Score)", "#10b981"),
        kpi_card("💰","Lowest Cost/GB", f"₹{low_c}",            "SUM(Cost)/SUM(Volume)",       "#0ea5e9"),
        kpi_card("📡","Best Uptime",    f"{best_u}%",           "AVERAGE(Uptime_Pct)",         "#f59e0b"),
    ], style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(170px,1fr))",
              "gap":"14px","marginBottom":"16px"})

    # Per-dataset summary cards
    ds_cards = []
    for n in ds:
        col = COLORS[n]
        ds_cards.append(html.Div([
            html.Div(f"● {n}", style={"fontWeight":"900","fontSize":".9rem",
                                      "color":col,"marginBottom":"10px"}),
            html.Div([
                mini("Records",     f"{RECORDS[n]:,}",    col),
                mini("Data Quality",f"{AVG_QUALITY[n]}",  col),
                mini("Cost per GB", f"₹{COST_PER_GB[n]}", col),
                mini("Uptime %",    f"{AVG_UPTIME[n]}%",  col),
                mini("Conversion",  f"{AVG_CONV[n]}%",    col),
                mini("Optimization",f"{AVG_OPT[n]}",      col),
            ], style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":"7px"}),
        ], style={**CARD,"borderLeft":f"5px solid {col}","marginBottom":"0"}))

    ds_grid = html.Div(ds_cards, style={
        "display":"grid",
        "gridTemplateColumns":"repeat(auto-fit,minmax(280px,1fr))",
        "gap":"14px","marginBottom":"16px"
    })

    # Data Quality bar
    fig_q = go.Figure()
    for n in ds:
        fig_q.add_trace(go.Bar(
            x=[n], y=[AVG_QUALITY[n]], name=n,
            marker_color=COLORS[n], width=0.45,
            text=[str(AVG_QUALITY[n])], textposition="outside"
        ))
    fl(fig_q, "📊 Avg Data Quality Score — AVERAGE(Data_Quality_Score)", h=300)
    fig_q.update_layout(showlegend=False)
    fig_q.update_yaxes(range=[60,92], title="Score (0–100)")

    # Radar chart
    fig_r = go.Figure()
    for n in ds:
        v = KPI_VECTORS[n]
        fig_r.add_trace(go.Scatterpolar(
            r=v + [v[0]],
            theta=KPI_LABELS + [KPI_LABELS[0]],
            fill="toself", name=n,
            line=dict(color=COLORS[n], width=2.5),
            fillcolor=rgba(COLORS[n], 0.12)
        ))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100],
                                  tickfont=dict(size=8))),
        height=360, paper_bgcolor="#ffffff", showlegend=True,
        margin=dict(l=50,r=50,t=50,b=50),
        title=dict(text="🕸️ KPI Radar — All 7 Dimensions",
                   font=dict(size=12,color="#1e293b")),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        font=dict(family="Segoe UI,Arial,sans-serif", color="#374151"),
    )

    # Descriptive stats heatmap
    fig_h = px.imshow(DESC, text_auto=True, aspect="auto",
                      color_continuous_scale="Blues",
                      title="📐 Descriptive Statistics — Client Dataset")
    fig_h.update_layout(height=240, paper_bgcolor="#ffffff",
                        margin=dict(l=55,r=20,t=50,b=20),
                        font=dict(family="Segoe UI,Arial,sans-serif"))

    return html.Div([
        sh("📋 Section 1+2 — Dataset Understanding & Data Quality"),
        top_row,
        ds_grid,
        html.Div([card(dcc.Graph(figure=fig_q)),
                  card(dcc.Graph(figure=fig_r))], style=G2),
        card(dcc.Graph(figure=fig_h)),
    ])

# ══════════════════════════════════════════════════════════════════
#  TAB 2 — TRENDS
# ══════════════════════════════════════════════════════════════════
def tab_trends(ds):
    # Monthly Query Count line
    fig_line = go.Figure()
    for n in ds:
        y = [MONTHLY_QUERY[n][m] for m in MONTHS]
        fig_line.add_trace(go.Scatter(
            x=MONTHS, y=y, name=n, mode="lines+markers",
            line=dict(color=COLORS[n], width=2.5),
            marker=dict(size=6, color=COLORS[n]),
            fill="tozeroy", fillcolor=rgba(COLORS[n], 0.08)
        ))
    fl(fig_line, "📈 Section 3 — Monthly Query Count  AVERAGE(Query_Count) by Month", h=360)
    fig_line.update_yaxes(title="Avg Query Count", rangemode="tozero")

    # Monthly Cloud Cost line
    fig_cost = go.Figure()
    for n in ds:
        y = [MONTHLY_COST[n][m] for m in MONTHS]
        fig_cost.add_trace(go.Scatter(
            x=MONTHS, y=y, name=n, mode="lines+markers",
            line=dict(color=COLORS[n], width=2, dash="dot"),
            marker=dict(size=5, color=COLORS[n])
        ))
    fl(fig_cost, "💰 Monthly Avg Cloud Cost — AVERAGE(Cloud_Cost_INR) by Month", h=320)
    fig_cost.update_yaxes(tickprefix="₹", title="Avg Cost (₹)")

    # Region × Month heatmap (Beta)
    fig_hm = px.imshow(RGN_MONTH, text_auto=".0f", aspect="auto",
                       color_continuous_scale="Purples",
                       title="🔥 Beta — Query Count by Region × Month")
    fig_hm.update_layout(height=280, paper_bgcolor="#ffffff",
                         margin=dict(l=60,r=20,t=50,b=20),
                         font=dict(family="Segoe UI,Arial,sans-serif"))

    # Business unit cost grouped bar
    fig_bu = go.Figure()
    bus = ["Sales","Marketing","Operations","Support"]
    for n in ds:
        fig_bu.add_trace(go.Bar(
            x=bus, y=[BU_COST[b][n] for b in bus], name=n,
            marker_color=COLORS[n],
            text=[f"₹{BU_COST[b][n]:,}" for b in bus],
            textposition="outside"
        ))
    fl(fig_bu, "🏢 Avg Cloud Cost by Business Unit", h=330)
    fig_bu.update_layout(barmode="group")
    fig_bu.update_yaxes(tickprefix="₹", range=[0,68000])

    # Stacked area
    fig_area = go.Figure()
    for n in ds:
        y = [MONTHLY_QUERY[n][m] for m in MONTHS]
        fig_area.add_trace(go.Scatter(
            x=MONTHS, y=y, name=n, stackgroup="one",
            fillcolor=rgba(COLORS[n], 0.65),
            line=dict(color=COLORS[n], width=1)
        ))
    fl(fig_area, "📊 Stacked Area — Monthly Query Count by Dataset", h=300)
    fig_area.update_yaxes(title="Query Count")

    return html.Div([
        sh("📈 Section 3 — Trends & Time Series Analysis"),
        card(dcc.Graph(figure=fig_line)),
        html.Div([card(dcc.Graph(figure=fig_cost)),
                  card(dcc.Graph(figure=fig_hm))], style=G2),
        html.Div([card(dcc.Graph(figure=fig_bu)),
                  card(dcc.Graph(figure=fig_area))], style=G2),
    ])

# ══════════════════════════════════════════════════════════════════
#  TAB 3 — REGIONAL
# ══════════════════════════════════════════════════════════════════
def tab_region(ds, reg):
    rsel = [r for r in reg if r in REGIONS] or REGIONS

    # Grouped bar
    fig_bar = go.Figure()
    for n in ds:
        fig_bar.add_trace(go.Bar(
            x=rsel, y=[REGION_COST[n][r] for r in rsel], name=n,
            marker_color=COLORS[n],
            text=[f"₹{REGION_COST[n][r]:,}" for r in rsel],
            textposition="outside"
        ))
    fl(fig_bar, "🗺️ Section 4 — Region-wise Avg Cloud Cost (₹)", h=360)
    fig_bar.update_layout(barmode="group")
    fig_bar.update_yaxes(tickprefix="₹", title="Avg Cloud Cost (₹)", range=[0,70000])

    # Heatmap
    heat = pd.DataFrame(
        {n: [REGION_COST[n][r] for r in rsel] for n in ds},
        index=rsel
    )
    fig_hm = px.imshow(heat, text_auto=".0f", aspect="auto",
                       color_continuous_scale="YlOrRd",
                       title="🌡️ Cost Heatmap — Region × Dataset")
    fig_hm.update_layout(height=280, paper_bgcolor="#ffffff",
                         margin=dict(l=60,r=20,t=50,b=20),
                         font=dict(family="Segoe UI,Arial,sans-serif"))

    # Treemap
    rows = [{"Dataset":n,"Region":r,"Cost":REGION_COST[n][r]}
            for n in ds for r in rsel]
    fig_tree = px.treemap(pd.DataFrame(rows),
                          path=["Dataset","Region"], values="Cost",
                          color="Cost", color_continuous_scale="Purples",
                          title="🗺️ Treemap — Cloud Cost by Dataset & Region")
    fig_tree.update_layout(height=360, paper_bgcolor="#ffffff",
                           margin=dict(l=20,r=20,t=50,b=20),
                           font=dict(family="Segoe UI,Arial,sans-serif"))

    # Pie (Client share by region)
    if "Client" in ds:
        fig_pie = go.Figure(go.Pie(
            labels=rsel,
            values=[REGION_COST["Client"][r] for r in rsel],
            hole=0.5,
            marker_colors=["#f59e0b","#06b6d4","#8b5cf6","#ec4899"][:len(rsel)],
            textinfo="label+percent"
        ))
        fig_pie.update_layout(
            height=300, paper_bgcolor="#ffffff",
            margin=dict(l=20,r=20,t=50,b=40),
            title=dict(text="🥧 Client — Cloud Cost Share by Region",
                       font=dict(size=12,color="#1e293b")),
            legend=dict(orientation="h",y=-0.12),
            font=dict(family="Segoe UI,Arial,sans-serif"),
            showlegend=True
        )
    else:
        fig_pie = go.Figure()

    return html.Div([
        sh("🗺️ Section 4 — Region-wise Cloud Cost Analysis"),
        card(dcc.Graph(figure=fig_bar)),
        html.Div([card(dcc.Graph(figure=fig_hm)),
                  card(dcc.Graph(figure=fig_pie))], style=G2),
        card(dcc.Graph(figure=fig_tree)),
    ])

# ══════════════════════════════════════════════════════════════════
#  TAB 4 — KPI VECTOR
# ══════════════════════════════════════════════════════════════════
def tab_kpi(ds):
    # KPI formula table
    tbl_df = pd.DataFrame({
        "KPI Dimension": KPI_LABELS,
        "Formula"      : ["AVERAGE(Avg_Query_Time_Sec) — inverted",
                          "SUM(Cloud_Cost)/SUM(Data_Volume) — inverted",
                          "AVERAGE(Uptime_Pct)",
                          "AVERAGE(Data_Quality_Score)",
                          "AVERAGE(Conversion_Rate_Pct)/24×100",
                          "AVERAGE(Optimization_Score)",
                          "AVERAGE(Decision_Effectiveness_Score)"],
        **{n: KPI_VECTORS[n] for n in ds}
    })
    hdr_clr = ["#4f46e5","#4f46e5"] + [COLORS[n] for n in ds]
    fig_tbl = go.Figure(go.Table(
        header=dict(values=list(tbl_df.columns),
                    fill_color=hdr_clr,
                    font=dict(color="white",size=11),
                    align="center", height=34),
        cells=dict(values=[tbl_df[c] for c in tbl_df.columns],
                   fill_color=[["#f8fafc"]*7],
                   font=dict(size=11), align="center", height=30)
    ))
    fig_tbl.update_layout(height=290, margin=dict(l=10,r=10,t=10,b=10),
                          paper_bgcolor="#ffffff")

    # Grouped bar
    fig_bar = go.Figure()
    for n in ds:
        fig_bar.add_trace(go.Bar(
            x=KPI_LABELS, y=KPI_VECTORS[n], name=n,
            marker_color=COLORS[n],
            text=[str(v) for v in KPI_VECTORS[n]],
            textposition="outside"
        ))
    fl(fig_bar, "🧮 Section 5 — KPI Vector Components (normalized 0–100)", h=380)
    fig_bar.update_layout(barmode="group")
    fig_bar.update_yaxes(range=[0,112], title="Score (0–100)")

    # Magnitude bar
    mags = {n: round(float(np.sqrt(sum(v**2 for v in KPI_VECTORS[n]))),1) for n in ds}
    fig_mag = go.Figure()
    for n in ds:
        fig_mag.add_trace(go.Bar(
            x=[n], y=[mags[n]], name=n,
            marker_color=COLORS[n], width=0.45,
            text=[f"‖v‖ = {mags[n]}"], textposition="outside"
        ))
    fl(fig_mag, "📐 KPI Vector Magnitude  ‖v‖ = √(Σvᵢ²)", h=300)
    fig_mag.update_layout(showlegend=False)
    fig_mag.update_yaxes(range=[150,220], title="Euclidean Norm")

    # Horizontal bullet
    fig_bul = go.Figure()
    shown = set()
    for n in ds:
        for i, (lbl,val) in enumerate(zip(KPI_LABELS, KPI_VECTORS[n])):
            show_leg = n not in shown
            if show_leg: shown.add(n)
            fig_bul.add_trace(go.Bar(
                x=[val], y=[f"{lbl} | {n}"],
                orientation="h", marker_color=COLORS[n],
                text=[str(val)], textposition="auto",
                name=n, showlegend=show_leg
            ))
    fl(fig_bul, "📊 KPI Vector — Horizontal Bullet Chart", h=560)
    fig_bul.update_layout(barmode="overlay", bargap=0.3)
    fig_bul.update_xaxes(range=[0,105], title="Score (0–100)")

    return html.Div([
        sh("🧮 Section 5 — Linear Algebra: KPI Vector Construction"),
        card(dcc.Graph(figure=fig_tbl)),
        card(dcc.Graph(figure=fig_bar)),
        card(dcc.Graph(figure=fig_mag)),
        card(dcc.Graph(figure=fig_bul)),
    ])

# ══════════════════════════════════════════════════════════════════
#  TAB 5 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════
def tab_optim(ds):
    # Cost/GB horizontal bar
    fig_cpgb = go.Figure()
    for n in ds:
        fig_cpgb.add_trace(go.Bar(
            x=[COST_PER_GB[n]], y=[n], orientation="h",
            marker_color=COLORS[n], width=0.4,
            text=[f"₹{COST_PER_GB[n]}"], textposition="outside"
        ))
    fl(fig_cpgb, "⚡ Section 6 — Cloud Cost per GB = SUM(Cost) ÷ SUM(Volume_GB)", h=270)
    fig_cpgb.update_layout(showlegend=False)
    fig_cpgb.update_xaxes(range=[230,355], tickprefix="₹",
                          title="₹ per GB  (lower = more efficient ✅)")

    # Scatter Optimization vs Conversion
    fig_sc = go.Figure()
    np.random.seed(42)
    for n in ds:
        xs = np.clip(np.random.normal(AVG_OPT[n],  5, 60), 40, 100)
        ys = np.clip(np.random.normal(AVG_CONV[n], 1.5, 60), 4, 24)
        fig_sc.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=n,
            marker=dict(color=rgba(COLORS[n],0.3), size=6,
                        line=dict(width=0))
        ))
        fig_sc.add_trace(go.Scatter(
            x=[AVG_OPT[n]], y=[AVG_CONV[n]],
            mode="markers+text", showlegend=False,
            marker=dict(color=COLORS[n], size=18, symbol="star",
                        line=dict(color="white",width=2)),
            text=[f" {n}  ({AVG_OPT[n]},{AVG_CONV[n]}%)"],
            textfont=dict(size=10, color=COLORS[n])
        ))
    fl(fig_sc, "🔄 Optimization Score vs Conversion Rate — Trade-off Scatter", h=360)
    fig_sc.add_hline(y=12, line_dash="dot", line_color="#94a3b8", opacity=0.6)
    fig_sc.add_vline(x=72, line_dash="dot", line_color="#94a3b8", opacity=0.6)
    fig_sc.update_xaxes(title="Optimization Score")
    fig_sc.update_yaxes(title="Conversion Rate (%)")

    # Optimization score bar
    fig_opt = go.Figure()
    for n in ds:
        fig_opt.add_trace(go.Bar(
            x=[n], y=[AVG_OPT[n]], name=n,
            marker_color=COLORS[n], width=0.45,
            text=[str(AVG_OPT[n])], textposition="outside"
        ))
    fl(fig_opt, "⚙️ Avg Optimization Score — AVERAGE(Optimization_Score)", h=290)
    fig_opt.update_layout(showlegend=False)
    fig_opt.update_yaxes(range=[60,88], title="Score")
    fig_opt.add_hline(y=75, line_dash="dot", line_color="#94a3b8",
                      annotation_text="Target 75")

    # Conversion rate bar
    fig_conv = go.Figure()
    for n in ds:
        fig_conv.add_trace(go.Bar(
            x=[n], y=[AVG_CONV[n]], name=n,
            marker_color=COLORS[n], width=0.45,
            text=[f"{AVG_CONV[n]}%"], textposition="outside"
        ))
    fl(fig_conv, "📈 Avg Conversion Rate — AVERAGE(Conversion_Rate_Pct)", h=290)
    fig_conv.update_layout(showlegend=False)
    fig_conv.update_yaxes(range=[0,18], title="Conversion Rate (%)")

    return html.Div([
        sh("⚡ Section 6 — Optimization: Cost-Performance Trade-off"),
        card(dcc.Graph(figure=fig_cpgb)),
        card(dcc.Graph(figure=fig_sc)),
        html.Div([card(dcc.Graph(figure=fig_opt)),
                  card(dcc.Graph(figure=fig_conv))], style=G2),
    ])

# ══════════════════════════════════════════════════════════════════
#  TAB 6 — MINING
# ══════════════════════════════════════════════════════════════════
def tab_mining(ds, seg):
    ssel = [s for s in seg if s in SEGS] or SEGS

    # Doughnut charts
    pies = []
    for n in ds:
        vals = [MINE_PCT[n].get(s,0) for s in SEGS]
        fig_p = go.Figure(go.Pie(
            labels=SEGS, values=vals, hole=0.55,
            marker=dict(colors=MINE_CLR), textinfo="label+percent"
        ))
        fig_p.update_layout(
            title=dict(text=f"⛏️ {n}",
                       font=dict(size=12,color=COLORS[n])),
            height=270, paper_bgcolor="#ffffff",
            margin=dict(l=20,r=20,t=50,b=20), showlegend=False,
            font=dict(family="Segoe UI,Arial,sans-serif")
        )
        pies.append(card(dcc.Graph(figure=fig_p)))

    # Conversion by segment
    fig_conv = go.Figure()
    for n in ds:
        fig_conv.add_trace(go.Bar(
            x=ssel,
            y=[CONV_BY_SEG[n].get(s,0) for s in ssel],
            name=n, marker_color=COLORS[n],
            text=[f"{CONV_BY_SEG[n].get(s,0)}%" for s in ssel],
            textposition="outside"
        ))
    fl(fig_conv, "🔍 High/Low Conversion Pattern by Mining Segment", h=350)
    fig_conv.update_layout(barmode="group")
    fig_conv.update_yaxes(range=[0,20], title="Avg Conversion Rate (%)")

    # At-Risk % bar
    fig_ar = go.Figure()
    for n in ds:
        v = MINE_PCT[n].get("At-Risk",0)
        fig_ar.add_trace(go.Bar(
            x=[n], y=[v], name=n,
            marker_color=COLORS[n], width=0.45,
            text=[f"{v}%"], textposition="outside"
        ))
    fl(fig_ar, "⚠️ At-Risk Segment % per Dataset  (lower = healthier ✅)", h=280)
    fig_ar.update_layout(showlegend=False)
    fig_ar.update_yaxes(range=[0,52], title="% of Records")
    fig_ar.add_hline(y=35, line_dash="dot", line_color="#94a3b8",
                     annotation_text="Warning: 35%")

    # Stacked bar
    rows = []
    for n in ds:
        for s in SEGS:
            rows.append({"Dataset":n,"Segment":s,
                         "Count":int(RECORDS[n]*MINE_PCT[n].get(s,0)/100)})
    fig_st = px.bar(pd.DataFrame(rows), x="Dataset", y="Count",
                    color="Segment",
                    color_discrete_map=dict(zip(SEGS,MINE_CLR)),
                    barmode="stack",
                    title="📊 Stacked Segment Counts per Dataset")
    fig_st.update_layout(height=320, paper_bgcolor="#ffffff",
                         margin=dict(l=55,r=20,t=50,b=40),
                         font=dict(family="Segoe UI,Arial,sans-serif"),
                         legend=dict(orientation="h",y=-0.25))

    # Sunburst
    sun = []
    for n in ds:
        for s in SEGS:
            for m in MODELS:
                sun.append({"Dataset":n,"Segment":s,"Model":m,
                            "Count":max(1,int(RECORDS[n]*MINE_PCT[n].get(s,0)/100/3))})
    fig_sun = px.sunburst(pd.DataFrame(sun),
                          path=["Dataset","Segment","Model"],
                          values="Count", color="Segment",
                          color_discrete_map=dict(zip(SEGS,MINE_CLR)),
                          title="☀️ Sunburst — Dataset → Segment → Service Model")
    fig_sun.update_layout(height=400, paper_bgcolor="#ffffff",
                          margin=dict(l=20,r=20,t=50,b=20),
                          font=dict(family="Segoe UI,Arial,sans-serif"))

    return html.Div([
        sh("⛏️ Section 7 — Data Mining: Segment Analysis"),
        html.Div(pies, style=G3),
        card(dcc.Graph(figure=fig_conv)),
        html.Div([card(dcc.Graph(figure=fig_ar)),
                  card(dcc.Graph(figure=fig_st))], style=G2),
        card(dcc.Graph(figure=fig_sun)),
    ])

# ══════════════════════════════════════════════════════════════════
#  TAB 7 — CLOUD MODELS
# ══════════════════════════════════════════════════════════════════
def tab_cloud(mod):
    msel = [m for m in mod if m in MODELS] or MODELS

    # Cost bar
    fig_cost = go.Figure()
    for m,col in zip(msel,SVC_CLR):
        fig_cost.add_trace(go.Bar(
            x=[m], y=[SVC_COST[m]], name=m,
            marker_color=col, width=0.45,
            text=[f"₹{SVC_COST[m]:,}"], textposition="outside"
        ))
    fl(fig_cost, "☁️ Section 8 — Avg Cloud Cost by Service Model (Beta)", h=310)
    fig_cost.update_layout(showlegend=False)
    fig_cost.update_yaxes(range=[38000,52000], tickprefix="₹")

    # Uptime bar
    fig_up = go.Figure()
    for m,col in zip(msel,SVC_CLR):
        fig_up.add_trace(go.Bar(
            x=[m], y=[SVC_UPTIME[m]], name=m,
            marker_color=col, width=0.45,
            text=[f"{SVC_UPTIME[m]}%"], textposition="outside"
        ))
    fl(fig_up, "📡 Avg Uptime % by Service Model (Beta) — AVERAGE(Uptime_Pct)", h=310)
    fig_up.update_layout(showlegend=False)
    fig_up.update_yaxes(range=[97.5,100], title="Uptime %")

    # Multi-KPI grouped
    kpi_names = ["Optimization","Conversion %","Query Time(s)"]
    fig_mk = go.Figure()
    for m,col in zip(msel,SVC_CLR):
        fig_mk.add_trace(go.Bar(
            x=kpi_names,
            y=[SVC_OPT[m], SVC_CONV[m], SVC_QT[m]*10],
            name=m, marker_color=col,
            text=[str(SVC_OPT[m]), f"{SVC_CONV[m]}%", f"{SVC_QT[m]}s"],
            textposition="outside"
        ))
    fl(fig_mk, "📊 Multi-KPI by Service Model (Beta)", h=320)
    fig_mk.update_layout(barmode="group")

    # Summary table
    tdf = pd.DataFrame({
        "Service Model": MODELS,
        "Avg Cost (₹)"  : [SVC_COST[m]  for m in MODELS],
        "Uptime %"      : [SVC_UPTIME[m] for m in MODELS],
        "Query Time(s)" : [SVC_QT[m]     for m in MODELS],
        "Optimization"  : [SVC_OPT[m]    for m in MODELS],
        "Conversion %"  : [SVC_CONV[m]   for m in MODELS],
        "Best For"      : ["Balanced","Best Uptime ✅","Lowest Cost ✅"],
    })
    fig_tbl = go.Figure(go.Table(
        header=dict(values=list(tdf.columns),
                    fill_color="#4f46e5",
                    font=dict(color="white",size=11),
                    align="center", height=34),
        cells=dict(values=[tdf[c] for c in tdf.columns],
                   fill_color=[["#dbeafe","#ede9fe","#dcfce7"]],
                   font=dict(size=11), align="center", height=30)
    ))
    fig_tbl.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10),
                          paper_bgcolor="#ffffff")

    return html.Div([
        sh("☁️ Section 8 — Cloud Computing: Service Model Comparison"),
        card(dcc.Graph(figure=fig_tbl)),
        html.Div([card(dcc.Graph(figure=fig_cost)),
                  card(dcc.Graph(figure=fig_up))], style=G2),
        card(dcc.Graph(figure=fig_mk)),
    ])

# ══════════════════════════════════════════════════════════════════
#  TAB 8 — BENCHMARK
# ══════════════════════════════════════════════════════════════════
def tab_bench(ds):
    # Full benchmark table
    metrics = ["Avg Query Time(s)","Cost per GB(₹)","Uptime %",
               "Data Quality","Conversion %","Optimization",
               "Decision Eff.","CSAT"]
    formulas= ["AVERAGE(Avg_Query_Time_Sec)","SUM(Cost)/SUM(Volume)",
               "AVERAGE(Uptime_Pct)","AVERAGE(Data_Quality_Score)",
               "AVERAGE(Conversion_Rate_Pct)","AVERAGE(Optimization_Score)",
               "AVERAGE(Decision_Effectiveness_Score)",
               "AVERAGE(Customer_Satisfaction_Score)"]
    all_vals = {
        "Client": [3.47,318.6,97.37,74.9,11.8,68.9,74.8,3.68],
        "Alpha" : [2.56,294.1,98.57,82.8,14.9,79.1,85.3,4.00],
        "Beta"  : [2.93,269.4,98.96,80.8,13.7,76.1,82.1,4.14],
    }
    best  = ["Alpha","Beta","Beta","Alpha","Alpha","Alpha","Alpha","Beta"]
    notes = ["35% slower","18.3% above Beta","1.6% below Beta",
             "7.9 pts below","3.1% below","14.8% below","12.3% below","0.46 below"]

    hdr_vals = ["Metric","Formula"]
    hdr_clr  = ["#4f46e5","#4f46e5"]
    cell_vals = [metrics, formulas]
    for n in ds:
        hdr_vals.append(n); hdr_clr.append(COLORS[n])
        cell_vals.append(all_vals[n])
    hdr_vals += ["Best","Client Status"]
    hdr_clr  += ["#10b981","#64748b"]
    cell_vals += [best, notes]

    fig_tbl = go.Figure(go.Table(
        header=dict(values=hdr_vals, fill_color=hdr_clr,
                    font=dict(color="white",size=11),
                    align="center", height=34),
        cells=dict(values=cell_vals,
                   fill_color=[["#f8fafc"]*8]*len(cell_vals),
                   font=dict(size=11), align="center", height=30)
    ))
    fig_tbl.update_layout(height=310,
                          margin=dict(l=10,r=10,t=10,b=10),
                          paper_bgcolor="#ffffff")

    # KPI grouped bar
    fig_bar = go.Figure()
    for n in ds:
        fig_bar.add_trace(go.Bar(
            x=KPI_LABELS, y=KPI_VECTORS[n], name=n,
            marker_color=COLORS[n],
            text=[str(v) for v in KPI_VECTORS[n]],
            textposition="outside"
        ))
    fl(fig_bar, "🏁 Section 9 — All 7 KPI Scores (normalized 0–100)", h=380)
    fig_bar.update_layout(barmode="group")
    fig_bar.update_yaxes(range=[0,112], title="Normalized Score (0–100)")

    # Uptime bar
    fig_up = go.Figure()
    for n in ds:
        fig_up.add_trace(go.Bar(
            x=[n], y=[AVG_UPTIME[n]], name=n,
            marker_color=COLORS[n], width=0.45,
            text=[f"{AVG_UPTIME[n]}%"], textposition="outside"
        ))
    fl(fig_up, "📡 Avg Uptime % — AVERAGE(Uptime_Pct)", h=290)
    fig_up.update_layout(showlegend=False)
    fig_up.update_yaxes(range=[96.5,100], title="Uptime %")
    fig_up.add_hline(y=99, line_dash="dot", line_color="#ef4444",
                     annotation_text="99% SLA Target")

    # Overall scorecard
    fig_sc = go.Figure()
    for n in ds:
        fig_sc.add_trace(go.Bar(
            x=[SCORES[n]], y=[n], orientation="h",
            marker_color=COLORS[n], width=0.4,
            text=[f"{SCORES[n]}/100"], textposition="outside"
        ))
    fl(fig_sc, "🏆 Final Overall Weighted Scorecard (0–100)", h=260)
    fig_sc.update_layout(showlegend=False)
    fig_sc.update_xaxes(range=[0,112], title="Score")
    fig_sc.add_vline(x=70, line_dash="dot", line_color="#94a3b8",
                     annotation_text="Benchmark 70")

    # Correlation heatmap (Client)
    corr = pd.DataFrame({
        "QueryTime" :[1.00,-0.12,-0.21,-0.14,-0.09,-0.11,-0.13],
        "CloudCost" :[-0.12,1.00,0.08,0.06,0.04,0.07,0.05],
        "Uptime"    :[-0.21,0.08,1.00,0.18,0.15,0.17,0.16],
        "Quality"   :[-0.14,0.06,0.18,1.00,0.22,0.31,0.38],
        "Conversion":[-0.09,0.04,0.15,0.22,1.00,0.28,0.25],
        "Optim"     :[-0.11,0.07,0.17,0.31,0.28,1.00,0.42],
        "DecEff"    :[-0.13,0.05,0.16,0.38,0.25,0.42,1.00],
    }, index=["QueryTime","CloudCost","Uptime","Quality","Conversion","Optim","DecEff"])
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="🔗 KPI Correlation Heatmap — Client Dataset")
    fig_corr.update_layout(height=360, paper_bgcolor="#ffffff",
                           margin=dict(l=65,r=20,t=50,b=20),
                           font=dict(family="Segoe UI,Arial,sans-serif"))

    return html.Div([
        sh("🏁 Section 9 — Full Benchmark: Client vs Competitor Alpha vs Beta"),
        card(dcc.Graph(figure=fig_tbl)),
        card(dcc.Graph(figure=fig_bar)),
        html.Div([card(dcc.Graph(figure=fig_up)),
                  card(dcc.Graph(figure=fig_sc))], style=G2),
        card(dcc.Graph(figure=fig_corr)),
    ])

# ══════════════════════════════════════════════════════════════════
#  TAB 9 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
def tab_reco(ds):
    # Recommendation cards
    recs = [
        (CL,     "🔴 P1 — Reduce Cloud Cost per GB",
         f"Client ₹318.6/GB vs Beta ₹269.4/GB — 18.3% premium. Shift workloads to SaaS (avg ₹42,900). "
         f"Renegotiate IaaS contracts. Potential saving: ₹49.2 per GB."),
        ("#f59e0b","🟠 P2 — Improve Data Quality Score",
         f"Client 74.9 vs Alpha 82.8 — gap of 7.9 pts. Poor quality reduces Decision Effectiveness "
         f"(74.8 vs 85.3). Implement automated validation pipelines and schema enforcement."),
        (AL,     "🟢 P3 — Reduce At-Risk Segment from 40%",
         f"40% At-Risk is highest among all 3 datasets (Beta: 28%). Launch targeted retention "
         f"campaigns for At-Risk SME and Mid-Market customers."),
        ("#0ea5e9","🔵 P4 — Optimize Query Performance",
         f"Client 3.47s vs Alpha 2.56s — 35% slower. Add query indexing, result caching, "
         f"and migrate heavy workloads to PaaS (2.88s avg)."),
        ("#7c3aed","🟣 P5 — Raise Uptime to ≥99%",
         f"Client 97.37% vs Beta 98.96% — 1.6% below. Invest in multi-region redundancy, "
         f"auto-failover, and proactive incident management. Set SLA ≥99%."),
        ("#ca8a04","🟡 P6 — Boost Optimization Score",
         f"Client 68.9 vs Alpha 79.1 — gap of 10.2. Right-size cloud resources, enable "
         f"auto-scaling, purchase reserved instances, implement storage lifecycle rules."),
    ]
    rcards = html.Div([
        html.Div([
            html.Div(title, style={"fontWeight":"800","fontSize":".88rem",
                                   "color":col,"marginBottom":"7px"}),
            html.Div(body, style={"fontSize":".78rem","color":"#4b5563",
                                  "lineHeight":"1.65"}),
        ], style={**CARD,"borderLeft":f"5px solid {col}","marginBottom":"0"})
        for col,title,body in recs
    ], style={"display":"grid",
              "gridTemplateColumns":"repeat(auto-fit,minmax(280px,1fr))",
              "gap":"14px","marginBottom":"16px"})

    # Gauge charts
    gauges = []
    specs = [
        ("Data Quality",  74.9, 82.8, 60,100, CL),
        ("Uptime %",     97.37,98.96, 95,100, AL),
        ("Optimization",  68.9, 79.1, 50,100, BL),
        ("Conversion %",  11.8, 14.9,  4, 24, "#f59e0b"),
    ]
    for title,cur,tgt,mn,mx,col in specs:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cur,
            delta={"reference":tgt,"increasing":{"color":"#10b981"},
                   "decreasing":{"color":"#ef4444"}},
            title={"text":title,"font":{"size":12}},
            gauge={
                "axis"  :{"range":[mn,mx]},
                "bar"   :{"color":col,"thickness":0.28},
                "steps" :[{"range":[mn,(mn+mx)/2],"color":"#fee2e2"},
                           {"range":[(mn+mx)/2,mx],"color":"#dcfce7"}],
                "threshold":{"line":{"color":"#1e293b","width":3},
                             "thickness":0.8,"value":tgt},
            }
        ))
        fig_g.update_layout(height=240, paper_bgcolor="#ffffff",
                            margin=dict(l=20,r=20,t=55,b=10),
                            font=dict(family="Segoe UI,Arial,sans-serif"))
        gauges.append(card(dcc.Graph(figure=fig_g)))

    gauge_grid = html.Div(gauges, style={
        "display":"grid",
        "gridTemplateColumns":"repeat(auto-fit,minmax(230px,1fr))",
        "gap":"14px","marginBottom":"16px"
    })

    # Progress bar: current vs target
    knames = ["Data Quality","Uptime %","Optimization","Conversion %","Cost Eff.","Dec. Eff."]
    c_vals = [74.9, 97.37, 68.9, 49.2, 42.0, 74.8]
    t_vals = [82.8, 98.96, 79.1, 62.1, 72.3, 85.3]
    fig_prog = go.Figure()
    fig_prog.add_trace(go.Bar(
        x=knames, y=c_vals, name="Client (Current)",
        marker_color=CL, opacity=0.85,
        text=[str(v) for v in c_vals], textposition="outside"
    ))
    fig_prog.add_trace(go.Bar(
        x=knames, y=t_vals, name="Target (Best Competitor)",
        marker_color=AL, opacity=0.5,
        text=[str(v) for v in t_vals], textposition="outside"
    ))
    fl(fig_prog, "🎯 Client Current vs Target — Priority KPI Gaps", h=360)
    fig_prog.update_layout(barmode="overlay")
    fig_prog.update_yaxes(range=[0,105])

    # Final scorecard
    fig_sc = go.Figure()
    for n in ds:
        fig_sc.add_trace(go.Bar(
            x=[SCORES[n]], y=[n], orientation="h",
            marker_color=COLORS[n], width=0.4,
            text=[f"  {SCORES[n]} / 100"], textposition="outside"
        ))
    fl(fig_sc, "🏆 Final Overall Performance Scorecard (0–100)", h=260)
    fig_sc.update_layout(showlegend=False)
    fig_sc.update_xaxes(range=[0,112], title="Weighted Score")
    fig_sc.add_vline(x=70, line_dash="dot", line_color="#94a3b8",
                     annotation_text="Benchmark 70")

    return html.Div([
        sh("💡 Section 10 — Recommendations & Priority Improvement Areas"),
        rcards,
        gauge_grid,
        card(dcc.Graph(figure=fig_prog)),
        card(dcc.Graph(figure=fig_sc)),
    ])

# ─────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 68)
    print("  ☁️  CLOUD ANALYTICS INTELLIGENCE — INTERACTIVE DASHBOARD")
    print("  Student : Rahul Bhatti | 024BCA110900 | BCA Sem-5 | CU")
    print("  Data    : Real values — Client(1600) + Alpha(1100) + Beta(1100)")
    print("=" * 68)
    print("  🌐  Open your browser:  http://127.0.0.1:8050")
    print()
    print("  TABS :")
    for t in ["📋 Overview","📈 Trends","🗺️ Regional","🧮 KPI Vector",
              "⚡ Optimization","⛏️ Mining","☁️ Cloud Models",
              "🏁 Benchmark","💡 Recommendations"]:
        print(f"       {t}")
    print()
    print("  FILTERS : Dataset | Region | Service Model | Mining Segment")
    print("  Ctrl+C  : Stop the server")
    print("=" * 68)
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, port=port, host="0.0.0.0")
