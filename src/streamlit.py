from __future__ import annotations

import pathlib

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from pyvis.network import Network
from streamlit_option_menu import option_menu

# ======================== warna
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]
MERAH = "#d03b3b"
BIRU_MUDA = "#9dc2ef"

SKALA = [[0.0, "#f2f7fd"], [0.5, "#5b9be8"], [1.0, "#123f78"]]
HURUF_SIMPUL = 34   

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

WARNA_STATUS = dict(zip(
    ["Complete", "Shipped", "Processing", "Cancelled", "Returned"], SERIES))
WARNA_SUMBER = dict(zip(
    ["Email", "Adwords", "YouTube", "Facebook", "Organic"], SERIES))

X_SEPI = dict(showticklabels=False, showline=False, ticks="", showgrid=False)
TOP_N = 10

GRAIN = {"Harian": ("D", "hari"), "Mingguan": ("W-MON", "minggu"), "Bulanan": ("MS", "bulan")}

rp = lambda x: f"${x:,.0f}"


def potong(s, n=38):
    """Potong nama panjang di batas kata, beri elipsis supaya tidak terlihat terpotong."""
    s = str(s)
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def pasang_template() -> None:
    pio.templates["thelook"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, size=12, color=INK2),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=SERIES,
        margin=dict(l=16, r=16, t=56, b=16),
        xaxis=dict(showgrid=False, linecolor=AXIS, ticks="outside", automargin=True,
                   tickcolor=AXIS, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False, automargin=True,
                   linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=INK2), title_text=""),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    ))
    pio.templates.default = "thelook"


# ==================================== Load
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOKAL = ROOT / "data" / "dashboard"
HDFS = "hdfs://namenode:9000/output/dashboard"
SPARK_REMOTE = "sc://localhost:3002"
TABEL = ["fact_sales", "fact_orders", "fact_events", "dim_products"]


def tarik_dari_hdfs() -> dict[str, pd.DataFrame]:
    """Ambil tabel ringkasan dari HDFS sekali, lalu simpan salinan lokalnya."""
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder
             .appName("thelook-dashboard")
             .remote(SPARK_REMOTE)
             .getOrCreate())
    LOKAL.mkdir(parents=True, exist_ok=True)
    hasil = {}
    try:
        for nama in TABEL:
            df = spark.read.parquet(f"{HDFS}/{nama}").toPandas()
            df.attrs = {}
            df.to_parquet(LOKAL / f"{nama}.parquet", index=False)
            hasil[nama] = df
    finally:
        spark.stop()
    return hasil


@st.cache_data(show_spinner="Memuat tabel ringkasan…")
def load() -> dict[str, pd.DataFrame]:
    if all((LOKAL / f"{n}.parquet").exists() for n in TABEL):
        lokal = {n: pd.read_parquet(LOKAL / f"{n}.parquet") for n in TABEL}
        if all("tanggal" in lokal[n] for n in ("fact_sales", "fact_orders", "fact_events")):
            return lokal
        st.info("Salinan lokal masih format lama. Mengambil ulang dari HDFS…")
    return tarik_dari_hdfs()


HDFS_GRAPH = "hdfs://namenode:9000/output/graph/category_pairs"
GRAPH_LOKAL = LOKAL / "category_pairs.parquet"
GRAPH_HTML = LOKAL / "jaringan_kategori.html"


def tarik_graph_dari_hdfs() -> pd.DataFrame:
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder
             .appName("thelook-dashboard-graph")
             .remote(SPARK_REMOTE)
             .getOrCreate())
    try:
        df = spark.read.parquet(HDFS_GRAPH).toPandas()
        df.attrs = {}                     
        LOKAL.mkdir(parents=True, exist_ok=True)
        df.to_parquet(GRAPH_LOKAL, index=False)
        return df
    finally:
        spark.stop()


@st.cache_data(show_spinner="Memuat pasangan kategori…")
def load_graph() -> pd.DataFrame:
    if GRAPH_LOKAL.exists():
        return pd.read_parquet(GRAPH_LOKAL)
    return tarik_graph_dari_hdfs()


# -------------------------------------------------------------- hasil pelatihan
HDFS_ML = "hdfs://namenode:9000/output/ml"
ML_TABEL = ["skor", "prediksi", "dampak", "meta"]


def tarik_ml_dari_hdfs() -> dict[str, pd.DataFrame]:
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder
             .appName("thelook-dashboard-ml")
             .remote(SPARK_REMOTE)
             .getOrCreate())
    LOKAL.mkdir(parents=True, exist_ok=True)
    hasil = {}
    try:
        for nama in ML_TABEL:
            df = spark.read.parquet(f"{HDFS_ML}/{nama}").toPandas()
            df.attrs = {}
            df.to_parquet(LOKAL / f"ml_{nama}.parquet", index=False)
            hasil[nama] = df
    finally:
        spark.stop()
    return hasil


@st.cache_data(show_spinner="Memuat hasil pelatihan model…")
def load_ml() -> dict[str, pd.DataFrame]:
    if all((LOKAL / f"ml_{n}.parquet").exists() for n in ML_TABEL):
        return {n: pd.read_parquet(LOKAL / f"ml_{n}.parquet") for n in ML_TABEL}
    return tarik_ml_dari_hdfs()


def filter(df, mulai, selesai, kategori, status,
         pakai_kategori=True, pakai_status=True):
    """Terapkan filter sidebar ke salah satu tabel ringkasan."""
    m = df["tanggal"].between(mulai, selesai)
    if pakai_kategori and kategori and "category" in df:
        m &= df["category"].isin(kategori)
    if pakai_status and status and "status" in df:
        m &= df["status"].isin(status)
    return df[m]


# ==================================================================== grafik
def kartu_ringkasan(sales, orders) -> str:
    kartu = [
        ("Gross Sales", rp(sales["gross_sales"].sum())),
        ("Profit", rp(sales["profit"].sum())),
        ("Order", f"{orders['orders'].sum():,}"),
        ("Cancel Rate",
         f"{orders.loc[orders['status'] == 'Cancelled', 'orders'].sum() / max(orders['orders'].sum(), 1) * 100:.1f}%"),
        ("Produk Terjual", f"{sales['product_id'].nunique():,}"),
    ]
    sel = "".join(
        f'<div style="flex:1;min-width:150px;background:{SURFACE};'
        f'border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:14px 16px">'
        f'<div style="font:500 16px {FONT};color:{MUTED};letter-spacing:.02em">{k}</div>'
        f'<div style="font:600 26px {FONT};color:{INK};margin-top:6px">{v}</div></div>'
        for k, v in kartu)
    return (f'<div style="display:flex;gap:10px;flex-wrap:wrap;'
            f'font-family:{FONT};margin-bottom:18px">{sel}</div>')


def grafik_tren(sales, grain="Harian") -> go.Figure:
    aturan, satuan = GRAIN[grain]
    tren = (sales.set_index("tanggal")[["gross_sales", "profit"]]
                 .resample(aturan, label="left", closed="left").sum()
                 .reset_index().sort_values("tanggal"))
    fig = go.Figure()
    seri = [("Gross sales", "gross_sales", SERIES[0]), ("Profit", "profit", SERIES[1])]
    for nama, kol, warna in seri:
        fig.add_trace(go.Scatter(
            x=tren["tanggal"], y=tren[kol], name=nama, mode="lines",
            line=dict(color=warna, width=2),
            hovertemplate="%{x|%d %b %Y}<br>" + nama + " %{y:$,.0f}<extra></extra>"))

    if len(tren):
        x_akhir = tren["tanggal"].iloc[-1].to_pydatetime()
        for nama, kol, warna in seri:
            fig.add_annotation(x=x_akhir, y=float(tren[kol].iloc[-1]), text=f"  {nama}",
                               showarrow=False, xanchor="left",
                               font=dict(color=warna, size=11))

    fig.update_layout(
        title=f"Penjualan dan laba per {satuan}",
        height=420, hovermode="x unified", margin=dict(r=110),
        yaxis=dict(tickprefix="$", tickformat="~s"),
        xaxis=dict(rangeslider=dict(visible=True, thickness=.06)))
    return fig


def grafik_status(orders) -> go.Figure:
    st_ = (orders.groupby("status")["orders"].sum()
                 .reindex(list(WARNA_STATUS)).dropna().sort_values())
    if st_.empty:                      
        return go.Figure(layout=dict(title="Order menurut status", height=320))
    total = max(st_.sum(), 1)
    fig = go.Figure(go.Bar(
        x=st_.values, y=st_.index, orientation="h",
        marker=dict(color=[WARNA_STATUS[s] for s in st_.index],
                    line=dict(color=SURFACE, width=2)),
        text=[f"{v:,.0f}  ({v / total * 100:.1f}%)" for v in st_.values],
        textposition="outside", textfont=dict(color=INK2, size=11),
        hovertemplate="%{y}: %{x:,.0f} order<extra></extra>"))
    fig.update_layout(title="Order menurut status", height=320,
                      xaxis=dict(**X_SEPI, range=[0, st_.max() * 1.25]),
                      yaxis=dict(showgrid=False), showlegend=False)
    return fig

TAHAP = ["product", "cart", "purchase", "cancel"]
NAMA_TAHAP = {"product": "Lihat produk", "cart": "Masuk keranjang",
              "purchase": "Purchase", "cancel": "Cancel"}
WARNA_TAHAP = {"product": RAMP[0], "cart": RAMP[1], "purchase": RAMP[2],
               "cancel": MERAH}

BARIS_FUNNEL = ["Lihat produk", "Masuk keranjang", "Akhir kunjungan"]


def grafik_funnel(events) -> go.Figure:
    f_ = events.groupby("event_type")["sessions"].sum().reindex(TAHAP).fillna(0)
    puncak = max(f_.iloc[0], 1)
    pct = lambda v: f"{v / puncak * 100:.1f}%"

    seri = [
        ("Purchase",
         [f_["product"], f_["cart"], f_["purchase"]],
         [WARNA_TAHAP["product"], WARNA_TAHAP["cart"], WARNA_TAHAP["purchase"]],
         ["Lihat produk", "Masuk keranjang", "Purchase"]),
        ("Cancel",
         [0, 0, f_["cancel"]],
         ["rgba(0,0,0,0)", "rgba(0,0,0,0)", WARNA_TAHAP["cancel"]],
         ["", "", "Cancel"]),
    ]

    fig = go.Figure()
    for nama, nilai, warna, label in seri:
        teks = []
        for i, v in enumerate(nilai):
            if not v:
                teks.append("")
            elif i < 2:
                teks.append(f"{v:,.0f}<br>{pct(v)}")
            else:                       # pita terakhir ikut menyebut namanya
                teks.append(f"{label[i]}<br>{v:,.0f} ({pct(v)})")
        fig.add_trace(go.Funnel(
            name=nama, y=BARIS_FUNNEL, x=nilai,
            customdata=[[l, pct(v)] for l, v in zip(label, nilai)],
            marker=dict(color=warna, line=dict(color=SURFACE, width=2)),
            connector=dict(line=dict(color=AXIS, width=1, dash="dot")),
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color=SURFACE, size=12),
            text=teks, textinfo="text",
            hovertemplate="%{customdata[0]}<br>%{x:,.0f} sesi "
                          "(%{customdata[1]} dari sesi lihat produk)"
                          "<extra></extra>"))

    fig.update_layout(funnelmode="stack",
                      title="Funnel kunjungan (jumlah sesi)", height=380,
                      xaxis=dict(**X_SEPI), yaxis=dict(showgrid=False),
                      showlegend=False)
    return fig


def grafik_trafik(events) -> go.Figure:
    sesi = (events[events["event_type"] == "product"]
            .groupby("traffic_source")["sessions"].sum().sort_values())
    fig = go.Figure(go.Bar(
        x=sesi.values, y=sesi.index, orientation="h",
        marker=dict(color=[WARNA_SUMBER.get(s, SERIES[0]) for s in sesi.index],
                    line=dict(color=SURFACE, width=2)),
        text=[f"{v:,.0f} Sesi" for v in sesi.values], textposition="outside",
        textfont=dict(color=INK2, size=11),
        hovertemplate="%{y}: %{x:,.0f} sesi<extra></extra>"))
    fig.update_layout(title="Sesi Berdasarkan Asal Traffic", height=400,
                      showlegend=False, yaxis=dict(showgrid=False),
                      xaxis=dict(**X_SEPI, range=[0, sesi.max() * 1.2]))
    return fig


def batang_top(seri, judul, fmt, label_map=None) -> go.Figure:
    d = seri.nlargest(TOP_N).sort_values()
    penuh = [str(label_map.get(i, i)) if label_map is not None else str(i)
             for i in d.index]
    fig = go.Figure(go.Bar(
        x=d.values, y=[potong(p) for p in penuh], orientation="h",
        marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
        text=[fmt(v) for v in d.values], textposition="outside",
        textfont=dict(color=INK2, size=11),
        customdata=[[p] for p in penuh],
        hovertemplate="%{customdata[0]}<br>%{x:,.0f}<extra></extra>"))
    fig.update_layout(title=judul, height=420, showlegend=False,
                      xaxis=dict(**X_SEPI, range=[0, d.max() * 1.3]),
                      yaxis=dict(showgrid=False))
    return fig


def tabel_peringkat(unit, laba) -> pd.DataFrame:
    baris = []
    for label, seri in [("unit terjual", unit), ("profit", laba)]:
        urut = seri.sort_values(ascending=False)
        if len(urut) <= TOP_N:
            continue
        atas, ambang = urut.iloc[:TOP_N], urut.iloc[TOP_N - 1]
        baris.append({
            "peringkat": label,
            "#1": urut.iloc[0],
            f"#{TOP_N}": ambang,
            "kembar di 10 besar": int((atas == ambang).sum()),
            f"#{TOP_N + 1}": urut.iloc[TOP_N],
        })
    return pd.DataFrame(baris).set_index("peringkat")


# ============================================================ graph analytics
def volume_kategori(pairs) -> pd.Series:
    return (pd.concat([pairs.groupby("kat1")["n"].sum(),
                       pairs.groupby("kat2")["n"].sum()], axis=1)
              .fillna(0).sum(axis=1).sort_values(ascending=False))


def grafik_pasangan(pairs, n=15) -> go.Figure:
    top = pairs.nlargest(n, "n").iloc[::-1]
    label = top["kat1"] + "  +  " + top["kat2"]
    fig = go.Figure(go.Bar(
        x=top["n"], y=label, orientation="h", marker_color=SERIES[0],
        text=[f"{v:,}" for v in top["n"]], textposition="outside",
        textfont=dict(color=INK2, size=11),
        hovertemplate="%{y}<br>%{x:,} order<extra></extra>"))
    fig.update_layout(
        title=f"{n} pasangan kategori yang paling sering dibeli bersama",
        xaxis_title="jumlah order yang memuat kedua kategori",
        height=520, bargap=.28, showlegend=False)
    fig.update_xaxes(range=[0, top["n"].max() * 1.13])
    return fig


def grafik_matriks(pairs, urut) -> go.Figure:
    M = pd.DataFrame(0.0, index=urut, columns=urut)
    for _, r in pairs.iterrows():
        M.loc[r["kat1"], r["kat2"]] = r["n"]
        M.loc[r["kat2"], r["kat1"]] = r["n"]

    Z = M.to_numpy(dtype=float).copy()
    for i in range(len(urut)):
        Z[i, i] = float("nan")

    fig = go.Figure(go.Heatmap(
        z=Z, x=urut, y=urut, colorscale=SKALA, hoverongaps=False, xgap=1, ygap=1,
        colorbar=dict(title=dict(text="order"), thickness=14),
        hovertemplate="%{y}<br>%{x}<br>%{z:,.0f} order<extra></extra>"))
    fig.update_layout(
        title="Matriks co-purchase antar kategori<br>",
        height=760, margin=dict(t=110))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-45)
    return fig


@st.cache_data(show_spinner="Menyusun jaringan…")
def jaringan_kategori(pairs) -> pathlib.Path:
    vol = volume_kategori(pairs)
    net = Network(height="760px", width="100%", bgcolor=SURFACE, font_color=INK,
                  notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-38000, central_gravity=0.55,
                   spring_length=520, spring_strength=0.004, damping=0.6)
    net.options.edges.smooth.enabled = False

    for k in vol.index:
        net.add_node(k, label=k, value=float(vol[k]), color=SERIES[0],
                     font={"size": HURUF_SIMPUL, "color": INK},
                     title=f"{k} — {vol[k]:,.0f} co-purchase")
    # Semua pasangan digambar; yang membedakan hanya tebal garis lewat `value`.
    for _, r in pairs.iterrows():
        net.add_edge(r["kat1"], r["kat2"], value=float(r["n"]), color=BIRU_MUDA,
                     title=f"{r['kat1']} + {r['kat2']}: {r['n']:,} order")
    LOKAL.mkdir(parents=True, exist_ok=True)
    GRAPH_HTML.write_text(net.generate_html(notebook=False), encoding="utf-8")
    return GRAPH_HTML


def tabel_kategori(sales) -> pd.DataFrame:
    return (sales.groupby("category")
                 .agg(unit=("items", "sum"),
                      gross=("gross_sales", "sum"),
                      profit=("profit", "sum"),
                      produk_unik=("product_id", "nunique"))
                 .sort_values("gross", ascending=False))


# =============================================================== machine learning
UKURAN = {
    "R²":   dict(kol="r2",   fmt="{:.3f}",  skala=1,   naik_baik=True,
                 judul="R² — makin tinggi makin baik"),
    "RMSE": dict(kol="rmse", fmt="${:,.0f}", skala=1,  naik_baik=False,
                 judul="RMSE — makin rendah makin baik"),
    "MAPE": dict(kol="mape", fmt="{:.1f}%", skala=100, naik_baik=False,
                 judul="MAPE — makin rendah makin baik"),
}

def kartu_ml(skor) -> str:
    """Kartu ringkas: siapa yang menang, dan seberapa jauh dari patokan naive."""
    juara = skor[skor["terbaik"]].iloc[0]
    kartu = [
        ("Model terpilih", str(juara["model"])),
        ("R²", f"{juara['r2']:.3f}"),
        ("RMSE", rp(juara["rmse"])),
        ("MAPE", f"{juara['mape'] * 100:.1f}%"),
    ]
    sel = "".join(
        f'<div style="flex:1;min-width:150px;background:{SURFACE};'
        f'border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:14px 16px">'
        f'<div style="font:500 16px {FONT};color:{MUTED};letter-spacing:.02em">{k}</div>'
        f'<div style="font:600 26px {FONT};color:{INK};margin-top:6px">{v}</div></div>'
        for k, v in kartu)
    return (f'<div style="display:flex;gap:10px;flex-wrap:wrap;'
            f'font-family:{FONT};margin-bottom:18px">{sel}</div>')


def grafik_prediksi(prediksi, nama_model) -> go.Figure:
    """Time series periode uji: nilai asli vs tebakan model."""
    d = prediksi.sort_values("week")
    fig = go.Figure()
    seri = [
        ("Laba sebenarnya", "aktual", INK, 2.4, "solid"),
        (f"Tebakan {nama_model}", "prediksi", SERIES[0], 2.0, "solid"),
    ]
    for nama, kol, warna, tebal, garis in seri:
        fig.add_trace(go.Scatter(
            x=d["week"], y=d[kol], name=nama, mode="lines",
            line=dict(color=warna, width=tebal, dash=garis),
            hovertemplate="%{x|%d %b %Y}<br>" + nama + " %{y:$,.0f}<extra></extra>"))
    fig.update_layout(
        title="Periode uji: Predicted vs Actual <br>",
        height=430, hovermode="x unified",
        yaxis=dict(tickprefix="$", tickformat="~s"))
    return fig



def grafik_banding(skor, ukuran) -> go.Figure:
    """scikit-learn vs Spark MLlib untuk satu metrik."""
    u = UKURAN[ukuran]
    d = skor[skor["r2_sklearn"].notna()].copy()       
    a, b = d[u["kol"]] * u["skala"], d[f"{u['kol']}_sklearn"] * u["skala"]

    fig = go.Figure()
    for nama, nilai, warna in [("scikit-learn", b, SERIES[0]),
                               ("Spark MLlib", a, SERIES[1])]:
        fig.add_trace(go.Bar(
            x=d["model"], y=nilai, name=nama, marker_color=warna,
            text=[u["fmt"].format(v) for v in nilai],
            textposition="outside", textfont=dict(color=INK2, size=11),
            hovertemplate="%{x}<br>" + nama + ": %{text}<extra></extra>"))
    fig.update_layout(title=u["judul"], height=500, barmode="group", bargap=.28,
                      yaxis=dict(title=ukuran))
    fig.update_yaxes(rangemode="tozero")
    return fig


def grafik_dampak(dampak, nama_model, n=10) -> go.Figure:
    """Koefisien x simpangan baku fitur: semua fitur jadi satuan yang sama."""
    d = (dampak.sort_values("dampak", key=abs, ascending=False)
               .head(n).iloc[::-1])
    fig = go.Figure(go.Bar(
        x=d["dampak"], y=d["fitur"], orientation="h",
        marker=dict(color=[SERIES[0] if v >= 0 else SERIES[1] for v in d["dampak"]],
                    line=dict(color=SURFACE, width=2)),
        text=[f"{v:+,.0f}" for v in d["dampak"]], textposition="outside",
        textfont=dict(color=INK2, size=11)))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_layout(
        title=f"{n} fitur paling menentukan — {nama_model}",
        xaxis_title="Koefisien Model * Standard Deviation Data Train",
        height=460, showlegend=False, yaxis=dict(showgrid=False))
    lebar = max(d["dampak"].abs().max(), 1) * 1.3
    fig.update_xaxes(range=[-lebar if (d["dampak"] < 0).any() else 0, lebar])
    return fig


# -------------------------------------------------------------------------- app
def main() -> None:
    st.set_page_config(page_title="TheLook eCommerce", layout="wide")
    pasang_template()

    data = load()
    fact_sales = data["fact_sales"]
    dim_products = data["dim_products"]

    st.sidebar.title("TheLook eCommerce")

    with st.sidebar:
        halaman = option_menu(options=["Ringkasan Penjualan", "Perilaku Pengunjung", "Produk & Merek", "Machine Learning", "Graph"],
                              icons=['currency-dollar', 'person-circle', 'box-seam', 'cpu', 'asterisk'], 
                              menu_icon="list", 
                              menu_title="Menu",
                              default_index=0)
        

    st.sidebar.divider()
    st.sidebar.subheader("Filter")

    t_min = fact_sales["tanggal"].min().date()
    t_max = fact_sales["tanggal"].max().date()
    rentang = st.sidebar.date_input(
        "Rentang tanggal", value=(t_min, t_max),
        min_value=t_min, max_value=t_max)
    if isinstance(rentang, (list, tuple)) and len(rentang) == 2:
        mulai, selesai = (pd.Timestamp(rentang[0]), pd.Timestamp(rentang[1]))
    else:
        mulai, selesai = pd.Timestamp(t_min), pd.Timestamp(t_max)

    kategori = st.sidebar.multiselect(
        "Kategori", sorted(fact_sales["category"].unique()),
        help="Kosong berarti semua kategori.")
    status = st.sidebar.multiselect(
        "Status order", list(WARNA_STATUS),
        help="Kosong berarti semua status.")

    st.sidebar.divider()
    if st.sidebar.button("Muat ulang dari HDFS"):
        st.rerun()
    st.sidebar.caption(
        f"Data dasboard disimpan ke lokal: `{LOKAL.relative_to(ROOT.parent)}`. "
        "Tekan tombol di atas jika tabel ringkasan di HDFS sudah diperbarui.")

    sales = filter(fact_sales, mulai, selesai, kategori, status)
    orders = filter(data["fact_orders"], mulai, selesai, kategori, status)
    events = filter(data["fact_events"], mulai, selesai, kategori, status,
                  pakai_kategori=False, pakai_status=False)

    if sales.empty and halaman not in ("Machine Learning", "Graph"):
        st.warning("Tidak ada data pada filter ini. Longgarkan salah satu filter.")
        return

    # ------------------------------------------------------------------ halaman 1
    if halaman == "Ringkasan Penjualan":
        st.header("Ringkasan Penjualan")
        st.markdown(kartu_ringkasan(sales, orders), unsafe_allow_html=True)

        grain = st.radio("Butir waktu grafik", list(GRAIN), index=2,
                         horizontal=True, label_visibility="collapsed")
        st.plotly_chart(grafik_tren(sales, grain), width="stretch")

        st.plotly_chart(grafik_status(orders), width="stretch")

        st.subheader("Rincian per kategori")
        st.dataframe(
            tabel_kategori(sales).style.format({
                "unit": "{:,.0f}", "gross": "${:,.0f}", "profit": "${:,.0f}",
                "produk_unik": "{:,.0f}", "margin": "{:.1f}%"}),
            width="stretch")

    # ------------------------------------------------------------------ halaman 2
    elif halaman == "Perilaku Pengunjung":
        st.header("Perilaku Pengunjung")
        st.info(
            "Halaman ini hanya dapat menggunakan filter tanggal. Pengunjung yang sedang melihat-lihat belum memiliki kategori produk maupun status order, sehingga kedua filter tersebut tidak berlaku di sini.")

        st.plotly_chart(grafik_funnel(events), width="stretch")
        st.caption(
            "Funnel kunjungan memberikan informasi terkait berapa jumlah event dari customer yang melihat produk dan presentase hingga konversi maupun cancel.")

        st.plotly_chart(grafik_trafik(events), width="stretch")
        st.caption(
            "Sesi berdasarkan traffic memberikan informasi asal traffic mana saja yang berhasil menyalurkan traffic")

    # ------------------------------------------------------------------ halaman 3
    elif halaman == "Produk & Merek":
        st.header("Produk & Merek")
        nama_produk = dim_products.set_index("product_id")["product_name"]
        merek_produk = dim_products.set_index("product_id")["brand"]

        unit = sales.groupby("product_id")["items"].sum()
        laba = sales.groupby("product_id")["profit"].sum()
        per_merek = sales.assign(brand=sales["product_id"].map(merek_produk))

        st.subheader("Produk")
        kiri, kanan = st.columns(2)
        with kiri:
            st.plotly_chart(
                batang_top(unit, f"Top {TOP_N} produk terlaris (unit terjual)",
                           lambda v: f"{v:,.0f}", nama_produk),
                width="stretch")
        with kanan:
            st.plotly_chart(
                batang_top(laba, f"Top {TOP_N} produk paling menguntungkan",
                           rp, nama_produk),
                width="stretch")

        st.subheader("Brand")
        kiri, kanan = st.columns(2)
        with kiri:
            st.plotly_chart(
                            batang_top(per_merek.groupby("brand")["items"].sum(),
                                       f"Top {TOP_N} brand terlaris (unit terjual)",
                                       lambda v: f"{v:,.0f}"),
                            width="stretch")
        with kanan:
            st.plotly_chart(
                            batang_top(per_merek.groupby("brand")["profit"].sum(),
                                       f"Top {TOP_N} brand paling menguntungkan", rp),
                            width="stretch")
    # ------------------------------------------------------------------ halaman 4
    elif halaman == "Machine Learning":
        st.header("Sales Forecast")
        st.info(
            "Halaman ini menampilkan hasil pelatihan yang sudah tersimpan dan tidak menanggapi filter di sidebar. Modelnya dilatih sekali melalui `/src/train_mllib.py`, lalu hasilnya dibaca dari HDFS")

        try:
            ml = load_ml()
        except Exception as exc:
            st.error(
                f"Hasil pelatihan belum ada di `{HDFS_ML}`.\n\n"
                "Jalankan `code/src/train_mllib.py` lebih dulu supaya tabelnya "
                "terbentuk, lalu buka halaman ini lagi.")
            st.caption(f"Pesan aslinya: {exc}")
            return

        skor, prediksi, dampak = ml["skor"], ml["prediksi"], ml["dampak"]
        meta = ml["meta"].iloc[0]
        juara = skor[skor["terbaik"]].iloc[0]
        nama_juara = str(juara["model"])

        st.markdown(kartu_ml(skor), unsafe_allow_html=True)
        st.caption(
            f"Dilatih pada {meta['n_latih']} minggu pertama dan diuji pada "
            f"{meta['n_uji']} minggu terakhir, dipotong di {meta['batas']}. "
            f"Semua minggu uji berada setelah seluruh minggu latih, jadi model "
            f"tidak pernah melihat periode yang dinilainya. **{nama_juara} dipilih berdasarkan MAPE**")


        st.subheader("Predicted vs Actual")
        st.plotly_chart(grafik_prediksi(prediksi, nama_juara), width="stretch")

        st.subheader("Perbandingan Skor Dengan Spark MLlib dan tanpa Spark")
        ukuran = st.radio("Ukuran", list(UKURAN), index=1,
                          horizontal=True, label_visibility="collapsed")
        st.plotly_chart(grafik_banding(skor, ukuran), width="stretch")


        st.subheader("Skor Lengkap Spark MLlib")
        tabel = (skor[["model", "reg_param", "r2", "mape", "rmse", "mse"]]
                 .rename(columns={"model": "Model", "reg_param": "regParam",
                                  "r2": "R²", "mape": "MAPE", "rmse": "RMSE",
                                  "mse": "MSE"})
                 .set_index("Model"))
        st.dataframe(
            tabel.style.format({"regParam": "{:,.3f}", "R²": "{:.4f}",
                                "MAPE": "{:.4f}", "RMSE": "{:,.0f}",
                                "MSE": "{:,.0f}"}, na_rep="—"),
            width="stretch")

        st.subheader("Fitur yang Paling Berdampak Pada Model (*semi-standardized coefficient*)")
        st.plotly_chart(grafik_dampak(dampak, nama_juara), width="stretch")

    # ------------------------------------------------------------------ halaman 5

    else:
        st.header("Graph Analytics")
        st.info(
            "Halaman ini memakai seluruh riwayat order dan tidak menanggapi filter di sidebar. Angkanya dihitung dari isi satu keranjang belanja sehingga memotong data per tanggal atau per status akan mengubah arti angkanya.")

        try:
            pairs = load_graph()
        except Exception as exc:                     # tabel belum pernah dibuat
            st.error(
                f"Tabel pasangan kategori belum ada di `{HDFS_GRAPH}`.\n\n"
                "Jalankan `code/src/build_output.py` lebih dulu "
                "supaya tabelnya terbentuk, lalu buka halaman ini lagi.")
            st.caption(f"Pesan aslinya: {exc}")
            return

        vol = volume_kategori(pairs)
        kemungkinan = len(vol) * (len(vol) - 1) // 2
        k1, k2, k3 = st.columns(3)
        k1.metric("Kategori", f"{len(vol):,}")
        k2.metric("Pasangan pernah terjadi",
                  f"{len(pairs):,} / {kemungkinan:,}")
        k3.metric("Kepadatan jaringan", f"{len(pairs) / kemungkinan * 100:.0f}%")

        st.plotly_chart(grafik_pasangan(pairs), width="stretch")

        st.plotly_chart(grafik_matriks(pairs, list(vol.index)), width="stretch")

        st.subheader("Graph co-purchase")
        st.iframe(jaringan_kategori(pairs), height=780)

if __name__ == "__main__":
    main()
