# ============================================================
# app.py — Dashboard Klasterisasi Tenaga Kesehatan Jawa Barat
# Jalankan  : streamlit run app.py
# Dependensi: streamlit, pandas, numpy, matplotlib, seaborn, sklearn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import json
import os
import re
from sklearn.decomposition import PCA
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# ── Coba import library peta ──
try:
    import folium
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

try:
    from streamlit_folium import st_folium
    _ST_FOLIUM_OK = True
except ImportError:
    _ST_FOLIUM_OK = False

# ─────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Klasterisasi Tenaga Kesehatan Jawa Barat",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"   # sidebar dilipat karena tidak dipakai
)

# ─────────────────────────────────────────────────────────────
# CSS GLOBAL  (tidak diubah, hanya ditambah badge baru)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .judul-utama {
        font-size: 2rem; font-weight: 800; color: #0d2b4e;
        text-align: center; padding: 0.4rem 0 0;
    }
    .sub-judul {
        font-size: 0.95rem; color: #4a6b8a; text-align: center;
        margin-bottom: 0.5rem;
    }
    .section-head {
        font-size: 1.1rem; font-weight: 700; color: #0d2b4e;
        border-bottom: 2.5px solid #dee2e6; padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem;
    }
    .badge-sangat-tinggi { background:#c3e6f7; color:#0c5460; padding:2px 10px;
                           border-radius:12px; font-weight:700; }
    .badge-tinggi  { background:#d4edda; color:#155724; padding:2px 10px;
                     border-radius:12px; font-weight:700; }
    .badge-sedang  { background:#fff3cd; color:#856404; padding:2px 10px;
                     border-radius:12px; font-weight:700; }
    .badge-rendah  { background:#f8d7da; color:#721c24; padding:2px 10px;
                     border-radius:12px; font-weight:700; }
    .filter-box {
        background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px;
        padding:0.8rem 1rem 0.4rem; margin-bottom:1rem;
    }
    footer { text-align:center; color:#888; font-size:0.82rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KONSTANTA  — k = 4, empat label
# ─────────────────────────────────────────────────────────────
FITUR_RASIO = [
    'rasio_dokter_umum', 'rasio_dokter_spesialis', 'rasio_dokter_gigi',
    'rasio_perawat', 'rasio_bidan', 'rasio_kefarmasian',
    'rasio_kesmas', 'rasio_lingkungan', 'rasio_gizi'
]
LABEL_FITUR = [
    'Dokter Umum', 'Dokter Spesialis', 'Dokter Gigi',
    'Perawat', 'Bidan', 'Kefarmasian',
    'Kes. Masyarakat', 'Kes. Lingkungan', 'Gizi'
]
STANDAR_NASIONAL = {
    'rasio_dokter_umum'      : 50,
    'rasio_dokter_spesialis' : 12,
    'rasio_dokter_gigi'      : 14,
    'rasio_perawat'          : 200,
    'rasio_bidan'            : 130,
    'rasio_kefarmasian'      : 30,
    'rasio_kesmas'           : 18,
    'rasio_lingkungan'       : 20,
    'rasio_gizi'             : 18
}

# ── 4 warna untuk 4 klaster (urutan: Rendah → Sangat Tinggi) ──
WARNA_KLASTER = {
    'Rendah'       : '#C44E52',   # merah
    'Sedang'       : '#DD8452',   # oranye
    'Tinggi'       : '#55A868',   # hijau
    'Sangat Tinggi': '#4C72B0',   # biru
}
URUTAN_LABEL = ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
IKON_LABEL   = {
    'Rendah'       : '🔴',
    'Sedang'       : '🟠',
    'Tinggi'       : '🟢',
    'Sangat Tinggi': '🔵',
}
MARKER_MAP = {
    'Rendah'       : 'o',
    'Sedang'       : 's',
    'Tinggi'       : '^',
    'Sangat Tinggi': 'D',
}
BADGE_CSS = {
    'Rendah'       : 'badge-rendah',
    'Sedang'       : 'badge-sedang',
    'Tinggi'       : 'badge-tinggi',
    'Sangat Tinggi': 'badge-sangat-tinggi',
}

KOLOM_WILAYAH = 'Kabupaten_Kota'

# ─────────────────────────────────────────────────────────────
# FUNGSI LOAD DATA & MODEL  (tidak diubah)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path='dataset_clustered.csv'):
    return pd.read_csv(path)

@st.cache_resource
def load_artifacts():
    try:
        with open('kmeans_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_metadata():
    try:
        with open('metadata.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

# ── Muat semua data ──
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "❌ File **dataset_clustered.csv** tidak ditemukan.\n\n"
        "Pastikan file berada satu folder dengan `app.py`.\n"
        "Jalankan notebook Colab terlebih dahulu untuk menghasilkan file ini."
    )
    st.stop()

model, scaler = load_artifacts()
meta          = load_metadata()
LABEL_MAP     = meta.get('label_map', {})

# ── Pastikan kolom label_cluster ada & menggunakan 4 label ──
def buat_label_4(df_in):
    """Mapping cluster → label 4 tingkat berdasarkan rata-rata rasio (terendah→tertinggi)."""
    rata = df_in.groupby('cluster')[FITUR_RASIO].mean().sum(axis=1)
    rank = rata.rank(ascending=True).astype(int)   # rank 1 = terendah = Rendah
    return {c: URUTAN_LABEL[r - 1] for c, r in rank.items()}

# Paksa rebuild jika label tidak sesuai 4 level
if 'label_cluster' not in df.columns or \
        not set(df['label_cluster'].dropna().unique()).issubset(set(URUTAN_LABEL)):
    LABEL_MAP = buat_label_4(df)
    df['label_cluster'] = df['cluster'].map(LABEL_MAP)

# ── Pastikan nama kolom wilayah benar ──
if KOLOM_WILAYAH not in df.columns:
    KOLOM_WILAYAH = df.columns[0]

# ── Hitung X_scaled untuk PCA ──
if scaler is not None:
    X_scaled = scaler.transform(df[FITUR_RASIO])
else:
    from sklearn.preprocessing import StandardScaler
    _sc = StandardScaler()
    X_scaled = _sc.fit_transform(df[FITUR_RASIO])

# ─────────────────────────────────────────────────────────────
# HEADER UTAMA
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="judul-utama">🏥 Dashboard Klasterisasi Tenaga Kesehatan Jawa Barat</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-judul">Analisis Disparitas Rasio Tenaga Kesehatan Per Kapita • '
            '27 Kabupaten/Kota • BPS Jawa Barat 2024</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── [TAMBAHAN] Penjelasan Umum Dashboard ──
st.info(
    "**Selamat datang di Dashboard Klasterisasi Tenaga Kesehatan Jawa Barat! 👋**\n\n"
    "Dashboard ini membantu Anda memahami **seberapa merata ketersediaan tenaga kesehatan** "
    "(dokter, perawat, bidan, dan lainnya) di 27 kabupaten/kota di Jawa Barat.\n\n"
    "Dengan dashboard ini, Anda bisa mengetahui:\n"
    "- 📍 Wilayah mana yang sudah memiliki tenaga kesehatan yang cukup\n"
    "- ⚠️ Wilayah mana yang masih kekurangan dan perlu mendapat perhatian lebih\n"
    "- 📊 Perbandingan jumlah tenaga kesehatan antar wilayah dan dengan standar nasional\n\n"
    "Data bersumber dari **BPS Provinsi Jawa Barat tahun 2024**, dianalisis menggunakan "
    "metode pengelompokan otomatis (K-Means Clustering)."
)

st.markdown(
    """
    > 💡 **Cara membaca dashboard ini:** Gunakan tab-tab di bawah untuk berpindah antara
    > tampilan peta, grafik, dan tabel. Setiap bagian dilengkapi penjelasan agar mudah dipahami.
    """
)

# ─────────────────────────────────────────────────────────────
# METRIC SUMMARY  — 4 klaster + total = 5 kolom
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">📊 Ringkasan Klasterisasi (k = 4)</div>',
            unsafe_allow_html=True)

# ── [TAMBAHAN] Penjelasan Klasterisasi ──
st.markdown(
    """
    **Apa itu klaster?** Klaster adalah hasil pengelompokan wilayah berdasarkan kemiripan
    kondisi tenaga kesehatannya. Setiap kabupaten/kota dimasukkan ke dalam salah satu dari
    4 kelompok berikut, berdasarkan jumlah tenaga kesehatan yang tersedia dibandingkan
    jumlah penduduknya.
    """
)

cnt = df.groupby('label_cluster')[KOLOM_WILAYAH].count()
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Total Wilayah",     f"{len(df)}",                          "Kab/Kota Jawa Barat")
mc2.metric("🔴 Rendah",         f"{cnt.get('Rendah', 0)} wilayah",     "Prioritas pemerataan")
mc3.metric("🟠 Sedang",         f"{cnt.get('Sedang', 0)} wilayah")
mc4.metric("🟢 Tinggi",         f"{cnt.get('Tinggi', 0)} wilayah")
mc5.metric("🔵 Sangat Tinggi",  f"{cnt.get('Sangat Tinggi', 0)} wilayah")

# ── [TAMBAHAN] Konteks ringkasan ──
st.caption(
    "Angka di atas menunjukkan jumlah kabupaten/kota yang masuk ke masing-masing kelompok. "
    "Semakin banyak wilayah di klaster Rendah, semakin besar ketimpangan yang perlu diatasi."
)

# ─────────────────────────────────────────────────────────────
# PETA CHOROPLETH — DISTRIBUSI KLASTER JAWA BARAT
#
# Dependensi tambahan (jalankan satu kali):
#   pip install folium streamlit-folium requests
#
# GeoJSON:
#   Simpan file batas wilayah kabupaten/kota Jawa Barat sebagai
#   'jabar.geojson' di folder yang sama dengan app.py.
#   Atau biarkan kosong — kode akan mengunduh otomatis dari
#   sumber publik (butuh koneksi internet).
#
#   Sumber yang direkomendasikan:
#   https://github.com/superpikar/indonesia-geojson
#   (unduh indonesia-kab.json, rename → jabar.geojson, ATAU
#    biarkan kode mengunduh & memfilter otomatis)
# ─────────────────────────────────────────────────────────────

# ── Warna peta (sama dengan WARNA_KLASTER) ──
_WARNA_PETA = {
    'Rendah'       : '#C44E52',
    'Sedang'       : '#DD8452',
    'Tinggi'       : '#55A868',
    'Sangat Tinggi': '#4C72B0',
}

# ── Kunci properti GeoJSON yang akan dicoba untuk nama wilayah ──
_NAME_KEYS = [
    'Kabupaten', 'KABUPATEN', 'name', 'Name', 'NAME',
    'nama', 'Nama', 'NAMA', 'WADMKK', 'KABKOT',
    'nmkab', 'kabkot', 'district', 'District',
]


@st.cache_data(show_spinner=False)
def _muat_geojson():
    """
    Coba muat GeoJSON dari file lokal terlebih dahulu.
    Jika tidak ada, unduh dari sumber publik dan filter Jawa Barat.
    Mengembalikan (geojson_dict | None, pesan_error | None).
    """
    # 1. File lokal
    if os.path.exists('jabar.geojson'):
        try:
            with open('jabar.geojson', 'r', encoding='utf-8') as f:
                return json.load(f), None
        except Exception as e:
            return None, f"Gagal membaca jabar.geojson: {e}"

    # 2. Unduh otomatis
    try:
        import requests
    except ImportError:
        return None, (
            "Library **requests** tidak ditemukan. "
            "Jalankan: `pip install requests`"
        )

    # Daftar URL yang akan dicoba (prioritas dari atas)
    urls = [
        # Kabupaten/kota seluruh Indonesia dari superpikar
        'https://raw.githubusercontent.com/superpikar/indonesia-geojson/'
        'master/indonesia-kab.json',
        # Alternatif dari ans-4175
        'https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/'
        'master/jawa_barat/jawa-barat-kabupaten.min.json',
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            raw = resp.json()
            feats = raw.get('features', [])

            # Filter hanya fitur Jawa Barat
            jabar = []
            for feat in feats:
                props = feat.get('properties', {})
                prov  = ' '.join(str(v) for v in props.values()).upper()
                # Masukkan jika mengandung penanda Jawa Barat,
                # atau jika file sudah spesifik Jawa Barat (ans-4175)
                if ('JAWA BARAT' in prov or 'WEST JAVA' in prov
                        or 'ans-4175' in url):
                    jabar.append(feat)

            if len(jabar) >= 20:   # minimal 20 dari 27 kab/kota
                return {'type': 'FeatureCollection', 'features': jabar}, None
        except Exception:
            continue  # coba URL berikutnya

    return None, (
        "**GeoJSON tidak dapat dimuat secara otomatis.**\n\n"
        "Harap unduh file GeoJSON batas wilayah kabupaten/kota Jawa Barat "
        "secara manual dan simpan sebagai `jabar.geojson` di folder yang sama "
        "dengan `app.py`.\n\n"
        "**Sumber yang direkomendasikan:**\n"
        "- https://github.com/superpikar/indonesia-geojson "
        "(unduh `indonesia-kab.json`, rename jadi `jabar.geojson`)\n"
        "- https://github.com/ans-4175/peta-indonesia-geojson\n\n"
        "Setelah menyimpan file, muat ulang halaman ini."
    )


# ── 27 Kabupaten/Kota Jawa Barat (nama persis sesuai dataset) ──
# Format: Kabupaten → nama saja (tanpa prefix)
#         Kota      → "Kota Xxx"
_WILAYAH_JABAR_27 = [
    # 18 Kabupaten
    'Bogor', 'Sukabumi', 'Cianjur', 'Bandung', 'Garut',
    'Tasikmalaya', 'Ciamis', 'Kuningan', 'Cirebon', 'Majalengka',
    'Sumedang', 'Indramayu', 'Subang', 'Purwakarta', 'Karawang',
    'Bekasi', 'Bandung Barat', 'Pangandaran',
    # 9 Kota
    'Kota Bogor', 'Kota Sukabumi', 'Kota Bandung', 'Kota Cirebon',
    'Kota Bekasi', 'Kota Depok', 'Kota Cimahi', 'Kota Tasikmalaya',
    'Kota Banjar',
]


def _clean(s):
    """Bersihkan string: uppercase, hapus non-alfanumerik, normalisasi spasi."""
    s = str(s).upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _buat_lookup(df_kl):
    """
    Bangun tabel lookup komprehensif:
      normalized_geojson_name (str) → (df_exact_name, label_cluster)

    Dataset memakai konvensi:
      • Kabupaten : nama inti saja, mis. "Bogor", "Bandung Barat"
      • Kota      : "Kota Xxx", mis. "Kota Bogor", "Kota Bandung"

    GeoJSON bisa memakai berbagai format, mis.:
      • "KABUPATEN BOGOR", "KAB. BOGOR", "BOGOR"  → cocok ke "Bogor"
      • "KOTA BOGOR"                               → cocok ke "Kota Bogor"
      • "KABUPATEN BANDUNG BARAT"                  → cocok ke "Bandung Barat"
      • "KOTA BANDUNG"                             → cocok ke "Kota Bandung"
    """
    lookup = {}   # key (uppercase str) → (df_name, klaster)

    for _, row in df_kl.iterrows():
        df_name = str(row[KOLOM_WILAYAH]).strip()
        klstr   = row['label_cluster']
        upper   = _clean(df_name)
        is_kota = upper.startswith('KOTA ')
        core    = upper[5:].strip() if is_kota else upper   # nama inti

        if is_kota:
            # "Kota Bogor" → cocok ke "KOTA BOGOR"
            for key in [upper, 'KOTA ' + core]:
                lookup[key] = (df_name, klstr)
        else:
            # "Bogor" → cocok ke "BOGOR", "KABUPATEN BOGOR",
            #           "KAB BOGOR", "KAB  BOGOR", "KABBOGOR" dst.
            for key in [
                core,
                'KABUPATEN ' + core,
                'KAB ' + core,
                'KAB  ' + core,
            ]:
                lookup[key] = (df_name, klstr)

    return lookup


def _cocokkan_nama(raw_gj, lookup):
    """
    Cocokkan satu nama GeoJSON ke entry di lookup.
    Strategi (berhenti di kecocokan pertama):
      1. Exact match setelah _clean()
      2. Buang semua prefix KAB(UPATEN)/KAB lalu exact match
      3. Tambah 'KOTA' lalu cari cocok (untuk GeoJSON yang sudah buang 'KOTA')
      4. Substring match pada core name
    """
    cleaned = _clean(raw_gj)

    # Tahap 1: exact
    if cleaned in lookup:
        return lookup[cleaned]

    # Tahap 2: strip prefix KABUPATEN / KAB
    core = re.sub(r'^(KABUPATEN|KAB\.?)\s+', '', cleaned).strip()
    if core in lookup:
        return lookup[core]
    if 'KABUPATEN ' + core in lookup:
        return lookup['KABUPATEN ' + core]

    # Tahap 3: coba tempelkan KOTA di depan core (GeoJSON tanpa prefix)
    if 'KOTA ' + core in lookup:
        return lookup['KOTA ' + core]

    # Tahap 4: substring — core GeoJSON ⊆ core df  atau  core df ⊆ core GeoJSON
    for key, val in lookup.items():
        key_core = re.sub(r'^(KABUPATEN|KAB\.?|KOTA)\s+', '', key).strip()
        if key_core and core and (key_core == core
                or key_core in core or core in key_core):
            return val

    return None   # tidak ketemu


def _deteksi_key_nama(gj_data):
    """Deteksi otomatis kunci nama wilayah dalam properti GeoJSON."""
    if not gj_data.get('features'):
        return None
    sample = gj_data['features'][0].get('properties', {})
    for k in _NAME_KEYS:
        if k in sample and isinstance(sample[k], str):
            return k
    for k, v in sample.items():
        if isinstance(v, str):
            return k
    return None


def _buat_peta(gj_data, df_kl):
    """
    Buat objek folium.Map dengan choropleth klaster.
    Mengembalikan (folium.Map | None, pesan_error | None).
    """
    name_key = _deteksi_key_nama(gj_data)
    if name_key is None:
        return None, "Tidak dapat mendeteksi kunci nama wilayah dalam GeoJSON."

    lookup = _buat_lookup(df_kl)

    matched_names = []   # untuk debugging
    unmatched_names = []

    matched, unmatched = 0, 0
    for feat in gj_data['features']:
        raw    = feat['properties'].get(name_key, '')
        result = _cocokkan_nama(raw, lookup)

        if result:
            df_nm, klstr = result
            matched += 1
            matched_names.append(f"{raw} → {df_nm}")
        else:
            df_nm  = raw
            klstr  = None
            unmatched += 1
            unmatched_names.append(raw)

        feat['properties']['_klaster'] = klstr or '—'
        feat['properties']['_warna']   = _WARNA_PETA.get(klstr, '#CCCCCC')
        # Tampilkan nama dari dataset jika cocok, fallback ke nama GeoJSON
        feat['properties']['_wilayah'] = df_nm

    # Buat peta — terpusat di Jawa Barat
    m = folium.Map(
        location=[-6.90, 107.55],
        zoom_start=8,
        tiles='CartoDB positron',
        scrollWheelZoom=True,
        attributionControl=False,
    )

    folium.GeoJson(
        gj_data,
        name='Klaster Tenaga Kesehatan',
        style_function=lambda feat: {
            'fillColor'  : feat['properties'].get('_warna', '#CCCCCC'),
            'color'      : '#555555',
            'weight'     : 1.0,
            'fillOpacity': 0.80,
        },
        highlight_function=lambda _: {
            'weight'     : 2.5,
            'color'      : '#222222',
            'fillOpacity': 0.95,
        },
        tooltip=folium.GeoJsonTooltip(
            fields    =['_wilayah', '_klaster'],
            aliases   =['📍 Wilayah:', '🏥 Klaster:'],
            style     =(
                'background-color:#fff; border-radius:6px; '
                'font-family:sans-serif; font-size:13px; '
                'padding:6px 10px; box-shadow:0 1px 4px rgba(0,0,0,.2);'
            ),
            localize  =True,
        ),
    ).add_to(m)

    # ── Legend ──
    legend_html = """
    <div style="
        position : fixed;
        bottom   : 30px;
        left     : 30px;
        z-index  : 1000;
        background: white;
        padding  : 12px 18px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.22);
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
        line-height: 1.8;
    ">
        <b style="font-size:14px;">🏥 Klaster Tenaga Kesehatan</b><br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
                     background:#C44E52;margin-right:8px;vertical-align:middle;"></span>
        Rendah<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
                     background:#DD8452;margin-right:8px;vertical-align:middle;"></span>
        Sedang<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
                     background:#55A868;margin-right:8px;vertical-align:middle;"></span>
        Tinggi<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
                     background:#4C72B0;margin-right:8px;vertical-align:middle;"></span>
        Sangat Tinggi<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
                     background:#CCCCCC;margin-right:8px;vertical-align:middle;"></span>
        Data tidak tersedia
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    info = None
    if unmatched > 0:
        detail = ', '.join(unmatched_names[:10])
        info   = (
            f"ℹ️ **{matched}/{matched+unmatched}** wilayah berhasil dipetakan. "
            f"{unmatched} wilayah GeoJSON tidak cocok (abu-abu): `{detail}`."
        )
    return m, info


# ── Tampilkan seksi peta ──
st.markdown("---")
st.markdown(
    '<div class="section-head">🗺️ Peta Distribusi Klaster Kabupaten/Kota Jawa Barat</div>',
    unsafe_allow_html=True
)

# ── [TAMBAHAN] Penjelasan Peta ──
st.markdown(
    """
    Peta di bawah ini menampilkan **sebaran kondisi tenaga kesehatan** di seluruh
    kabupaten/kota Jawa Barat. Setiap wilayah diwarnai berdasarkan kelompok (klaster)
    yang telah ditentukan dari hasil analisis data.

    **Cara membaca peta:**
    - 🔴 **Merah** → Klaster Rendah
    - 🟠 **Oranye** → Klaster Sedang
    - 🟢 **Hijau** → Klaster Tinggi
    - 🔵 **Biru** → Klaster Sangat Tinggi
    - ⬜ **Abu-abu** → Data tidak tersedia untuk wilayah tersebut

    💡 *Arahkan kursor ke wilayah untuk melihat nama dan klasternya.*
    """
)

if not _FOLIUM_OK:
    st.error(
        "⚠️ Library **folium** belum terpasang.\n\n"
        "Jalankan perintah berikut di terminal, lalu muat ulang:\n"
        "```\npip install folium streamlit-folium requests\n```"
    )
else:
    with st.spinner("Memuat data peta Jawa Barat…"):
        _gj_data, _gj_err = _muat_geojson()

    if _gj_err or _gj_data is None:
        st.warning(_gj_err or "GeoJSON tidak dapat dimuat.")
    else:
        _peta, _info = _buat_peta(_gj_data, df)
        if _peta is None:
            st.warning(f"⚠️ {_info}")
        else:
            if _info:
                st.info(_info)
            if _ST_FOLIUM_OK:
                st_folium(
                    _peta,
                    width  ="100%",
                    height =500,
                    returned_objects=[],
                    key="peta_jabar",
                )
            else:
                components.html(_peta._repr_html_(), height=520, scrolling=False)
                st.caption(
                    "💡 Untuk tampilan interaktif yang lebih baik, "
                    "install: `pip install streamlit-folium`"
                )

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TAB NAVIGASI  — 4 tab (Scatter+Tabel digabung)
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔵 Scatter Plot & Tabel",
    "📊 Bar Chart",
    "🎯 Standar Nasional",
    "🔎 Detail Wilayah",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SCATTER PLOT + TABEL DATA (digabung, filter dari atas)
# ══════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-head">Scatter Plot Klaster K-Means — PCA 2D</div>',
                unsafe_allow_html=True)

    # ── [TAMBAHAN] Penjelasan Scatter Plot & PCA ──
    st.info(
        "**Apa yang ditampilkan grafik ini?**\n\n"
        "Grafik ini memperlihatkan posisi setiap kabupaten/kota berdasarkan "
        "keseluruhan data tenaga kesehatannya — dari 9 jenis sekaligus. "
        "Karena mustahil menggambar 9 dimensi sekaligus, data disederhanakan menjadi "
        "2 sumbu (horizontal dan vertikal) menggunakan teknik yang disebut **PCA**. "
        "Anggap saja seperti 'foto dari atas' yang merangkum semua informasi penting.\n\n"
        "**Cara membacanya:**\n"
        "- Setiap **titik** mewakili satu kabupaten/kota\n"
        "- **Warna & bentuk** titik menunjukkan klaster wilayah tersebut\n"
        "- Titik yang **berdekatan** artinya kondisi tenaga kesehatannya mirip\n"
        "- **Bintang (★)** besar di tengah tiap kelompok adalah pusat rata-rata klaster tersebut\n"
        "- Wilayah yang **berjauhan** di grafik artinya kondisi tenaga kesehatannya sangat berbeda"
    )

    # 🔎 FILTER KHUSUS TAB 1
    semua_label = [l for l in URUTAN_LABEL if l in df['label_cluster'].unique()]

    pilih_cluster = st.multiselect(
        "🔎 Filter Klaster:",
        options=semua_label,
        default=semua_label,
        help="Filter hanya untuk Scatter Plot & Tabel"
    )

    # ✅ FILTER DATA DI SINI
    df_filtered = df[df['label_cluster'].isin(pilih_cluster)].copy()
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    expl  = pca.explained_variance_ratio_

    idx_filtered = df_filtered.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 7.5))
    plotted = set()

    for i in idx_filtered:
        row    = df.loc[i]
        label  = row['label_cluster']
        color  = WARNA_KLASTER.get(label, '#888')
        marker = MARKER_MAP.get(label, 'o')
        lbl    = f"Klaster {label}" if label not in plotted else '_nolegend_'
        ax.scatter(X_pca[i, 0], X_pca[i, 1],
                   c=color, marker=marker, s=140,
                   edgecolors='white', linewidth=0.8,
                   label=lbl, zorder=3)
        plotted.add(label)
        ax.annotate(row[KOLOM_WILAYAH], (X_pca[i, 0], X_pca[i, 1]),
                    textcoords='offset points', xytext=(7, 4),
                    fontsize=7.5, color='#222')

    if model is not None:
        centroids_pca = pca.transform(model.cluster_centers_)
        for c, (cx, cy) in enumerate(centroids_pca):
            lc = LABEL_MAP.get(c, '')
            ax.scatter(cx, cy, c=WARNA_KLASTER.get(lc, '#555'),
                       marker='*', s=480, edgecolors='black', linewidth=1.2, zorder=5)

    ax.set_title(
        f"Klasterisasi K-Means (k=4) — PCA 2D\n"
        f"PC1={expl[0]*100:.1f}%  PC2={expl[1]*100:.1f}%  Total={sum(expl)*100:.1f}%",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)")

    # Legend diurutkan sesuai URUTAN_LABEL
    handles_leg, labels_leg = ax.get_legend_handles_labels()
    order = {f"Klaster {l}": i for i, l in enumerate(URUTAN_LABEL)}
    paired = sorted(zip(labels_leg, handles_leg), key=lambda x: order.get(x[0], 99))
    if paired:
        lbl_sorted, hdl_sorted = zip(*paired)
        ax.legend(hdl_sorted, lbl_sorted, fontsize=10)
    else:
        ax.legend(fontsize=10)

    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── [TAMBAHAN] Catatan bawah scatter plot ──
    st.caption(
        f"Dua sumbu grafik (PC1 & PC2) bersama-sama menjelaskan "
        f"{sum(expl)*100:.1f}% dari total variasi data — "
        "semakin tinggi persentase ini, semakin akurat gambaran yang ditampilkan."
    )

    st.markdown("---")

    # ── Tabel Data (dalam tab yang sama) ──
    st.markdown('<div class="section-head">Tabel Data Wilayah</div>',
                unsafe_allow_html=True)

    # ── [TAMBAHAN] Penjelasan Tabel & Rasio ──
    st.markdown(
        """
        Tabel di bawah menampilkan data lengkap setiap kabupaten/kota beserta
        angka rasio masing-masing jenis tenaga kesehatan.

        **Apa itu "rasio per 100.000 penduduk"?**
        Rasio ini menjawab pertanyaan: *"Dari setiap 100.000 orang penduduk,
        ada berapa tenaga kesehatan yang tersedia?"*
        Misalnya, rasio dokter umum = 30 artinya ada **30 dokter umum**
        untuk melayani setiap 100.000 penduduk di wilayah tersebut.
        Semakin besar angkanya, semakin baik ketersediaannya.

        Warna pada kolom **Klaster** menunjukkan kategori masing-masing wilayah —
        merah untuk Rendah, oranye untuk Sedang, hijau untuk Tinggi, biru untuk Sangat Tinggi.
        """
    )

    kolom_tampil    = [KOLOM_WILAYAH, 'label_cluster'] + FITUR_RASIO
    df_show         = df_filtered[kolom_tampil].copy()
    df_show.columns = ['Kabupaten/Kota', 'Klaster'] + LABEL_FITUR

    def warnai_klaster(val):
        w = {
            'Sangat Tinggi': 'background-color:#c3e6f7; color:#0c5460',
            'Tinggi'       : 'background-color:#d4edda; color:#155724',
            'Sedang'       : 'background-color:#fff3cd; color:#856404',
            'Rendah'       : 'background-color:#f8d7da; color:#721c24',
        }
        return w.get(val, '')

    st.dataframe(
        df_show.style
        .applymap(warnai_klaster, subset=['Klaster'])
        .format({k: '{:.2f}' for k in LABEL_FITUR}),
        use_container_width=True, height=430
    )
    st.caption(f"Menampilkan {len(df_show)} dari {len(df)} wilayah.")

    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Tabel (CSV)", csv_data,
                       file_name="klaster_filtered.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════
# TAB 2 — BAR CHART RATA-RATA PER CLUSTER
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">Rata-rata Rasio Tenaga Kesehatan per Klaster</div>',
                unsafe_allow_html=True)

    # ── [TAMBAHAN] Penjelasan Bar Chart ──
    st.info(
        "**Apa yang ditampilkan grafik ini?**\n\n"
        "Grafik batang (bar chart) ini membandingkan rata-rata ketersediaan "
        "setiap jenis tenaga kesehatan untuk masing-masing klaster. "
        "Dengan melihat grafik ini, Anda bisa langsung tahu:\n"
        "- Jenis tenaga kesehatan mana yang paling berbeda antar klaster\n"
        "- Apakah perbedaan antar klaster besar atau kecil untuk setiap jenis\n\n"
        "**Cara membacanya:** Setiap kelompok batang mewakili satu jenis tenaga kesehatan. "
        "Batang yang lebih tinggi = rata-rata lebih banyak tenaga kesehatan di klaster tersebut. "
        "Bandingkan tinggi batang merah 🔴, oranye 🟠, hijau 🟢, dan biru 🔵 "
        "untuk melihat seberapa besar kesenjangan antar klaster."
    )

    rata_cluster  = df.groupby('label_cluster')[FITUR_RASIO].mean()
    rata_provinsi = df[FITUR_RASIO].mean()

    urutan_ada = [l for l in URUTAN_LABEL if l in rata_cluster.index]
    n          = len(urutan_ada)
    bar_width  = 0.18
    offs       = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * bar_width

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(FITUR_RASIO))

    for i, label in enumerate(urutan_ada):
        vals = [rata_cluster.loc[label, f] for f in FITUR_RASIO]
        ax.bar(x + offs[i], vals, bar_width,
               label=f"Klaster {label}",
               color=WARNA_KLASTER[label], edgecolor='white', alpha=0.88)

    ax.set_title("Rata-rata Rasio Tenaga Kesehatan per Klaster (k=4)",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_FITUR, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Rasio per 100.000 Penduduk")
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── [TAMBAHAN] Catatan bawah bar chart ──
    st.caption(
        "Sumbu vertikal (Y) menunjukkan rasio per 100.000 penduduk — "
        "semakin tinggi batang, semakin banyak tenaga kesehatan relatif terhadap jumlah penduduk."
    )

    st.markdown("**Tabel Rata-rata Rasio per Klaster:**")

    # ── [TAMBAHAN] Penjelasan tabel rata-rata ──
    st.markdown(
        "Tabel di bawah menyajikan angka pasti dari grafik di atas. "
        "Gradasi warna dari kuning ke merah menunjukkan nilai dari rendah ke tinggi — "
        "warna lebih tua berarti nilai lebih besar."
    )

    df_rata         = rata_cluster.loc[urutan_ada, FITUR_RASIO].round(2).copy()
    df_rata.columns = LABEL_FITUR
    st.dataframe(df_rata.style.background_gradient(cmap='YlOrRd', axis=1),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — PERBANDINGAN STANDAR NASIONAL
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">Perbandingan Rasio dengan Standar Nasional</div>',
                unsafe_allow_html=True)

    # ── [TAMBAHAN] Penjelasan Standar Nasional ──
    st.info(
        "**Apa itu Standar Nasional?**\n\n"
        "Standar Nasional adalah angka minimum tenaga kesehatan yang **seharusnya** tersedia "
        "untuk setiap 100.000 penduduk, sesuai dengan kebijakan pemerintah Indonesia "
        "(Keputusan Menko Kesra No. 54 Tahun 2013). "
        "Angka ini menjadi tolok ukur apakah sebuah wilayah sudah terpenuhi kebutuhan "
        "tenaga kesehatannya atau belum.\n\n"
    )

    st.caption("Standar: Keputusan Menko Kesra No. 54 Tahun 2013")

    pilih_jenis = st.selectbox(
        "Pilih Jenis Tenaga Kesehatan:",
        options=FITUR_RASIO,
        format_func=lambda x: LABEL_FITUR[FITUR_RASIO.index(x)]
    )

    # ── [TAMBAHAN] Penjelasan cara baca grafik standar nasional ──
    st.markdown(
        f"""
        **Cara membaca grafik ini:**
        - Setiap **batang horizontal** mewakili satu kabupaten/kota,
          panjangnya menunjukkan jumlah tenaga kesehatan yang tersedia
        - **Garis merah putus-putus** adalah batas standar nasional —
          wilayah yang batangnya **tidak mencapai garis ini** berarti masih kekurangan
        - **Garis biru titik-titik** adalah rata-rata seluruh Jawa Barat
        - Warna batang mengikuti klaster wilayah tersebut
        """
    )

    df_std_plot = df[[KOLOM_WILAYAH, pilih_jenis, 'label_cluster']].sort_values(pilih_jenis)
    rata_prov   = df[pilih_jenis].mean()
    std_val     = STANDAR_NASIONAL[pilih_jenis]

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_col = [WARNA_KLASTER.get(l, '#4C72B0') for l in df_std_plot['label_cluster']]
    ax.barh(df_std_plot[KOLOM_WILAYAH], df_std_plot[pilih_jenis],
            color=bar_col, edgecolor='white', alpha=0.88)
    ax.axvline(x=std_val, color='red', linestyle='--', linewidth=2.2,
               label=f'Standar Nasional: {std_val}')
    ax.axvline(x=rata_prov, color='royalblue', linestyle=':', linewidth=2,
               label=f'Rata-rata Provinsi: {rata_prov:.2f}')

    ax.set_title(
        f"Rasio {LABEL_FITUR[FITUR_RASIO.index(pilih_jenis)]} per Kabupaten/Kota\nvs Standar Nasional",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel("Rasio per 100.000 Penduduk")

    legend_patches = [mpatches.Patch(color=WARNA_KLASTER[l], label=f'Klaster {l}')
                      for l in URUTAN_LABEL if l in df['label_cluster'].unique()]
    all_h = ax.get_legend_handles_labels()[0] + legend_patches
    all_l = ax.get_legend_handles_labels()[1] + [p.get_label() for p in legend_patches]
    ax.legend(all_h, all_l, fontsize=9, loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Gap Rata-rata Klaster terhadap Standar Nasional:**")

    # ── [TAMBAHAN] Penjelasan tabel gap ──
    st.markdown(
        "Tabel berikut merangkum seberapa jauh rata-rata setiap klaster dari standar nasional. "
        "Kolom **Gap** bertanda positif (✅) berarti sudah memenuhi standar, "
        "sedangkan negatif (⚠️) berarti masih ada kekurangan yang perlu dikejar."
    )

    rata_c = df.groupby('label_cluster')[pilih_jenis].mean()
    rata_c = rata_c.reindex([l for l in URUTAN_LABEL if l in rata_c.index])
    df_gap = pd.DataFrame({
        'Klaster'           : rata_c.index,
        'Rata-rata Rasio'   : rata_c.round(2).values,
        'Standar Nasional'  : std_val,
        'Gap (Rasio − Std)' : (rata_c - std_val).round(2).values
    })
    df_gap['Status'] = df_gap['Gap (Rasio − Std)'].apply(
        lambda x: '✅ Memenuhi' if x >= 0 else '⚠️ Belum Memenuhi'
    )
    st.dataframe(df_gap, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — DETAIL WILAYAH  (dropdown wilayah di dalam tab ini)
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-head">Detail Wilayah Terpilih</div>',
                unsafe_allow_html=True)

    # ── [TAMBAHAN] Penjelasan Tab Detail ──
    st.markdown(
        """
        Di sini Anda bisa melihat **profil lengkap satu kabupaten/kota** secara mendalam —
        mulai dari klaster yang dimiliki, angka tiap jenis tenaga kesehatan,
        hingga perbandingannya dengan standar nasional.

        Pilih wilayah dari dropdown di bawah untuk mulai menjelajah.
        """
    )

    # Dropdown wilayah hanya ada di tab ini
    daftar_wilayah = sorted(df[KOLOM_WILAYAH].dropna().tolist())
    pilih_wilayah  = st.selectbox(
        "🗺️ Pilih Kabupaten/Kota:",
        options=["— Pilih Wilayah —"] + daftar_wilayah,
        key="detail_wilayah"
    )

    if pilih_wilayah == "— Pilih Wilayah —":
        # Tampilkan daftar anggota tiap klaster jika belum memilih
        st.info("Pilih kabupaten/kota di atas untuk melihat detail profil rasio wilayah tersebut.")

        # ── [TAMBAHAN] Penjelasan daftar anggota klaster ──
        st.markdown(
            "Sambil menunggu, berikut daftar lengkap anggota setiap klaster. "
            "Klik nama wilayah di dropdown di atas untuk melihat detail profilnya."
        )

        st.markdown("**Daftar Anggota Setiap Klaster:**")

        label_ada    = [l for l in URUTAN_LABEL if l in df['label_cluster'].unique()]
        cols_klaster = st.columns(len(label_ada))

        for col_ui, klabel in zip(cols_klaster, label_ada):
            with col_ui:
                anggota = df[df['label_cluster'] == klabel][KOLOM_WILAYAH].tolist()
                ikon    = IKON_LABEL[klabel]
                st.markdown(f"**{ikon} Klaster {klabel}** ({len(anggota)} wilayah)")
                for w in anggota:
                    st.markdown(f"• {w}")

    else:
        row    = df[df[KOLOM_WILAYAH] == pilih_wilayah].iloc[0]
        klabel = row['label_cluster']
        badge  = BADGE_CSS.get(klabel, '')

        st.markdown(
            f"**Wilayah:** {pilih_wilayah} &nbsp;|&nbsp; "
            f"**Klaster:** <span class='{badge}'>{klabel}</span>",
            unsafe_allow_html=True
        )

        # ── [TAMBAHAN] Konteks kartu metrik ──
        st.markdown(
            "Kartu-kartu di bawah menampilkan angka rasio untuk 6 jenis tenaga kesehatan utama. "
            "Angka **hijau (↑)** berarti wilayah ini **sudah melampaui** standar nasional, "
            "sedangkan angka **merah (↓)** berarti **masih di bawah** standar. "
            "Tanda **Δ** (delta) menunjukkan selisih dari standar nasional."
        )

        # Metric per jenis nakes
        metrics = [
            ('Dokter Umum',      'rasio_dokter_umum',      50),
            ('Perawat',          'rasio_perawat',           200),
            ('Bidan',            'rasio_bidan',             130),
            ('Dokter Spesialis', 'rasio_dokter_spesialis',  12),
            ('Kefarmasian',      'rasio_kefarmasian',       30),
            ('Kes. Lingkungan',  'rasio_lingkungan',        20),
        ]
        c1, c2, c3 = st.columns(3)
        for idx, (nama, fitur, std_n) in enumerate(metrics):
            val   = row[fitur]
            delta = round(val - std_n, 2)
            # st.metric mewarnai delta berdasarkan apakah string diawali "-".
            # Format "Std:… | Δ…" tidak diawali "-" meski nilainya negatif,
            # sehingga warna selalu hijau. Solusi: gunakan delta_color="inverse"
            # saat delta negatif — Streamlit membalik warna menjadi merah.
            delta_color = "normal" if delta >= 0 else "inverse"
            [c1, c2, c3][idx % 3].metric(
                nama,
                f"{val:.2f}",
                f"Std:{std_n} | Δ{delta:+.2f}",
                delta_color=delta_color,
            )

        # Bar chart profil wilayah
        st.markdown("**Profil Rasio vs Standar Nasional:**")

        # ── [TAMBAHAN] Penjelasan grafik profil wilayah ──
        st.caption(
            f"Grafik berikut membandingkan angka tenaga kesehatan di {pilih_wilayah} "
            "(batang berwarna) dengan standar nasional (batang abu-abu) untuk setiap jenis. "
            "Jika batang berwarna lebih pendek dari batang abu-abu, "
            "artinya jenis tenaga kesehatan tersebut masih kurang di wilayah ini."
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        x_d = np.arange(len(LABEL_FITUR))
        ax.bar(x_d - 0.2, [row[f] for f in FITUR_RASIO], 0.38,
               label=pilih_wilayah,
               color=WARNA_KLASTER.get(klabel, '#4C72B0'),
               edgecolor='white', alpha=0.9)
        ax.bar(x_d + 0.2, [STANDAR_NASIONAL[f] for f in FITUR_RASIO], 0.38,
               label='Standar Nasional', color='#6c757d', edgecolor='white', alpha=0.72)
        ax.set_title(
            f"Profil Rasio Tenaga Kesehatan — {pilih_wilayah}  [Klaster: {klabel}]",
            fontsize=12, fontweight='bold'
        )
        ax.set_xticks(x_d)
        ax.set_xticklabels(LABEL_FITUR, rotation=28, ha='right', fontsize=9)
        ax.set_ylabel("Rasio per 100.000 Penduduk")
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tabel lengkap
        st.markdown("**Tabel Nilai Lengkap:**")

        # ── [TAMBAHAN] Penjelasan tabel lengkap wilayah ──
        st.caption(
            "Tabel ini memuat semua 9 jenis tenaga kesehatan beserta angka rasio, "
            "standar nasional, dan selisihnya (Gap). "
            "Gap berwarna hijau = memenuhi standar; merah = masih kurang. "
            "Gunakan tabel ini untuk mengetahui jenis tenaga kesehatan mana yang "
            "paling perlu ditingkatkan di wilayah ini."
        )

        df_det = pd.DataFrame({
            'Jenis Tenaga Kesehatan': LABEL_FITUR,
            'Rasio Wilayah'         : [round(row[f], 4) for f in FITUR_RASIO],
            'Standar Nasional'      : [STANDAR_NASIONAL[f] for f in FITUR_RASIO],
            'Gap'                   : [round(row[f] - STANDAR_NASIONAL[f], 4)
                                       for f in FITUR_RASIO]
        })
        df_det['Status'] = df_det['Gap'].apply(
            lambda x: '✅ Memenuhi' if x >= 0 else '⚠️ Belum Memenuhi'
        )

        def warna_gap(val):
            return 'color:green;font-weight:bold' if val >= 0 else 'color:red;font-weight:bold'

        st.dataframe(
            df_det.style.applymap(warna_gap, subset=['Gap']),
            use_container_width=True, hide_index=True
        )

# ─────────────────────────────────────────────────────────────
# FOOTER  (tidak diubah)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<footer>
    🏥 Dashboard Klasterisasi Tenaga Kesehatan Jawa Barat &nbsp;|&nbsp;
    <b>Laura Amelia</b> — 22416255201146 &nbsp;|&nbsp;
    Teknik Informatika, UBP Karawang 2026 &nbsp;|&nbsp;
    Sumber Data: BPS Provinsi Jawa Barat 2024
</footer>
""", unsafe_allow_html=True)