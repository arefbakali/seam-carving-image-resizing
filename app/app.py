# app.py
import io
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st



def bytes_to_rgb_array(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)

def rgb_array_to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()

def to_grayscale(img: np.ndarray) -> np.ndarray:
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b

def compute_energy(gray: np.ndarray, is_horizontal: bool = False) -> np.ndarray:
    """Calcule l'énergie de l'image avec une méthode améliorée.
    Pour les seams horizontaux, on utilise une énergie qui préserve mieux les détails verticaux."""
    padded = np.pad(gray, 1, mode="edge")
    
    # Gradients de base (méthode Sobel améliorée)
    # Gradient horizontal (détecte les transitions verticales)
    dx = padded[1:-1, 2:] - padded[1:-1, :-2]
    # Gradient vertical (détecte les transitions horizontales)  
    dy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    
    # Gradients diagonaux pour plus de robustesse
    dx_diag1 = padded[2:, 2:] - padded[:-2, :-2]
    dx_diag2 = padded[2:, :-2] - padded[:-2, 2:]
    
    if is_horizontal:
        # Pour retirer des lignes horizontales, on veut préserver les éléments verticaux
        # Les gradients verticaux (dy) indiquent les transitions horizontales importantes
        # Les gradients horizontaux (dx) indiquent les transitions verticales importantes
        
        # Énergie de base combinant tous les gradients
        energy_base = (np.abs(dx) + np.abs(dy) + 0.5 * (np.abs(dx_diag1) + np.abs(dx_diag2)))
        
        # Identifier les zones avec de forts gradients verticaux (éléments verticaux importants)
        vertical_strength = np.abs(dy)
        
        # Créer une carte d'importance pour les zones verticales
        # Les zones avec de forts gradients verticaux doivent être protégées
        vertical_importance = vertical_strength / (vertical_strength.max() + 1e-9)
        
        # Augmenter l'énergie dans les zones verticales importantes pour les protéger
        # Multiplier par un facteur pour bien les protéger
        protected_energy = energy_base * (1.0 + 3.0 * vertical_importance)
        
        return protected_energy
    else:
        # Pour les seams verticaux, énergie standard améliorée
        return np.abs(dx) + np.abs(dy) + 0.3 * (np.abs(dx_diag1) + np.abs(dx_diag2))

def cumulative_energy_map_fast(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = E.shape
    M = E.astype(np.float32).copy()
    parent = np.full((H, W), -1, dtype=np.int32)

    for y in range(1, H):
        prev = M[y - 1]
        left = np.roll(prev, 1)
        right = np.roll(prev, -1)
        left[0] = np.inf
        right[-1] = np.inf

        stacked = np.vstack([left, prev, right])
        argmin = np.argmin(stacked, axis=0)
        best = stacked[argmin, np.arange(W)]

        M[y] += best
        parent[y] = np.arange(W) + (argmin - 1)

    return M, parent

def find_vertical_seam_fast(E: np.ndarray) -> np.ndarray:
    M, parent = cumulative_energy_map_fast(E)
    H, _ = E.shape
    seam = np.zeros(H, dtype=np.int32)
    seam[H - 1] = int(np.argmin(M[H - 1]))
    for y in range(H - 2, -1, -1):
        seam[y] = parent[y + 1, seam[y + 1]]
    return seam

def cumulative_energy_map_horizontal(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative energy map pour seams horizontaux (de gauche à droite)"""
    H, W = E.shape
    M = E.astype(np.float32).copy()
    parent = np.full((H, W), -1, dtype=np.int32)

    # Pour chaque colonne x (de gauche à droite)
    for x in range(1, W):
        prev = M[:, x - 1]  # Énergies cumulatives de la colonne précédente
        
        # Créer les 3 options pour chaque ligne y
        # Option 1: monter (y-1), Option 2: même ligne (y), Option 3: descendre (y+1)
        up = np.pad(prev[:-1], (1, 0), constant_values=np.inf)  # Monter: décaler vers le bas
        same = prev  # Même ligne
        down = np.pad(prev[1:], (0, 1), constant_values=np.inf)  # Descendre: décaler vers le haut
        
        # Empiler les 3 options
        stacked = np.vstack([up, same, down])
        
        # Trouver le minimum pour chaque ligne
        argmin = np.argmin(stacked, axis=0)
        best = stacked[argmin, np.arange(H)]
        
        # Mettre à jour l'énergie cumulative
        M[:, x] += best
        
        # Mettre à jour le parent (argmin: 0=up, 1=same, 2=down)
        parent[:, x] = np.arange(H) + (argmin - 1)

    return M, parent

def find_horizontal_seam_fast(E: np.ndarray) -> np.ndarray:
    """Trouve un seam horizontal (de gauche à droite)"""
    # E est H x W, on veut un seam de longueur W
    # Le seam donne seam[x] = y pour chaque colonne x
    M, parent = cumulative_energy_map_horizontal(E)
    H, W = E.shape
    seam = np.zeros(W, dtype=np.int32)
    
    # Trouver le y minimal dans la dernière colonne
    seam[W - 1] = int(np.argmin(M[:, W - 1]))
    
    # Remonter le chemin
    for x in range(W - 2, -1, -1):
        seam[x] = parent[seam[x + 1], x + 1]
    
    return seam

def remove_vertical_seam_fast(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    mask = np.ones((H, W), dtype=bool)
    mask[np.arange(H), seam] = False
    return img[mask].reshape(H, W - 1, C)

def remove_horizontal_seam_fast(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    mask = np.ones((H, W), dtype=bool)
    mask[seam, np.arange(W)] = False
    return img[mask].reshape(H - 1, W, C)

def seam_overlay_rgb(img: np.ndarray, seam: np.ndarray, is_vertical: bool = True) -> np.ndarray:
    out = img.copy()
    if is_vertical:
        out[np.arange(out.shape[0]), seam] = np.array([255, 0, 0], dtype=np.uint8)
    else:
        out[seam, np.arange(out.shape[1])] = np.array([255, 0, 0], dtype=np.uint8)
    return out

def seams_overlay_rgb(img: np.ndarray, seams: list, is_vertical: bool = True) -> np.ndarray:
    """Overlay multiple seams on image"""
    out = img.copy()
    for seam in seams:
        if is_vertical:
            out[np.arange(out.shape[0]), seam] = np.array([255, 0, 0], dtype=np.uint8)
        else:
            out[seam, np.arange(out.shape[1])] = np.array([255, 0, 0], dtype=np.uint8)
    return out

def energy_to_uint8(E: np.ndarray) -> np.ndarray:
    E_norm = E - E.min()
    denom = E_norm.max() + 1e-9
    return (255.0 * (E_norm / denom)).astype(np.uint8)

def seam_carve_width(img: np.ndarray, new_width: int, progress_cb=None, store_seams=False, store_frames=False) -> tuple:
    H, W, _ = img.shape
    if new_width >= W:
        raise ValueError(f"new_width must be < current width (current W={W}).")

    carved = img.copy()
    seams_to_remove = W - new_width
    seams_list = []
    frames = []

    if store_frames:
        frames.append(img.copy())  # Commencer par l'image originale
    
    for i in range(seams_to_remove):
        gray = to_grayscale(carved)
        E = compute_energy(gray)
        seam = find_vertical_seam_fast(E)
        if store_seams:
            # Store seam in original image coordinates
            seams_list.append(seam.copy())
        carved = remove_vertical_seam_fast(carved, seam)
        if store_frames:
            frames.append(carved.copy())  # Stocker après retrait du seam
        if progress_cb:
            progress_cb(i + 1, seams_to_remove)
    
    if store_seams or store_frames:
        return carved, seams_list, frames
    return carved

def seam_carve_height(img: np.ndarray, new_height: int, progress_cb=None, store_seams=False, store_frames=False) -> tuple:
    H, W, _ = img.shape
    if new_height >= H:
        raise ValueError(f"new_height must be < current height (current H={H}).")

    carved = img.copy()
    seams_to_remove = H - new_height
    seams_list = []
    frames = []

    if store_frames:
        frames.append(img.copy())  # Commencer par l'image originale
    
    for i in range(seams_to_remove):
        gray = to_grayscale(carved)
        E = compute_energy(gray, is_horizontal=True)  # Utiliser l'énergie améliorée pour seams horizontaux
        seam = find_horizontal_seam_fast(E)
        if store_seams:
            seams_list.append(seam.copy())
        carved = remove_horizontal_seam_fast(carved, seam)
        if store_frames:
            frames.append(carved.copy())  # Stocker après retrait du seam
        if progress_cb:
            progress_cb(i + 1, seams_to_remove)
    
    if store_seams or store_frames:
        return (carved, seams_list, frames)
    return carved


# =========================
# Streamlit UI + Advanced CSS
# =========================

st.set_page_config(page_title="Seam Carving APP", layout="wide")

st.markdown("""
<style>
/* ---------- Global layout ---------- */
:root{
  --bg1: #0a0e27;
  --bg2: #141b2d;
  --card: rgba(255,255,255,0.08);
  --card2: rgba(255,255,255,0.12);
  --stroke: rgba(255,255,255,0.15);
  --stroke2: rgba(255,255,255,0.25);
  --text: rgba(255,255,255,1);
  --muted: rgba(255,255,255,1);
  --muted2: rgba(255,255,255,1);
  --accent: #6366f1;        /* indigo moderne */
  --accent2: #8b5cf6;       /* violet moderne */
  --accent3: #ec4899;       /* rose moderne */
  --success: #10b981;       /* vert émeraude */
  --warn: #f59e0b;          /* amber */
  --danger: #ef4444;        /* red */
  --shadow: 0 20px 60px rgba(0,0,0,0.5);
  --shadow2: 0 15px 40px rgba(0,0,0,0.4);
  --radius: 24px;
  --radius2: 18px;
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1400px 700px at 15% -5%, rgba(99,102,241,0.4), transparent 65%),
    radial-gradient(1200px 650px at 105% 5%, rgba(139,92,246,0.35), transparent 60%),
    radial-gradient(1000px 750px at 50% 115%, rgba(236,72,153,0.25), transparent 60%),
    linear-gradient(180deg, var(--bg1), var(--bg2)) !important;
  color: var(--text) !important;
}

/* reduce default top padding */
.block-container{
  padding-top: 1.2rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1250px;
}

/* ---------- Header ---------- */
.app-hero{
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  padding: 28px 28px;
  background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
  margin-bottom: 20px;
}
.app-hero:before{
  content:"";
  position:absolute;
  inset:-2px;
  background:
    radial-gradient(600px 250px at 20% 0%, rgba(99,102,241,0.35), transparent 65%),
    radial-gradient(550px 280px at 85% 20%, rgba(139,92,246,0.28), transparent 65%);
  filter: blur(12px);
  opacity: 0.95;
  z-index:0;
}
.app-hero *{ position: relative; z-index:1; }
.hero-title{
  font-size: 38px;
  font-weight: 800;
  letter-spacing: -0.03em;
  margin: 0;
  line-height: 1.05;
}
.hero-sub{
  margin-top: 6px;
  color: rgba(255,255,255,1) !important;
  font-size: 14.5px;
  font-weight: 600;
  line-height: 1.55;
}
.badges{
  margin-top: 14px;
  display:flex;
  gap:10px;
  flex-wrap:wrap;
}
.badge{
  border: 1px solid var(--stroke2);
  background: rgba(255,255,255,0.06);
  padding: 7px 10px;
  border-radius: 999px;
  font-size: 12.5px;
  color: rgba(255,255,255,1) !important;
  font-weight: 700;
  backdrop-filter: blur(10px);
}

/* ---------- Cards ---------- */
.card{
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.08);
  border-radius: var(--radius);
  padding: 20px 20px;
  box-shadow: var(--shadow2);
  backdrop-filter: blur(10px);
}
.card-title{
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 10px;
  color: var(--text);
}
.card-note{
  margin-top: 8px;
  font-size: 12.5px;
  color: rgba(255,255,255,1) !important;
  font-weight: 600;
  line-height: 1.45;
}
.metric-row{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top: 10px;
}
.metric{
  flex: 1 1 160px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.045);
  border-radius: 16px;
  padding: 12px 12px;
}
.metric .k{
  font-size: 12px;
  color: rgba(255,255,255,1) !important;
  font-weight: 700;
}
.metric .v{
  font-size: 20px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: rgba(255,255,255,1) !important;
}

/* ---------- Streamlit elements ---------- */
[data-testid="stFileUploader"]{
  border: 2px dashed rgba(255,255,255,0.4) !important;
  border-radius: var(--radius);
  padding: 20px 20px;
  background: rgba(255,255,255,0.08) !important;
  backdrop-filter: blur(10px);
}
[data-testid="stFileUploader"] section{
  background: transparent !important;
}
[data-testid="stFileUploader"] p{
  color: rgba(255,255,255,1) !important;
  font-weight: 700 !important;
  font-size: 15px !important;
}
[data-testid="stFileUploader"] small{
  color: rgba(255,255,255,1) !important;
  font-weight: 700 !important;
}
[data-testid="stFileUploader"] button{
  background: rgba(255,255,255,0.15) !important;
  color: rgba(255,255,255,1) !important;
  font-weight: 700 !important;
  border: 1px solid rgba(255,255,255,0.3) !important;
  border-radius: 12px !important;
  padding: 8px 16px !important;
}
[data-testid="stFileUploader"] button:hover{
  background: rgba(255,255,255,0.25) !important;
  border-color: rgba(255,255,255,0.5) !important;
}
[data-testid="stSlider"]{
  padding: 6px 8px;
  border-radius: 14px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
}
.stButton>button{
  width: 100%;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.2);
  padding: 0.85rem 1.2rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  box-shadow: 0 16px 40px rgba(99,102,241,0.4);
  transition: transform .1s ease, filter .1s ease, box-shadow .1s ease;
}
.stButton>button:hover{
  transform: translateY(-2px);
  filter: brightness(1.1);
  box-shadow: 0 20px 50px rgba(99,102,241,0.5);
}
.stButton>button:active{
  transform: translateY(0px) scale(0.98);
}
div[data-testid="stDownloadButton"] button{
  background: linear-gradient(135deg, var(--accent2), var(--accent3)) !important;
  box-shadow: 0 16px 40px rgba(139,92,246,0.4);
}

/* progress bar */
[data-testid="stProgress"] > div > div{
  border-radius: 999px;
}

/* images look like cards */
[data-testid="stImage"] img{
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 18px 45px rgba(0,0,0,0.35);
}

/* expander */
details{
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 8px 10px;
}

/* Force all text to white */
p, span, div, label, small, caption, h1, h2, h3, h4, h5, h6{
  color: rgba(255,255,255,1) !important;
}

/* Streamlit specific elements */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span{
  color: rgba(255,255,255,1) !important;
  font-weight: 600 !important;
}

/* Labels for inputs */
label, [data-baseweb="form-control"] label{
  color: rgba(255,255,255,1) !important;
  font-weight: 700 !important;
}

/* Radio buttons and toggles */
[data-testid="stRadio"] label,
[data-testid="stToggle"] label{
  color: rgba(255,255,255,1) !important;
  font-weight: 700 !important;
}

/* Slider labels */
[data-testid="stSlider"] label{
  color: rgba(255,255,255,1) !important;
  font-weight: 700 !important;
}

/* Image captions */
[data-testid="stImage"] + div,
[data-testid="stImage"] caption{
  color: rgba(255,255,255,1) !important;
  font-weight: 600 !important;
}

/* Info, warning, error, success messages */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
.stAlert p,
.stAlert div{
  color: rgba(255,255,255,1) !important;
  font-weight: 600 !important;
}

/* Expander text */
[data-testid="stExpander"] p,
[data-testid="stExpander"] div,
[data-testid="stExpander"] label{
  color: rgba(255,255,255,1) !important;
  font-weight: 600 !important;
}

/* Status messages */
[data-testid="stStatus"] p,
[data-testid="stStatus"] div{
  color: rgba(255,255,255,1) !important;
  font-weight: 600 !important;
}

/* Progress text */
[data-testid="stProgress"] + div{
  color: rgba(255,255,255,1) !important;
  font-weight: 700 !important;
}

/* All Streamlit text elements */
.stText, .stMarkdown, .stCaption{
  color: rgba(255,255,255,1) !important;
}

/* hide streamlit default menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-hero">
  <div class="hero-title"> Seam Carving APP </div>
  <div class="hero-sub">
   Content-aware image resizing works by iteratively removing seams with the lowest energy, allowing the image to be resized while preserving visually significant regions.
The user can specify the exact number of seams to remove, either along the vertical axis (width reduction) or along the horizontal axis (height reduction).
  </div>

</div>
""", unsafe_allow_html=True)

# Main area uploader (NOT sidebar)
uploader_col, settings_col = st.columns([1.6, 1.0], vertical_alignment="top")

with uploader_col:
    st.markdown('<div class="card"><div class="card-title">1) Upload image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Browse a JPG/PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('<div class="card-note">Astuce: si le traitement est lent, active “Speed mode” et limite la largeur.</div></div>', unsafe_allow_html=True)

with settings_col:
    st.markdown('<div class="card"><div class="card-title">2) Options</div>', unsafe_allow_html=True)
    speed_mode = st.toggle("Speed mode (auto downscale)", value=True)
    max_dim = st.slider("Max dimension (speed mode)", 400, 1800, 1100, 50, disabled=not speed_mode)
    show_debug = st.toggle("Show debug (energy maps)", value=True)
    create_animation = st.toggle("Create animation GIF", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Charger une image d'exemple si disponible et aucune image n'est uploadée
if not uploaded:
    import os
    example_files = ["giraffe.jpg", "original.jpg"]
    example_file = None
    for f in example_files:
        if os.path.exists(f):
            example_file = f
            break
    
    if example_file:
        st.info(f"📸 Image d'exemple chargée automatiquement : **{example_file}** (vous pouvez aussi uploader votre propre image)")
        with open(example_file, "rb") as f:
            orig = bytes_to_rgb_array(f.read())
    else:
        st.info("👆 Veuillez uploader une image JPG ou PNG pour commencer.")
        st.stop()
else:
    orig = bytes_to_rgb_array(uploaded.read())

# optional downscale for speed
if speed_mode:
    H_orig, W_orig = orig.shape[:2]
    max_dim_current = max(H_orig, W_orig)
    if max_dim_current > max_dim:
        pil = Image.fromarray(orig)
        if W_orig > H_orig:
            new_w = max_dim
            new_h = int(H_orig * (max_dim / W_orig))
        else:
            new_h = max_dim
            new_w = int(W_orig * (max_dim / H_orig))
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        orig = np.array(pil, dtype=np.uint8)
        st.warning(f"Speed mode: image downscaled to {orig.shape[1]}×{orig.shape[0]} for faster processing.")

H, W, _ = orig.shape

st.markdown("")

controls, preview = st.columns([1.0, 1.8], vertical_alignment="top")

with controls:
    st.markdown('<div class="card"><div class="card-title">3) Configuration</div>', unsafe_allow_html=True)

    # Choix du mode (horizontal ou vertical)
    carving_mode = st.radio("Direction", ["Vertical (réduire largeur)", "Horizontal (réduire hauteur)"], horizontal=True)
    is_vertical = carving_mode.startswith("Vertical")
    
    # Choix du nombre de seams
    if is_vertical:
        max_seams = W - 1
        default_seams = min(50, max_seams // 2)
        seams_to_remove = st.slider("Nombre de seams à retirer", 1, max_seams, default_seams, 1)
        target_width = W - seams_to_remove
        target_height = H
    else:
        max_seams = H - 1
        default_seams = min(50, max_seams // 2)
        seams_to_remove = st.slider("Nombre de seams à retirer", 1, max_seams, default_seams, 1)
        target_height = H - seams_to_remove
        target_width = W

    st.markdown(f"<div class='card-note'>Retirer <b>{seams_to_remove}</b> seams {'verticales' if is_vertical else 'horizontales'}</div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    if is_vertical:
        st.markdown(f"""
          <div class="metric"><div class="k">Original</div><div class="v">{W}×{H}</div></div>
          <div class="metric"><div class="k">Résultat</div><div class="v">{target_width}×{H}</div></div>
          <div class="metric"><div class="k">Seams</div><div class="v">{seams_to_remove}</div></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
          <div class="metric"><div class="k">Original</div><div class="v">{W}×{H}</div></div>
          <div class="metric"><div class="k">Résultat</div><div class="v">{W}×{target_height}</div></div>
          <div class="metric"><div class="k">Seams</div><div class="v">{seams_to_remove}</div></div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("Lancer Seam Carving", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

with preview:
    st.markdown('<div class="card"><div class="card-title">Preview</div>', unsafe_allow_html=True)
    st.image(orig, caption=f"Original ({W}×{H})", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not run:
    st.stop()

# Initial energy map
gray0 = to_grayscale(orig)
E0 = compute_energy(gray0, is_horizontal=(not is_vertical))
initial_energy_map = energy_to_uint8(E0)

# Run with progress
progress = st.progress(0)
status = st.empty()

t0 = time.time()

def progress_cb(done, total):
    p = 100 if total <= 0 else int((done / total) * 100)
    progress.progress(p)
    status.write(f"Traitement: {done}/{total} ({p}%)")

try:
    if is_vertical:
        result = seam_carve_width(orig, target_width, progress_cb=progress_cb, 
                                   store_seams=True, store_frames=create_animation)
        if create_animation:
            carved, seams_list, frames = result
        else:
            carved, seams_list, _ = result
            frames = None
    else:
        result = seam_carve_height(orig, target_height, progress_cb=progress_cb,
                                    store_seams=True, store_frames=create_animation)
        if create_animation:
            carved, seams_list, frames = result
        else:
            carved, seams_list, _ = result
            frames = None
except Exception as e:
    st.error(f"Erreur: {e}")
    st.stop()

dt = time.time() - t0
status.success(f"Terminé  ({dt:.2f}s)")

st.markdown("")

# Afficher les seams sur l'image originale
if seams_list and len(seams_list) > 0:
    st.markdown('<div class="card"><div class="card-title">🔴 Visualisation des Seams</div>', unsafe_allow_html=True)
    # Calculer les premiers seams directement sur l'image originale pour visualisation
    num_seams_to_show = min(len(seams_list), 30)  # Limiter pour performance
    seams_overlay = orig.copy()
    temp_img = orig.copy()
    
    # Calculer et afficher les premiers seams
    for i in range(num_seams_to_show):
        gray = to_grayscale(temp_img)
        E = compute_energy(gray)
        if is_vertical:
            seam = find_vertical_seam_fast(E)
            # Afficher le seam directement (les coordonnées sont correctes pour l'image actuelle)
            seams_overlay = seam_overlay_rgb(seams_overlay, seam, is_vertical=True)
            temp_img = remove_vertical_seam_fast(temp_img, seam)
        else:
            seam = find_horizontal_seam_fast(E)
            seams_overlay = seam_overlay_rgb(seams_overlay, seam, is_vertical=False)
            temp_img = remove_horizontal_seam_fast(temp_img, seam)
    
    st.image(seams_overlay, caption=f"Visualisation des {len(seams_list)} seams à retirer (rouge - premiers {num_seams_to_show} calculés)", use_container_width=True)
    st.markdown(f'<div class="card-note">Note: {len(seams_list)} seams {"verticales" if is_vertical else "horizontales"} seront retirées</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")

# Energy maps
if show_debug:
    st.markdown('<div class="card"><div class="card-title">⚡ Energy Maps</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.image(initial_energy_map, caption="Energy map initiale", use_container_width=True)
    with e2:
        # Energy map finale
        gray_final = to_grayscale(carved)
        E_final = compute_energy(gray_final)
        final_energy_map = energy_to_uint8(E_final)
        st.image(final_energy_map, caption="Energy map finale", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")

# Before / After
st.markdown('<div class="card"><div class="card-title">✅ Avant / Après</div>', unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    st.image(orig, caption=f"Avant ({W}×{H})", use_container_width=True)
with b2:
    st.image(carved, caption=f"Après ({carved.shape[1]}×{carved.shape[0]})", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Animation GIF
if create_animation and frames and len(frames) > 0:
    st.markdown("")
    st.markdown('<div class="card"><div class="card-title">🎬 Animation de la transformation</div>', unsafe_allow_html=True)
    
    # Créer le GIF
    gif_frames = []
    # Prendre un échantillon des frames pour le GIF (max 40 frames pour performance)
    step = max(1, len(frames) // 40)
    sampled_frames = frames[::step]
    if len(sampled_frames) < len(frames) or sampled_frames[-1] is not frames[-1]:
        sampled_frames.append(frames[-1])
    
    # Taille finale pour redimensionner toutes les frames
    final_h, final_w = carved.shape[:2]
    
    for frame in sampled_frames:
        pil_frame = Image.fromarray(frame)
        # Redimensionner à la taille finale pour montrer la réduction progressive
        pil_frame_resized = pil_frame.resize((final_w, final_h), Image.Resampling.LANCZOS)
        gif_frames.append(pil_frame_resized)
    
    # Créer le GIF en mémoire
    gif_buffer = io.BytesIO()
    if len(gif_frames) > 1:
        gif_frames[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=gif_frames[1:],
            duration=80,  # 80ms par frame pour animation plus fluide
            loop=0
        )
        gif_buffer.seek(0)
        gif_bytes = gif_buffer.getvalue()
        
        st.image(gif_bytes, caption="Animation GIF montrant la réduction progressive de l'image", use_container_width=True)
        
        st.download_button(
            "Télécharger l'animation GIF",
            data=gif_bytes,
            file_name="seam_carving_animation.gif",
            mime="image/gif",
            use_container_width=True
        )
    else:
        st.info("Animation non disponible (trop peu de frames)")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

st.markdown('<div class="card"><div class="card-title">⬇️ Télécharger</div>', unsafe_allow_html=True)
png_bytes = rgb_array_to_png_bytes(carved)
st.download_button(
    " Télécharger le résultat (PNG)",
    data=png_bytes,
    file_name="seam_carved.png",
    mime="image/png",
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)