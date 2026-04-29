import os
import json
import kagglehub
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Waveline", page_icon="🎵", layout="wide", initial_sidebar_state="collapsed")

VIBE_COLORS = {
    "High Energy":    "#f04060",
    "Chill":          "#40a0f0",
    "Acoustic":       "#60c060",
    "Dance":          "#f0a020",
    "Melancholic":    "#a060f0",
    "Upbeat Pop":     "#f06090",
    "Instrumental":   "#40c0b0",
    "Dark & Intense": "#8040a0",
}

TIER_COLORS = ["#ffd700","#c0c0c0","#cd7f32","#7c5cfc","#6b44f0","#5a3dcc","#4a2eaa","#3d25a0","#2a1880","#1e1060"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body { background: #000 !important; }
.main, .block-container,
[data-testid="stAppViewContainer"],[data-testid="stHeader"],
[data-testid="stToolbar"],[data-testid="stBottom"],
.stApp { background: #000 !important; background-color: #000 !important; }
.block-container { padding: 1.8rem 3rem 4rem !important; max-width: 1240px !important; }
html, body, [class*="css"], .stMarkdown, p, div, span, label {
  font-family: 'DM Sans', sans-serif !important; color: #c0c0d8 !important;
}
#MainMenu, footer, header,
[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* INPUTS */
.stTextInput > div > div > input {
  background: #0a0a14 !important; border: 1px solid #1e1e32 !important;
  border-radius: 10px !important; color: #d0d0e8 !important;
  font-size: 0.9rem !important; padding: 0.7rem 1rem !important; caret-color: #7c5cfc !important;
}
.stTextInput > div > div > input:focus { border-color: #7c5cfc !important; box-shadow: 0 0 0 3px #7c5cfc18 !important; outline: none !important; }
.stTextInput > div > div > input::placeholder { color: #252540 !important; }
.stTextInput label { display: none !important; }

/* BUTTONS */
.stButton > button {
  background: linear-gradient(135deg, #5c36d4, #8a58f0) !important;
  border: none !important; border-radius: 10px !important; color: #fff !important;
  font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
  font-size: 0.9rem !important; padding: 0.6rem 1.2rem !important;
  width: 100% !important; transition: opacity 0.18s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* MAIN NAV TABS */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border: none !important;
  border-bottom: 1px solid #0e0e1e !important; border-radius: 0 !important;
  padding: 0 !important; gap: 0 !important;
  display: flex !important; justify-content: space-between !important; width: 100% !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border: none !important; border-radius: 0 !important;
  color: #4a3a80 !important; font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 0.88rem !important; padding: 0.6rem 0 !important;
  letter-spacing: 0.04em; flex: 1 !important; text-align: center !important;
  border-bottom: 2px solid transparent !important; transition: color 0.18s, border-color 0.18s !important;
}
.stTabs [aria-selected="true"] { background: transparent !important; color: #9b7dfd !important; border-bottom: 2px solid #7c5cfc !important; }
.stTabs [data-baseweb="tab"]:hover { color: #7c5cfc !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 1.8rem !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* INNER TABS */
.inner-tabs .stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border: none !important;
  border-bottom: none !important; gap: 6px !important;
  justify-content: flex-start !important; padding: 0 !important; margin-bottom: 1rem !important;
}
.inner-tabs .stTabs [data-baseweb="tab"] {
  background: #0e0e1e !important; border: 1px solid #1e1e34 !important;
  border-radius: 999px !important; color: #4a4a70 !important; font-size: 0.82rem !important;
  font-weight: 500 !important; padding: 6px 20px !important; flex: 0 !important;
  border-bottom: 1px solid #1e1e34 !important; letter-spacing: 0.01em !important;
}
.inner-tabs .stTabs [aria-selected="true"] {
  background: #fff !important; color: #000 !important;
  border-color: #fff !important; border-bottom: 1px solid #fff !important;
}

/* SELECTBOX */
[data-baseweb="select"] > div { background: #0a0a14 !important; border: 1px solid #1e1e32 !important; border-radius: 10px !important; color: #c0c0d8 !important; }
[data-baseweb="popover"] { background: #0a0a14 !important; border: 1px solid #1e1e32 !important; }
[data-baseweb="menu"] { background: #0a0a14 !important; }
[role="option"] { background: #0a0a14 !important; color: #a0a0c0 !important; }
[role="option"]:hover { background: #14142a !important; }

/* STAT CARDS */
.stat-card { background: #080810; border: 1px solid #121220; border-radius: 14px; padding: 1.3rem 1.2rem; text-align: center; }
.stat-val { font-family: 'Syne', sans-serif !important; font-size: 2rem; font-weight: 700; color: #e8e8f8 !important; line-height: 1; }
.stat-label { font-size: 0.7rem; color: #282840 !important; margin-top: 7px; text-transform: uppercase; letter-spacing: 0.1em; }

/* SONG ROWS */
.song-row { display: flex; align-items: center; gap: 14px; background: #070710; border: 1px solid #10101e; border-radius: 11px; padding: 0.75rem 1rem; margin-bottom: 6px; }
.song-row:hover { border-color: #2a1e60; background: #0a0a18; }
.song-num { font-family: 'Syne', sans-serif !important; font-size: 0.78rem; font-weight: 700; color: #1e1e38 !important; min-width: 22px; text-align: center; cursor: default; user-select: none; transition: all 0.15s; }
.song-num.playing { color: #7c5cfc !important; font-size: 1rem; }
.song-info { flex: 1; min-width: 0; }
.song-name { font-weight: 500 !important; font-size: 0.86rem !important; color: #c0c0de !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.song-name:hover { color: #a080ff !important; }
.song-name.has-preview { cursor: pointer; }
.song-name.no-preview { cursor: default; }
.song-artist { font-size: 0.73rem !important; color: #282840 !important; margin-top: 2px; }
.song-badge { background: #0a0a1a; border: 1px solid #181830; border-radius: 5px; padding: 2px 8px; font-size: 0.7rem !important; color: #4a3a80 !important; white-space: nowrap; flex-shrink: 0; }
.song-plays { font-size: 0.73rem !important; color: #1a1a30 !important; min-width: 50px; text-align: right; }
.score-bar-bg { height: 3px; border-radius: 2px; background: #0e0e1e; margin-top: 5px; }
.score-bar-fill { height: 3px; border-radius: 2px; background: linear-gradient(90deg, #5c36d4, #8a58f0); }

/* VIBE BADGES */
.vibe-badge { display: inline-block; border-radius: 6px; padding: 2px 10px; font-size: 0.68rem !important; font-weight: 600; letter-spacing: 0.04em; white-space: nowrap; flex-shrink: 0; border: 1px solid transparent; }

/* VIBE CARDS - clickable */
.vibe-card-btn { border-radius: 11px; padding: 0.65rem 1rem; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; transition: opacity 0.18s, transform 0.1s; border: 1px solid transparent; }
.vibe-card-btn:hover { opacity: 0.8; transform: translateX(3px); }
.vibe-card-name { font-family: 'Syne', sans-serif !important; font-size: 0.82rem; font-weight: 700; }

/* SEASON CARDS - clickable, no bar */
.season-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
.season-card { border-radius: 14px; padding: 1rem 1.2rem; flex: 1; min-width: 130px; border: 1px solid transparent; cursor: pointer; transition: opacity 0.18s, transform 0.1s; }
.season-card:hover { opacity: 0.85; transform: translateY(-2px); }
.season-card-name { font-family: 'Syne', sans-serif !important; font-size: 0.9rem; font-weight: 700; }
.season-card-count { font-size: 0.7rem !important; opacity: 0.6; margin-top: 3px; text-transform: uppercase; letter-spacing: 0.08em; }

/* GENRE PILLS - clickable multi-select */
.genre-pill { display: inline-block; background: #0e0e1e; border: 1px solid #1e1e34; border-radius: 999px; padding: 6px 16px; font-size: 0.82rem !important; color: #5a5a80 !important; margin: 4px; cursor: pointer; transition: all 0.18s; user-select: none; }
.genre-pill:hover { border-color: #7c5cfc; color: #9080d0 !important; }
.genre-pill.active { background: #7c5cfc !important; border-color: #7c5cfc !important; color: #fff !important; }

/* AI SEARCH */
.ai-search-wrap { background: #080812; border: 1px solid #1a1a2e; border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1.4rem; }
.ai-search-label { font-size: 0.7rem; color: #3a3a5a; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px; }

/* MORE BUTTON */
.more-btn-wrap { display: flex; justify-content: center; margin-top: 1rem; }
.more-btn-wrap .stButton > button { width: auto !important; padding: 8px 32px !important; background: transparent !important; border: 1px solid #2a2a44 !important; color: #7070a0 !important; font-size: 0.85rem !important; border-radius: 999px !important; }
.more-btn-wrap .stButton > button:hover { border-color: #7c5cfc !important; color: #a090d0 !important; }

/* TIER LEADERBOARD */
.tier-row { display: flex; align-items: center; gap: 14px; padding: 10px 14px; border-radius: 10px; margin-bottom: 5px; background: #070710; border: 1px solid #10101e; }
.tier-rank { font-family: 'Syne', sans-serif !important; font-size: 1.1rem; font-weight: 800; min-width: 30px; text-align: center; }
.tier-info { flex: 1; min-width: 0; }
.tier-name { font-size: 0.88rem; font-weight: 600; color: #d0d0e8; }
.tier-sub { font-size: 0.72rem; color: #282840; margin-top: 2px; }
.tier-bar-wrap { width: 160px; }
.tier-plays { font-size: 0.75rem; color: #3a3a5a; text-align: right; min-width: 70px; }

/* PROFILE */
.profile-avatar { width: 66px; height: 66px; border-radius: 50%; background: linear-gradient(135deg, #3a1e90, #6b44f0); display: flex; align-items: center; justify-content: center; font-family: 'Syne', sans-serif !important; font-size: 1.5rem; font-weight: 700; color: #fff !important; margin-bottom: 0.8rem; }
.vibe-card { border-radius: 12px; padding: 0.9rem 1.1rem; display: flex; flex-direction: column; gap: 4px; min-width: 110px; flex: 1; border: 1px solid transparent; }
.vibe-card-count-sm { font-size: 0.68rem !important; opacity: 0.55; text-transform: uppercase; letter-spacing: 0.08em; }
.vibe-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }

/* MISC */
.wl-logo { font-family: 'Syne', sans-serif !important; font-size: 1.45rem; font-weight: 800; letter-spacing: -0.04em; color: #e0e0f0 !important; margin-bottom: 1.4rem; }
.wl-logo span { color: #7c5cfc !important; }
.section-label { font-family: 'Syne', sans-serif !important; font-size: 0.62rem; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase; color: #28205a !important; margin-bottom: 4px; }
hr.wl-divider { border: none; border-top: 1px solid #0c0c1a; margin: 1.4rem 0; }
.stSpinner > div { border-top-color: #7c5cfc !important; }

/* SIGN-IN */
.si-card { background: #060610; border: 1px solid #14142a; border-radius: 22px; padding: 3rem 2.8rem 2.5rem; width: 460px; text-align: center; }
.si-pill { display: inline-block; background: #0e0a22; border: 1px solid #2a1e5a; color: #7c5cfc !important; font-size: 0.68rem !important; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; padding: 5px 16px; border-radius: 999px; margin-bottom: 1.4rem; }
.si-title { font-family: 'Syne', sans-serif !important; font-size: 2.8rem; font-weight: 800; color: #e8e8f8 !important; letter-spacing: -0.05em; line-height: 1; margin-bottom: 0.5rem; }
.si-title span { color: #7c5cfc !important; }
.si-sub { color: #282848 !important; font-size: 0.86rem; margin-bottom: 2rem; line-height: 1.55; }
audio { display: none; }
</style>

<script>
window._wl_audio = null; window._wl_playing_id = null; window._wl_timer = null;
function wlPlay(rowId, url) {
  var numEl = document.getElementById('num_' + rowId);
  if (window._wl_playing_id === rowId) {
    if (window._wl_audio) { window._wl_audio.pause(); window._wl_audio = null; }
    window._wl_playing_id = null;
    if (window._wl_timer) { clearTimeout(window._wl_timer); window._wl_timer = null; }
    if (numEl) { numEl.textContent = numEl.dataset.num; numEl.classList.remove('playing'); }
    return;
  }
  if (window._wl_audio) {
    window._wl_audio.pause();
    var prevEl = document.getElementById('num_' + window._wl_playing_id);
    if (prevEl) { prevEl.textContent = prevEl.dataset.num; prevEl.classList.remove('playing'); }
  }
  if (window._wl_timer) { clearTimeout(window._wl_timer); window._wl_timer = null; }
  if (!url || url === 'None' || url === '') return;
  var audio = new Audio(url);
  audio.crossOrigin = 'anonymous';
  window._wl_audio = audio; window._wl_playing_id = rowId;
  if (numEl) { numEl.textContent = '▶'; numEl.classList.add('playing'); }
  audio.play().catch(function(e){});
  audio.addEventListener('ended', function() {
    window._wl_audio = null; window._wl_playing_id = null;
    if (numEl) { numEl.textContent = numEl.dataset.num; numEl.classList.remove('playing'); }
  });
  window._wl_timer = setTimeout(function() {
    if (window._wl_playing_id === rowId && numEl) { numEl.textContent = numEl.dataset.num; numEl.classList.remove('playing'); }
  }, 3000);
}
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path    = kagglehub.dataset_download("undefinenull/million-song-dataset-spotify-lastfm")
    music   = pd.read_csv(os.path.join(path, "Music Info.csv"))
    history = pd.read_csv(os.path.join(path, "User Listening History.csv"))
    music   = music.sample(min(50000, len(music)), random_state=42)
    history = history.sample(min(200000, len(history)), random_state=42)
    return music, history


@st.cache_resource
def build_model(music, history):
    sd    = music.copy()
    feats = ["danceability","energy","valence","tempo","acousticness","instrumentalness","liveness","speechiness","loudness"]
    sd    = sd.dropna(subset=feats).drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    for col in ["name","artist","genre","tags","spotify_preview_url"]:
        sd[col] = sd[col].fillna("") if col in sd.columns else ""
    scaler        = StandardScaler()
    audio_scaled  = scaler.fit_transform(sd[feats])
    kmeans        = KMeans(n_clusters=8, random_state=42, n_init=10)
    sd["cluster"] = kmeans.fit_predict(audio_scaled)
    centers       = pd.DataFrame(kmeans.cluster_centers_, columns=feats)
    vibe_names    = _auto_label_clusters(centers)
    sd["vibe"]    = sd["cluster"].map(vibe_names)
    sd["text_features"] = sd["tags"] + " " + sd["genre"] + " " + sd["artist"]
    text_matrix   = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(sd["text_features"])
    pop           = history.groupby("track_id")["playcount"].sum().reset_index()
    sd            = sd.merge(pop, on="track_id", how="left")
    sd["playcount"]        = sd["playcount"].fillna(0)
    sd["popularity_score"] = MinMaxScaler().fit_transform(sd[["playcount"]])
    return sd, audio_scaled, text_matrix, vibe_names, feats


def _auto_label_clusters(centers):
    labels, used = {}, set()
    order = ["Acoustic","Instrumental","Dark & Intense","Dance","High Energy","Upbeat Pop","Melancholic","Chill"]
    for idx, row in centers.iterrows():
        e, d, v, ac, ins = row["energy"], row["danceability"], row["valence"], row["acousticness"], row["instrumentalness"]
        if   ac > 0.6  and e < 0.4  and "Acoustic"       not in used: label = "Acoustic"
        elif ins > 0.5              and "Instrumental"    not in used: label = "Instrumental"
        elif e > 0.8   and v < 0.35 and "Dark & Intense"  not in used: label = "Dark & Intense"
        elif e > 0.75  and d > 0.7  and "Dance"           not in used: label = "Dance"
        elif e > 0.7   and v > 0.6  and "High Energy"     not in used: label = "High Energy"
        elif v > 0.65  and d > 0.6  and "Upbeat Pop"      not in used: label = "Upbeat Pop"
        elif v < 0.35  and e < 0.5  and "Melancholic"     not in used: label = "Melancholic"
        elif e < 0.45              and "Chill"            not in used: label = "Chill"
        else:
            label = next((l for l in order if l not in used), f"Vibe {idx}")
        used.add(label); labels[idx] = label
    return labels


def get_vibe_color(vibe): return VIBE_COLORS.get(str(vibe), "#7c5cfc")


def hybrid_recommend(song_name, song_data, audio_scaled, text_matrix, limit=10, exclude_same_artist=True):
    matches = song_data[song_data["name"].str.lower().str.contains(song_name.lower(), na=False)]
    if matches.empty: return pd.DataFrame()
    idx    = matches.index[0]
    a      = cosine_similarity(audio_scaled[idx].reshape(1,-1), audio_scaled)[0]
    t      = cosine_similarity(text_matrix[idx], text_matrix)[0]
    cb     = (song_data["cluster"] == song_data.iloc[idx]["cluster"]).astype(int).values
    scores = 0.35*a + 0.45*t + 0.10*song_data["popularity_score"].values + 0.10*cb
    scores[idx] = -1
    res    = song_data.copy(); res["score"] = scores
    if exclude_same_artist:
        res = res[res["artist"] != song_data.iloc[idx]["artist"]]
    return res.sort_values("score", ascending=False).head(limit)


def recommend_by_user(user_id, history, song_data, audio_scaled, text_matrix, limit=50):
    uh = history[history["user_id"] == user_id]
    if uh.empty: return pd.DataFrame()
    top_tracks = uh.sort_values("playcount", ascending=False).head(5)["track_id"]
    all_recs   = []
    for tid in top_tracks:
        m = song_data[song_data["track_id"] == tid]
        if not m.empty:
            r = hybrid_recommend(m.iloc[0]["name"], song_data, audio_scaled, text_matrix, limit=limit)
            if not r.empty: all_recs.append(r)
    if not all_recs: return pd.DataFrame()
    final = pd.concat(all_recs)
    final = final[~final["track_id"].isin(uh["track_id"].tolist())]
    return final.drop_duplicates(subset=["track_id"]).sort_values("score", ascending=False).head(limit)


def recommend_by_genre_multi(genres, song_data, limit=50):
    """AND logic: songs must match ALL selected genres via genre or tags."""
    res = song_data.copy()
    for g in genres:
        g = g.strip().lower()
        mask = (
            song_data["genre"].str.lower().str.contains(g, na=False) |
            song_data["tags"].str.lower().str.contains(rf"\b{g}\b", na=False, regex=True)
        )
        res = res[mask]
    if res.empty: return pd.DataFrame()
    res["score"] = res["popularity_score"]
    return res.sort_values(["popularity_score","playcount"], ascending=False).head(limit)


def get_genre(row):
    if str(row.get("genre","")).strip(): return row["genre"]
    tags = str(row.get("tags","")).split(",")
    return tags[0].strip() if tags and tags[0].strip() else ""


def ai_playlist_from_query(query, user_id, history, song_data, audio_scaled, text_matrix):
    """
    Semantic keyword expansion: maps natural language → music tags,
    then personalizes using user listening history audio profile.
    """
    # ── Semantic expansion map ──
    # Keys are concepts the user might type; values are actual music tags to search
    SEMANTIC_MAP = {
        # Moods
        "sad": ["sad","melancholy","heartbreak","sorrow","lonely","cry","depression","grief","tears","blue"],
        "happy": ["happy","joy","upbeat","cheerful","positive","good vibes","fun","bright","optimistic","euphoric"],
        "angry": ["angry","rage","intense","aggressive","heavy","metal","hard","fury","dark"],
        "calm": ["calm","peaceful","serene","gentle","soft","quiet","mellow","tranquil","ambient"],
        "romantic": ["love","romance","romantic","intimate","passion","couples","heart","sweet","tender"],
        "melancholy": ["melancholy","sad","bittersweet","reflective","nostalgic","wistful","somber"],
        "nostalgic": ["nostalgic","throwback","classic","retro","memories","old school","vintage"],
        "hopeful": ["hopeful","uplifting","inspiring","positive","motivational","rise","overcome"],
        "lonely": ["lonely","alone","solitude","isolated","empty","quiet","missing"],
        "anxious": ["anxious","tense","nervous","uneasy","dark","atmospheric","suspense"],

        # Activities
        "workout": ["workout","gym","training","fitness","pump","run","energy","power","intense","sweat","cardio"],
        "running": ["run","running","jog","cardio","energy","fast","pump","athletic","sprint"],
        "study": ["study","focus","concentrate","ambient","instrumental","piano","background","calm","soft"],
        "sleep": ["sleep","lullaby","calm","peaceful","night","dream","rest","relax","soothe","quiet","soft"],
        "party": ["party","dance","club","dj","bass","banger","anthem","hype","festival","rave","bounce"],
        "driving": ["road","trip","drive","highway","journey","cruise","open","windows","fast","freedom"],
        "cooking": ["kitchen","warm","cozy","soul","feel good","comfort","home","groove"],
        "meditation": ["meditation","zen","ambient","peaceful","spiritual","mindful","breathe","calm","healing"],
        "reading": ["ambient","soft","instrumental","quiet","background","focus","piano","acoustic"],
        "hiking": ["adventure","nature","explore","free","outdoor","open","journey","discovery"],

        # Times/settings
        "morning": ["morning","sunrise","fresh","wake up","bright","start","coffee","new day","gentle"],
        "night": ["night","late","dark","moonlight","stars","midnight","dreamy","atmospheric","nocturnal"],
        "rainy": ["rain","rainy","grey","gloomy","cozy","indoor","melancholy","acoustic","soft"],
        "sunny": ["summer","sunny","bright","warm","cheerful","happy","beach","positive","feel good"],
        "winter": ["winter","cold","snow","cozy","fireplace","december","frost","christmas"],
        "summer": ["summer","beach","sun","hot","vacation","tropical","warm","festival","bbq"],
        "afternoon": ["afternoon","lazy","chill","mellow","relaxed","groove","soft","easy"],
        "friday": ["friday","weekend","party","hype","end of week","celebration","fun","dance"],

        # Genres/sounds
        "acoustic": ["acoustic","guitar","unplugged","folk","singer","songwriter","intimate","raw"],
        "electronic": ["electronic","synth","edm","techno","house","beats","digital","rave"],
        "jazz": ["jazz","blues","swing","saxophone","trumpet","improvise","smooth","soul"],
        "classical": ["classical","orchestra","piano","symphony","concerto","baroque","instrumental"],
        "hip hop": ["hip hop","rap","trap","beats","flow","rhyme","urban","street"],
        "rock": ["rock","guitar","drums","band","electric","riff","classic rock","alternative"],
        "pop": ["pop","catchy","mainstream","chart","hit","radio","hook","upbeat"],
        "folk": ["folk","acoustic","storytelling","indie","banjo","country","roots"],
        "soul": ["soul","r&b","groove","warm","vintage","smooth","funk","motown"],
        "chill": ["chill","relax","lo-fi","mellow","laid back","easy","smooth","vibes"],

        # Compound/descriptive
        "rainy afternoon": ["sad","rain","melancholy","acoustic","grey","soft","cozy","quiet","reflective"],
        "late night": ["night","late","dark","atmospheric","dreamy","mellow","ambient","deep"],
        "good vibes": ["happy","positive","feel good","upbeat","summer","dance","cheerful","fun"],
        "heartbreak": ["heartbreak","sad","lonely","miss","lost","cry","breakup","sorrow","tears"],
        "pump up": ["energy","pump","workout","intense","power","motivate","hype","strong","bass"],
        "feel good": ["happy","upbeat","positive","fun","joy","cheerful","summer","dance","bright"],
        "chill out": ["chill","relax","mellow","soft","calm","ambient","lo-fi","easy","laid back"],
        "road trip": ["road","trip","drive","highway","journey","freedom","adventure","cruise","travel"],
        "sunday morning": ["gentle","soft","acoustic","warm","cozy","peaceful","quiet","morning","calm"],
        "focus": ["focus","study","ambient","instrumental","concentrate","background","quiet","flow"],
    }

    q_lower = query.lower().strip()

    # Collect tags: first try compound phrases, then individual words
    expanded_tags = set()

    # Try full query
    if q_lower in SEMANTIC_MAP:
        expanded_tags.update(SEMANTIC_MAP[q_lower])

    # Try 2-word combos
    words = q_lower.split()
    for i in range(len(words)-1):
        combo = f"{words[i]} {words[i+1]}"
        if combo in SEMANTIC_MAP:
            expanded_tags.update(SEMANTIC_MAP[combo])

    # Try individual words
    for w in words:
        if w in SEMANTIC_MAP:
            expanded_tags.update(SEMANTIC_MAP[w])
        else:
            # Direct partial match — word itself as a tag
            expanded_tags.add(w)

    # If still empty, use words directly
    if not expanded_tags:
        expanded_tags = set(words)

    tags  = list(expanded_tags)
    title = query.title()
    mood  = f"Songs for: {query}"

    # Score songs by tag matches (partial string match, more flexible)
    def tag_score(row):
        combined = (str(row["tags"]) + " " + str(row["genre"])).lower()
        return sum(1 for t in tags if t in combined)

    scores_arr = [tag_score(row) for _, row in song_data.iterrows()]
    song_data["_tag_score"] = scores_arr
    matched = song_data[song_data["_tag_score"] > 0].copy()

    # If still empty, try even looser — any word in name or artist
    if matched.empty:
        def loose_score(row):
            combined = (str(row["name"]) + " " + str(row["artist"]) + " " + str(row["tags"])).lower()
            return sum(1 for w in words if w in combined)
        song_data["_tag_score"] = [loose_score(r) for _, r in song_data.iterrows()]
        matched = song_data[song_data["_tag_score"] > 0].copy()

    if matched.empty:
        song_data.drop(columns=["_tag_score"], inplace=True, errors="ignore")
        return pd.DataFrame(), title, mood

    # Personalize: blend tag score with user's audio profile
    uh = history[history["user_id"] == user_id]
    if not uh.empty:
        user_track_ids   = uh["track_id"].tolist()
        user_indices_all = song_data[song_data["track_id"].isin(user_track_ids)].index.tolist()
        if user_indices_all:
            user_vec = audio_scaled[user_indices_all].mean(axis=0).reshape(1, -1)
            matched["_audio_sim"] = cosine_similarity(user_vec, audio_scaled[matched.index])[0]
            max_ts = matched["_tag_score"].max() or 1
            matched["_final"] = (
                0.45 * (matched["_tag_score"] / max_ts) +
                0.35 * matched["_audio_sim"] +
                0.20 * matched["popularity_score"]
            )
        else:
            max_ts = matched["_tag_score"].max() or 1
            matched["_final"] = (matched["_tag_score"] / max_ts) * 0.7 + matched["popularity_score"] * 0.3
    else:
        max_ts = matched["_tag_score"].max() or 1
        matched["_final"] = (matched["_tag_score"] / max_ts) * 0.7 + matched["popularity_score"] * 0.3

    result = matched.sort_values("_final", ascending=False).drop_duplicates("track_id").head(50)
    song_data.drop(columns=["_tag_score"], inplace=True, errors="ignore")
    return result, title, mood


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "logged_in": False, "username": "", "user_id": None, "si_query": "",
    "taste_count": 10, "genre_count": 10, "song_count": 10,
    "taste_result": None, "genre_result": None, "song_result": None,
    "genre_input": "", "song_input": "",
    "active_tab": 0,
    "explore_source": None,   # "vibe", "season", "ai", "genre_multi"
    "explore_vibe": None,
    "explore_season": None,
    "explore_ai_query": "",
    "explore_ai_result": None,
    "explore_ai_title": "",
    "explore_ai_mood": "",
    "explore_ai_count": 10,
    "explore_season_data": {},
    "explore_season_count": {},
    "selected_genres": [],
    "genre_multi_result": None,
    "genre_multi_count": 10,
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with st.spinner("Loading Waveline…"):
    music, history = load_data()
    song_data, audio_scaled, text_matrix, vibe_names, AUDIO_FEATS = build_model(music, history)

user_ids = sorted(history["user_id"].dropna().unique().tolist())
all_genres_list = song_data["genre"].replace("", np.nan).dropna().value_counts().head(16).index.tolist()
TIER_MEDALS = {1:"#FFD700", 2:"#C0C0C0", 3:"#CD7F32"}


# ─────────────────────────────────────────────
# SIGN-IN
# ─────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("""
    <div style="height:6vh"></div>
    <div style="display:flex;justify-content:center;margin-bottom:2rem">
      <div class="si-card">
        <div class="si-pill">Now Playing</div>
        <div class="si-title">Wave<span>line</span></div>
        <div class="si-sub">Your AI-powered music universe.<br>Search your User ID below to tune in.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        query = st.text_input("uid_search", placeholder="Type 3+ characters of your User ID…", key="si_query", label_visibility="collapsed")
        if query and len(query) >= 3:
            matches = [u for u in user_ids if query.lower() in str(u).lower()][:7]
            if matches:
                st.markdown('<div style="background:#060610;border:1px solid #12122a;border-radius:10px;overflow:hidden;margin-top:4px">', unsafe_allow_html=True)
                for m in matches:
                    if st.button(f"  {m}", key=f"uid__{m}", use_container_width=True):
                        st.session_state.logged_in = True
                        st.session_state.username  = str(m)
                        st.session_state.user_id   = m
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#1e1e38;font-size:0.8rem;margin-top:8px;text-align:center">No matching IDs found</div>', unsafe_allow_html=True)
        elif query:
            st.markdown('<div style="color:#1e1e38;font-size:0.78rem;margin-top:8px;text-align:center">Keep typing…</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#141428;font-size:0.7rem;margin-top:1.4rem;text-align:center">No password required · Demo mode</div>', unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
uid      = st.session_state.user_id
initials = st.session_state.username[:2].upper()
short_id = "···" + st.session_state.username[-3:]

col_logo, _, col_user = st.columns([1, 4, 1])
with col_logo:
    st.markdown('<div class="wl-logo">Wave<span>line</span></div>', unsafe_allow_html=True)
with col_user:
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:flex-end;gap:10px;padding-top:4px">
      <span style="font-size:0.72rem;color:#2e2e50;font-family:monospace">{short_id}</span>
      <div style="width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#3a1e90,#6b44f0);
                  display:flex;align-items:center;justify-content:center;
                  font-family:Syne,sans-serif;font-size:0.78rem;font-weight:700;color:#fff">{initials}</div>
    </div>""", unsafe_allow_html=True)

# Pre-compute
user_history   = history[history["user_id"] == uid]
user_music     = user_history.merge(music, on="track_id", how="left")
user_song_data = user_history.merge(
    song_data[["track_id","vibe","cluster"] + AUDIO_FEATS + ["popularity_score","playcount"]],
    on="track_id", how="left"
)
top10_global = song_data.sort_values(["playcount","popularity_score"], ascending=False).drop_duplicates(subset=["track_id"]).head(10)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
_row_counter = [0]

def render_song_row(i, row, show_score=False, show_genre=True, show_vibe=False, id_prefix="r"):
    _row_counter[0] += 1
    row_id  = f"{id_prefix}_{_row_counter[0]}"
    genre   = get_genre(row) if show_genre else ""
    plays   = f"{int(row.get('playcount',0)):,}" if row.get("playcount",0) > 0 else ""
    preview = str(row.get("spotify_preview_url","")).strip()
    name    = str(row.get("name","Unknown")).replace('"','&quot;').replace("'","&#39;")
    artist  = str(row.get("artist",""))
    vibe    = str(row.get("vibe","")) if show_vibe else ""
    if preview and preview not in ("nan","None",""):
        nc, cls = f'onclick="wlPlay(\'{row_id}\', \'{preview}\')"', "song-name has-preview"
    else:
        nc, cls = "", "song-name no-preview"
    score_html = ""
    if show_score and "score" in row and pd.notna(row.get("score")):
        pct = max(0, min(100, int(float(row["score"])*100)))
        score_html = f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
    badge     = f'<span class="song-badge">{genre}</span>' if genre else ""
    vibe_html = ""
    if vibe and vibe not in ("nan","None",""):
        c = get_vibe_color(vibe)
        vibe_html = f'<span class="vibe-badge" style="background:{c}22;border-color:{c}44;color:{c}">{vibe}</span>'
    st.markdown(f"""
    <div class="song-row">
      <span class="song-num" id="num_{row_id}" data-num="{i+1}">{i+1}</span>
      <div class="song-info">
        <div class="{cls}" {nc}>{name}</div>
        <div class="song-artist">{artist}</div>
        {score_html}
      </div>
      {vibe_html}{badge}
      <span class="song-plays">{plays}</span>
    </div>
    """, unsafe_allow_html=True)


def radar_chart(values, labels, title, color="#7c5cfc", figsize=(4,4)):
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    vp = values + [values[0]]; angles += [angles[0]]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
    ax.plot(angles, vp, color=color, linewidth=2)
    ax.fill(angles, vp, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=7.5, color="#7070a0")
    ax.set_yticklabels([]); ax.set_ylim(0,1)
    ax.spines["polar"].set_color("#1a1a2a"); ax.grid(color="#141428", linewidth=0.8)
    ax.set_title(title, color="#c0c0e0", fontsize=9, pad=16)
    plt.tight_layout(); return fig


def donut_chart(labels, sizes, colors, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops={"edgecolor":"#000","linewidth":2,"width":0.55})
    ax.set_title(title, color="#c0c0e0", fontsize=9, pad=12)
    ax.legend(wedges, [f"{l}  {s}%" for l, s in zip(labels, sizes)],
              loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=7, labelcolor="#5a5a80")
    plt.tight_layout(); return fig


def treemap_html(labels, values, colors):
    total = sum(values)
    W, H  = 560, 340
    items = sorted(zip(values, labels, colors), reverse=True)
    def slice_layout(items, x, y, w, h):
        rects = []
        if not items: return rects
        total_a = sum(i[0] for i in items)
        if w >= h:
            cx = x
            for val, label, color in items:
                rw = w * val / total_a
                rects.append((cx, y, rw, h, label, val, color)); cx += rw
        else:
            cy = y
            for val, label, color in items:
                rh = h * val / total_a
                rects.append((x, cy, w, rh, label, val, color)); cy += rh
        return rects
    top4  = items[:4]; rest = items[4:]
    rects = slice_layout(top4, 0, 0, W, H*0.58) + slice_layout(rest, 0, H*0.58, W, H*0.42)
    pad   = 3
    svg_rects = ""
    for rx, ry, rw, rh, label, val, color in rects:
        cx, cy = rx+rw/2, ry+rh/2
        plays  = f"{int(val/1000)}k" if val>=1000 else str(int(val))
        fs_l   = max(8, min(15, int(rw*0.13)))
        fs_v   = max(7, min(11, int(rw*0.09)))
        show_v = rw > 60 and rh > 40
        val_txt = f"""<text x="{cx}" y="{cy+fs_l}" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="{fs_v}" fill="#ffffff66">{plays} plays</text>""" if show_v else ""
        svg_rects += f"""<rect x="{rx+pad}" y="{ry+pad}" width="{rw-pad*2}" height="{rh-pad*2}" rx="8" fill="{color}" fill-opacity="0.82"/>
<text x="{cx}" y="{cy - (8 if show_v else 0)}" text-anchor="middle" font-family="Syne,sans-serif" font-weight="700" font-size="{fs_l}" fill="#fff">{label}</text>{val_txt}"""
    return f"""<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:12px;display:block">{svg_rects}</svg>"""


def render_more_button(count_key, step=5, max_val=50, label="+ Show more"):
    if st.session_state.get(count_key, 10) < max_val:
        st.markdown('<div class="more-btn-wrap">', unsafe_allow_html=True)
        if st.button(label, key=f"more_{count_key}_{id(count_key)}"):
            st.session_state[count_key] = min(st.session_state.get(count_key, 10) + step, max_val)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def render_playlist(df, count, count_key, id_prefix, max_val=50):
    if df is None or df.empty:
        st.markdown('<div style="color:#28285a;font-size:0.85rem;margin-top:0.5rem">No songs found.</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div style="font-size:0.7rem;color:#28205a;margin-bottom:0.8rem;font-family:Syne,sans-serif;letter-spacing:0.14em;text-transform:uppercase">Showing {min(count,len(df))} of {len(df)} tracks</div>', unsafe_allow_html=True)
    for i, (_, row) in enumerate(df.head(count).iterrows()):
        render_song_row(i, row, show_vibe=True, id_prefix=id_prefix)
    if count < len(df) and count < max_val:
        render_more_button(count_key, max_val=max_val)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_home, tab_analytics, tab_explore, tab_genres, tab_recs, tab_profile = st.tabs([
    "Home", "Analytics", "Explore", "Genres", "For You", "Profile",
])


# ══════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════
with tab_home:
    _row_counter[0] = 0

    st.markdown('<div class="section-label">Global Trends</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.65rem;font-weight:700;color:#d8d8f0;margin-bottom:1.5rem;letter-spacing:-0.02em">What\'s Hot Right Now</div>', unsafe_allow_html=True)

    total_plays    = int(history["playcount"].sum())
    unique_songs   = song_data["track_id"].nunique()
    unique_artists = song_data["artist"].nunique()
    gs             = song_data["genre"].replace("", np.nan).dropna()
    top_genre_val  = gs.value_counts().index[0] if not gs.empty else "—"

    c1,c2,c3,c4 = st.columns(4)
    for col,val,label in [(c1,f"{total_plays/1_000_000:.1f}M","Total Plays"),(c2,f"{unique_songs:,}","Songs"),(c3,f"{unique_artists:,}","Artists"),(c4,top_genre_val[:12],"Top Genre")]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
    col_top, col_right = st.columns([1.3, 1])

    with col_top:
        st.markdown('<div class="section-label">Charts</div><div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Top 10 Most Played</div>', unsafe_allow_html=True)
        for i,(_, row) in enumerate(top10_global.iterrows()):
            render_song_row(i, row, show_vibe=True, id_prefix="home")

    with col_right:
        st.markdown('<div class="section-label">Sound Vibes</div><div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#c8c8e8;margin-bottom:0.6rem">Click to Explore</div>', unsafe_allow_html=True)
        vibe_counts = song_data["vibe"].value_counts()
        for vibe, cnt in vibe_counts.items():
            c = get_vibe_color(str(vibe))
            if st.button(f"{vibe}  ·  {cnt:,}", key=f"vibe_btn_{vibe}"):
                st.session_state.explore_source = "vibe"
                st.session_state.explore_vibe   = vibe
                st.rerun()
            # Style the last button as vibe card via CSS override per button
            st.markdown(f"""<style>
            button[kind="secondary"][data-testid="baseButton-secondary"]:has(div:contains("{vibe}")) {{
              background: {c}12 !important; border: 1px solid {c}30 !important;
              border-radius: 11px !important; color: {c} !important; text-align: left !important;
            }}
            </style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════
with tab_analytics:
    _row_counter[0] = 0

    st.markdown('<div class="section-label">Insights</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.65rem;font-weight:700;color:#d8d8f0;margin-bottom:1.5rem;letter-spacing:-0.02em">Global Analytics</div>', unsafe_allow_html=True)

    # Row 1: Genre bubble chart + Artist leaderboard
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Genre Universe</div>', unsafe_allow_html=True)
        genre_pop  = song_data[song_data["genre"] != ""].groupby("genre")["playcount"].sum().sort_values(ascending=False).head(12)
        bcolors    = ["#7c5cfc","#f04060","#40a0f0","#60c060","#f0a020","#f06090","#40c0b0","#8040a0","#a060f0","#c06030","#5a3dcc","#3d25a0"]
        tm_html = treemap_html(genre_pop.index.tolist(), genre_pop.values.tolist(), bcolors[:len(genre_pop)])
        st.markdown(tm_html, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Artist Leaderboard</div>', unsafe_allow_html=True)
        artist_reach = song_data.groupby("artist").agg(Songs=("track_id","nunique"), Plays=("playcount","sum")).reset_index().sort_values("Plays", ascending=False).head(10)
        max_ap = artist_reach["Plays"].max() + 1
        for rank, (_, row) in enumerate(artist_reach.iterrows(), 1):
            bar  = int(row["Plays"] / max_ap * 100)
            medal = TIER_MEDALS.get(rank, "#2a1880")
            rank_style = f"color:{medal};font-size:{'1.3rem' if rank <= 3 else '0.9rem'}"
            st.markdown(f"""
            <div class="tier-row">
              <span class="tier-rank" style="{rank_style}">{"🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else rank}</span>
              <div class="tier-info">
                <div class="tier-name">{row['artist']}</div>
                <div class="tier-sub">{int(row['Songs'])} songs</div>
                <div style="height:3px;background:#0e0e1e;border-radius:2px;margin-top:5px">
                  <div style="width:{bar}%;height:3px;background:{medal};border-radius:2px;opacity:0.7"></div>
                </div>
              </div>
              <span class="tier-plays">{int(row['Plays']):,}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="wl-divider">', unsafe_allow_html=True)

    # Row 2: Vibe donut + Audio radar
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Vibe Distribution</div>', unsafe_allow_html=True)
        vibe_dist  = song_data["vibe"].value_counts()
        total_v    = vibe_dist.sum()
        d_labels   = vibe_dist.index.tolist()
        d_sizes    = [int(v/total_v*100) for v in vibe_dist.values]
        d_colors   = [get_vibe_color(v) for v in d_labels]
        fig_d = donut_chart(d_labels, d_sizes, d_colors, "Song Vibe Breakdown")
        st.pyplot(fig_d, transparent=True); plt.close()

    with col_d:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Global Audio Fingerprint</div>', unsafe_allow_html=True)
        radar_feats  = ["danceability","energy","valence","acousticness","liveness","speechiness"]
        radar_labels = ["Dance","Energy","Mood","Acoustic","Live","Speech"]
        global_avgs  = song_data[radar_feats].mean().values.tolist()
        fig_r = radar_chart(global_avgs, radar_labels, "Average Audio Profile", color="#7c5cfc")
        st.pyplot(fig_r, transparent=True); plt.close()


# ══════════════════════════════════════════════
# EXPLORE
# ══════════════════════════════════════════════
with tab_explore:
    _row_counter[0] = 0

    st.markdown('<div class="section-label">Explore</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.65rem;font-weight:700;color:#d8d8f0;margin-bottom:0.4rem;letter-spacing:-0.02em">Discover Music</div>', unsafe_allow_html=True)

    # AI Search bar
    st.markdown('<div class="ai-search-wrap"><div class="ai-search-label">AI Playlist Generator — describe any mood, activity, or vibe</div>', unsafe_allow_html=True)
    ai_col1, ai_col2 = st.columns([5, 1])
    with ai_col1:
        ai_query = st.text_input("ai_search", placeholder='e.g. "sad rainy afternoon", "pump up gym session", "cozy sunday morning"…', label_visibility="collapsed", key="ai_search_input")
    with ai_col2:
        ai_go = st.button("Generate", key="ai_go_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if ai_go and ai_query and ai_query.strip():
        st.session_state.explore_ai_query  = ai_query
        st.session_state.explore_source    = "ai"
        st.session_state.explore_ai_count  = 10
        with st.spinner("AI is building your playlist…"):
            res, title, mood = ai_playlist_from_query(ai_query, uid, history, song_data, audio_scaled, text_matrix)
        st.session_state.explore_ai_result = res
        st.session_state.explore_ai_title  = title
        st.session_state.explore_ai_mood   = mood
        st.rerun()

    st.markdown('<hr class="wl-divider">', unsafe_allow_html=True)

    # Season/Mood cards — clickable, no dropdown
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;color:#c8c8e8;margin-bottom:0.8rem">Seasons & Moods</div>', unsafe_allow_html=True)

    SEASON_TAGS = {
        "Winter":    ["winter","cold","snow","cozy","fireplace","december","frost","ice"],
        "Spring":    ["spring","bloom","fresh","flowers","garden","rain","blossom","nature"],
        "Summer":    ["summer","beach","sun","surf","hot","vacation","tropical","pool"],
        "Autumn":    ["autumn","fall","harvest","leaves","october","maple","amber","crisp"],
        "Christmas": ["christmas","xmas","santa","carol","jingle","bells","sleigh","gifts"],
        "Halloween": ["halloween","spooky","horror","scary","creepy","ghost","witch","dark"],
        "New Year":  ["new year","nye","celebration","fireworks","party","countdown","midnight"],
        "Valentine": ["love","romance","valentine","heart","romantic","couples","crush","passion"],
        "Workout":   ["workout","gym","run","energy","pump","training","fitness","power","intense"],
        "Study":     ["study","focus","concentrate","calm","ambient","instrumental","piano","soft"],
        "Party":     ["party","dance","club","dj","bass","banger","anthem","hype","festival"],
        "Sad":       ["sad","cry","heartbreak","tears","grief","sorrow","lonely","miss","broken"],
        "Road Trip": ["road","trip","drive","highway","journey","travel","adventure","cruise"],
        "Sleep":     ["sleep","lullaby","calm","peaceful","night","dream","rest","relax","soothe"],
    }
    SEASON_COLORS = {
        "Winter":"#40a0f0","Spring":"#60c060","Summer":"#f0a020","Autumn":"#c06030",
        "Christmas":"#f04060","Halloween":"#f08020","New Year":"#a060f0","Valentine":"#f06090",
        "Workout":"#f04060","Study":"#40c0b0","Party":"#f0a020","Sad":"#a060f0",
        "Road Trip":"#60c060","Sleep":"#40a0f0",
    }

    # Season color CSS overrides — make each button its season color
    season_css = ""
    for s, c in SEASON_COLORS.items():
        season_css += f""".stButton button[aria-label="{s}"] {{ background: {c}18 !important; border: 1px solid {c}44 !important; color: {c} !important; }}""".strip() + " "
    st.markdown(f"<style>{season_css}</style>", unsafe_allow_html=True)

    # Render as clickable buttons in a grid
    season_names = list(SEASON_TAGS.keys())
    rows_of_4    = [season_names[i:i+4] for i in range(0, len(season_names), 4)]
    for row_items in rows_of_4:
        cols = st.columns(len(row_items))
        for col, s in zip(cols, row_items):
            c = SEASON_COLORS.get(s,"#7c5cfc")
            with col:
                if st.button(s, key=f"season_btn_{s}", use_container_width=True):
                    if s not in st.session_state.explore_season_data:
                        pattern = "|".join(SEASON_TAGS.get(s,[]))
                        res = song_data[song_data["tags"].str.lower().str.contains(pattern, na=False, regex=True)].copy()
                        if res.empty:
                            res = song_data[song_data["name"].str.lower().str.contains(pattern[:20], na=False, regex=True)].copy()
                        res = res.sort_values("popularity_score", ascending=False).drop_duplicates("track_id").head(50)
                        st.session_state.explore_season_data[s] = res
                        st.session_state.explore_season_count[s] = 10
                    st.session_state.explore_source  = "season"
                    st.session_state.explore_season  = s
                    st.session_state.explore_ai_query = ""
                    st.rerun()

    # Hidden Gems card
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    col_hg, _ = st.columns([1, 3])
    with col_hg:
        if st.button("Hidden Gems — Undiscovered Quality", key="hidden_gems_btn", use_container_width=True):
            song_data["audio_quality"] = song_data["energy"]*0.3 + song_data["danceability"]*0.3 + song_data["valence"]*0.4
            hg = song_data[song_data["popularity_score"] < 0.2].sort_values("audio_quality", ascending=False).drop_duplicates("track_id").head(50)
            st.session_state.explore_season_data["Hidden Gems"] = hg
            st.session_state.explore_season_count["Hidden Gems"] = 10
            st.session_state.explore_source  = "season"
            st.session_state.explore_season  = "Hidden Gems"
            st.session_state.explore_ai_query = ""
            st.rerun()

    st.markdown('<hr class="wl-divider">', unsafe_allow_html=True)

    # ── Render active playlist ──
    src = st.session_state.explore_source

    if src == "ai" and st.session_state.explore_ai_result is not None:
        title = st.session_state.explore_ai_title
        mood  = st.session_state.explore_ai_mood
        st.markdown(
            f'<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#d0d0f0;margin-bottom:3px">{title}</div>'
            f'<div style="font-size:0.82rem;color:#3a3a60;margin-bottom:1rem">{mood}</div>',
            unsafe_allow_html=True
        )
        render_playlist(st.session_state.explore_ai_result, st.session_state.explore_ai_count,
                        "explore_ai_count", "ai_pl")

    elif src == "vibe" and st.session_state.explore_vibe:
        vibe = st.session_state.explore_vibe
        c    = get_vibe_color(vibe)
        vibe_songs = song_data[song_data["vibe"] == vibe].sort_values("popularity_score", ascending=False).head(50)
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:{c};margin-bottom:1rem">{vibe} — {len(vibe_songs)} songs</div>', unsafe_allow_html=True)
        count_key = f"vibe_count_{vibe}"
        if count_key not in st.session_state: st.session_state[count_key] = 10
        render_playlist(vibe_songs, st.session_state[count_key], count_key, f"vibe_{vibe}")

    elif src == "season" and st.session_state.explore_season:
        s    = st.session_state.explore_season
        c    = SEASON_COLORS.get(s,"#7c5cfc")
        data = st.session_state.explore_season_data.get(s, pd.DataFrame())
        cnt  = st.session_state.explore_season_count.get(s, 10)
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:{c};margin-bottom:1rem">{s} Playlist — {len(data)} songs</div>', unsafe_allow_html=True)
        render_playlist(data, cnt, f"explore_season_cnt_{s}", f"season_{s}")
        if cnt < len(data):
            st.markdown('<div class="more-btn-wrap">', unsafe_allow_html=True)
            if st.button("+ Show more", key=f"more_season_{s}"):
                st.session_state.explore_season_count[s] = min(cnt + 5, len(data))
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div style="color:#28285a;font-size:0.85rem">Search above or pick a playlist card to get started.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# GENRES
# ══════════════════════════════════════════════
with tab_genres:
    _row_counter[0] = 0

    st.markdown('<div class="section-label">Browse</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.65rem;font-weight:700;color:#d8d8f0;margin-bottom:1rem;letter-spacing:-0.02em">Browse by Genre</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.82rem;color:#28285a;margin-bottom:1rem">Select one or more genres — songs must match all selected.</div>', unsafe_allow_html=True)

    # Clickable multi-select pills
    pills_html = ""
    for g in all_genres_list:
        active = "active" if g in st.session_state.selected_genres else ""
        pills_html += f'<span class="genre-pill {active}" onclick="void(0)">{g}</span>'

    # Render pills as buttons in a flex wrap
    pill_cols = st.columns(8)
    for i, g in enumerate(all_genres_list):
        with pill_cols[i % 8]:
            is_active = g in st.session_state.selected_genres
            label     = f"✓ {g}" if is_active else g
            if st.button(label, key=f"gpill_{g}", use_container_width=True):
                if g in st.session_state.selected_genres:
                    st.session_state.selected_genres.remove(g)
                else:
                    st.session_state.selected_genres.append(g)
                st.session_state.genre_multi_result = None
                st.session_state.genre_multi_count  = 10
                st.rerun()

    # Tag search
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    tag_search = st.text_input("tag_search", placeholder="Or search by tag (e.g. indie, 80s, dreamy)…", label_visibility="collapsed")

    st.markdown('<hr class="wl-divider">', unsafe_allow_html=True)

    if tag_search and tag_search.strip():
        tag = tag_search.strip().lower()
        tr  = song_data[song_data["tags"].str.lower().str.contains(rf"\b{tag}\b", na=False, regex=True)].sort_values(["popularity_score","playcount"], ascending=False).head(50)
        if tr.empty:
            tr = song_data[song_data["tags"].str.lower().str.contains(tag, na=False)].sort_values(["popularity_score","playcount"], ascending=False).head(50)
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Tag: <span style="color:#7c5cfc">"{tag_search}"</span></div>', unsafe_allow_html=True)
        render_playlist(tr if not tr.empty else None, 10, "tag_count", "tag")

    elif st.session_state.selected_genres:
        sg = st.session_state.selected_genres
        if st.session_state.genre_multi_result is None:
            st.session_state.genre_multi_result = recommend_by_genre_multi(sg, song_data, limit=50)

        result = st.session_state.genre_multi_result
        count  = st.session_state.genre_multi_count
        pills_active = " + ".join([f'<span style="color:#7c5cfc">{g}</span>' for g in sg])
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">{pills_active}</div>', unsafe_allow_html=True)

        if result is None or result.empty:
            st.markdown(f'<div style="color:#28285a;font-size:0.85rem">No songs found matching all selected genres. Try fewer genres.</div>', unsafe_allow_html=True)
        else:
            render_playlist(result, count, "genre_multi_count", "gmulti")
    else:
        st.markdown('<div style="color:#28285a;font-size:0.85rem;margin-top:0.5rem">Select genres above or search by tag.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# FOR YOU
# ══════════════════════════════════════════════
with tab_recs:
    _row_counter[0] = 0

    st.markdown('<div class="section-label">Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.65rem;font-weight:700;color:#d8d8f0;margin-bottom:1.2rem;letter-spacing:-0.02em">For You</div>', unsafe_allow_html=True)

    st.markdown('<div class="inner-tabs">', unsafe_allow_html=True)
    inner_tab_taste, inner_tab_genre, inner_tab_song = st.tabs([
        "Based on your taste", "By genre", "Similar to a song"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    with inner_tab_taste:
        _row_counter[0] = 0
        if st.session_state.taste_result is None:
            with st.spinner(""):
                st.session_state.taste_result = recommend_by_user(uid, history, song_data, audio_scaled, text_matrix, limit=50)
        render_playlist(st.session_state.taste_result, st.session_state.taste_count, "taste_count", "taste")

    with inner_tab_genre:
        _row_counter[0] = 0
        genre_input = st.text_input("genre_search", placeholder="Enter a genre, e.g. pop, jazz, metal…", label_visibility="collapsed", key="genre_input_field")
        if genre_input and genre_input.strip():
            if genre_input != st.session_state.genre_input:
                st.session_state.genre_input  = genre_input
                st.session_state.genre_result = recommend_by_genre_multi([genre_input], song_data, limit=50)
                st.session_state.genre_count  = 10
            render_playlist(st.session_state.genre_result, st.session_state.genre_count, "genre_count", "grecs")
        else:
            st.markdown('<div style="color:#28285a;font-size:0.85rem;margin-top:0.8rem">Type a genre to discover songs.</div>', unsafe_allow_html=True)

    with inner_tab_song:
        _row_counter[0] = 0
        song_input = st.text_input("song_search", placeholder="Song name, e.g. Creep, Bohemian Rhapsody…", label_visibility="collapsed", key="song_input_field")
        if song_input and song_input.strip():
            if song_input != st.session_state.song_input:
                st.session_state.song_input  = song_input
                st.session_state.song_result = hybrid_recommend(song_input, song_data, audio_scaled, text_matrix, limit=50)
                st.session_state.song_count  = 10
            matched = song_data[song_data["name"].str.lower().str.contains(song_input.lower(), na=False)]
            if not matched.empty:
                m  = matched.iloc[0]; vc = get_vibe_color(str(m.get("vibe","")))
                st.markdown(f'<div style="background:#0a0a18;border:1px solid #1a1a30;border-radius:12px;padding:0.9rem 1.2rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:12px"><div style="flex:1"><div style="font-size:0.7rem;color:#28205a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px">Because you searched</div><div style="font-size:0.95rem;font-weight:700;color:#d0d0e8">{m["name"]}</div><div style="font-size:0.75rem;color:#282840">{m["artist"]}</div></div><span style="background:{vc}22;border:1px solid {vc}44;color:{vc};border-radius:6px;padding:3px 10px;font-size:0.7rem;font-weight:600">{m.get("vibe","")}</span></div>', unsafe_allow_html=True)
            render_playlist(st.session_state.song_result, st.session_state.song_count, "song_count", "srecs")
        else:
            st.markdown('<div style="color:#28285a;font-size:0.85rem;margin-top:0.8rem">Enter a song name to find similar tracks.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PROFILE
# ══════════════════════════════════════════════
with tab_profile:
    _row_counter[0] = 0

    col_av, col_info = st.columns([1, 3])
    with col_av:
        st.markdown(
            f'<div class="profile-avatar">{initials}</div>'
            f'<div style="font-family:monospace;font-size:0.9rem;font-weight:700;color:#4a4a70">···{st.session_state.username[-3:]}</div>'
            f'<div style="font-size:0.72rem;color:#181830;margin-top:4px">Listener</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        if st.button("Sign out"):
            for k, v in defaults.items(): st.session_state[k] = v
            st.rerun()

    with col_info:
        umc = user_music.copy()
        umc["genre_clean"] = umc.apply(
            lambda r: str(r.get("tags","")).split(",")[0].strip()
            if (pd.isna(r.get("genre","")) or str(r.get("genre","")).strip() == "")
            else r.get("genre","Unknown"), axis=1
        )
        total_plays_u  = int(user_history["playcount"].sum())
        unique_songs_u = user_history["track_id"].nunique()
        fav_genre      = umc["genre_clean"].value_counts().index[0] if not umc.empty else "—"
        c1,c2,c3 = st.columns(3)
        for col,val,label in [(c1,f"{total_plays_u:,}","Total Plays"),(c2,f"{unique_songs_u:,}","Songs Played"),(c3,fav_genre[:10],"Fav Genre")]:
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="wl-divider">', unsafe_allow_html=True)
    col_radar, col_vibes = st.columns(2)

    with col_radar:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Your Sound DNA</div>', unsafe_allow_html=True)
        rf = ["danceability","energy","valence","acousticness","liveness","speechiness"]
        rl = ["Dance","Energy","Mood","Acoustic","Live","Speech"]
        ufd = user_song_data[rf].dropna()
        if not ufd.empty:
            fig_r = radar_chart(ufd.mean().values.tolist(), rl, "Audio Taste Profile", color="#7c5cfc")
            st.pyplot(fig_r, transparent=True); plt.close()

    with col_vibes:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Your Vibe Mix</div>', unsafe_allow_html=True)
        uvd = user_song_data.dropna(subset=["vibe"])
        if not uvd.empty:
            vc2 = uvd["vibe"].value_counts()
            cards = ""
            for vibe, cnt in vc2.items():
                c = get_vibe_color(str(vibe))
                cards += f'<div class="vibe-card" style="background:{c}12;border-color:{c}30"><div class="vibe-card-name" style="font-family:Syne,sans-serif;font-size:0.82rem;font-weight:700;color:{c}">{vibe}</div><div class="vibe-card-count-sm" style="color:{c}">{cnt:,} songs</div></div>'
            st.markdown(f'<div class="vibe-grid">{cards}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="wl-divider">', unsafe_allow_html=True)
    col_hist, col_art = st.columns(2)

    with col_hist:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Listening History</div>', unsafe_allow_html=True)
        tus = user_music.sort_values("playcount", ascending=False).drop_duplicates(subset=["track_id"]).head(8)
        tus = tus.merge(song_data[["track_id","vibe"]], on="track_id", how="left")
        for i,(_, row) in enumerate(tus.iterrows()):
            render_song_row(i, row, show_vibe=True, id_prefix="prof_hist")

    with col_art:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#c8c8e8;margin-bottom:1rem">Top Artists</div>', unsafe_allow_html=True)
        ta  = user_music.groupby("artist").agg(Songs=("track_id","nunique"),Plays=("playcount","sum")).reset_index().sort_values(["Plays","Songs"],ascending=False).head(8)
        mp2 = ta["Plays"].max() + 1
        for i,(_, row) in enumerate(ta.iterrows()):
            bar = min(100, int(row["Plays"]/mp2*100))
            st.markdown(f'<div class="song-row"><span class="song-num">{i+1}</span><div class="song-info"><div class="song-name no-preview">{row["artist"]}</div><div class="song-artist">{int(row["Songs"])} songs · {int(row["Plays"]):,} plays</div><div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar}%"></div></div></div></div>', unsafe_allow_html=True)