"""Presentation layer for the Streamlit app: colour tokens, the one place
custom CSS lives, and small HTML builders (crests, match card, probability
bar, form pills). Nothing here computes a prediction; it only formats
numbers it's handed.

Native theming (.streamlit/config.toml) does the heavy lifting. The CSS
below only styles elements this module generates itself (every class is
prefixed `plm-`), so it doesn't depend on Streamlit's auto-generated class
names and won't break when Streamlit's internals change.
"""
import base64
import hashlib
import html
import re

import matplotlib.pyplot as plt
import streamlit as st

from src.config import ROOT_DIR

# Keep in sync with .streamlit/config.toml.
BG = "#08110c"
SURFACE = "#122019"
SURFACE_2 = "#17291f"
BORDER = "#22362b"
TEXT = "#eef3ef"
MUTED = "#a9bcb1"        # ~10:1 on BG, fine for small text
INK = "#06100a"          # text drawn on top of the coloured bars/pills
HOME = "#2ee07a"         # pitch green
DRAW = "#f2b53a"         # amber
AWAY = "#4c8dff"         # blue
OUTCOME_COLORS = {"H": HOME, "D": DRAW, "A": AWAY}
FORM_COLORS = {"W": HOME, "D": "#8d9a93", "L": "#ff6b6b"}

CRESTS_DIR = ROOT_DIR / "assets" / "crests"
_CREST_MIME = {"png": "image/png", "svg": "image/svg+xml", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
# Fallback badge colours: mid-tones that all keep dark INK text above 4.5:1.
_FALLBACK_PALETTE = ["#5ec8a0", "#f2b53a", "#7aa7ff", "#e58fb7", "#b9a4ff", "#f28b5b", "#62c4d9", "#c4d65b"]


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .plm-hero {{ margin: 0 0 0.25rem 0; }}
        .plm-hero h1 {{ margin: 0; padding: 0; font-size: clamp(1.9rem, 4.5vw, 2.9rem); line-height: 1.05; }}
        .plm-hero .plm-kicker {{ color: {HOME}; font-weight: 600; font-size: 0.8rem; letter-spacing: 0.12em;
                                 text-transform: uppercase; margin-bottom: 0.35rem; }}
        .plm-hero p {{ color: {MUTED}; margin: 0.4rem 0 0 0; font-size: 1.02rem; }}

        .plm-team-head {{ display: flex; align-items: center; gap: 0.75rem; min-height: 56px; }}
        .plm-team-head .plm-role {{ color: {MUTED}; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; }}
        .plm-team-head .plm-name {{ font-weight: 700; font-size: 1.15rem; line-height: 1.2; }}
        .plm-vs {{ display: flex; align-items: center; justify-content: center; height: 100%; min-height: 64px;
                   font-weight: 800; font-size: 1.4rem; color: {MUTED}; letter-spacing: 0.08em; }}

        .plm-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
                     padding: clamp(1rem, 3vw, 1.75rem); }}
        .plm-teams {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 0.75rem; }}
        .plm-side {{ display: flex; flex-direction: column; align-items: center; text-align: center; gap: 0.4rem; min-width: 0; }}
        .plm-side .plm-name {{ font-weight: 700; font-size: clamp(1rem, 2.6vw, 1.45rem); line-height: 1.15;
                               overflow-wrap: anywhere; }}
        .plm-side .plm-role {{ color: {MUTED}; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; }}
        .plm-mid {{ text-align: center; color: {MUTED}; font-size: 0.8rem; }}
        .plm-mid .plm-vs-big {{ font-weight: 800; font-size: clamp(1.2rem, 3vw, 1.7rem); color: {TEXT}; letter-spacing: 0.08em; }}

        .plm-pick {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 0.5rem;
                     margin: 1.25rem 0 0.9rem 0; color: {MUTED}; font-size: 0.9rem; }}
        .plm-pill {{ display: inline-block; padding: 0.3rem 0.75rem; border-radius: 999px; font-weight: 700;
                     color: {INK}; font-size: 0.95rem; }}

        .plm-bar {{ display: flex; width: 100%; height: 46px; border-radius: 12px; overflow: hidden; background: {SURFACE_2}; }}
        .plm-seg {{ display: flex; align-items: center; justify-content: center; color: {INK}; font-weight: 700;
                    font-size: 0.95rem; white-space: nowrap; overflow: hidden; min-width: 0; }}
        .plm-legend {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.35rem 1rem;
                       margin-top: 0.6rem; font-size: 0.88rem; color: {TEXT}; }}
        .plm-legend span {{ display: inline-flex; align-items: center; gap: 0.4rem; }}
        .plm-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
        .plm-sentence {{ margin: 1rem 0 0 0; font-size: 1.02rem; text-align: center; }}
        .plm-note {{ margin: 0.5rem 0 0 0; font-size: 0.88rem; color: {MUTED}; text-align: center; }}

        .plm-empty {{ border: 1px dashed {BORDER}; border-radius: 16px; padding: 2.25rem 1rem; text-align: center; color: {MUTED}; }}
        .plm-empty .plm-empty-icon {{ font-size: 2.2rem; }}
        .plm-empty strong {{ color: {TEXT}; font-size: 1.1rem; display: block; margin: 0.4rem 0 0.2rem 0; }}

        .plm-form {{ display: flex; flex-wrap: wrap; gap: 6px; }}
        .plm-form span {{ display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px;
                          border-radius: 8px; color: {INK}; font-weight: 700; font-size: 0.8rem; }}

        .plm-footer {{ border-top: 1px solid {BORDER}; margin-top: 2.5rem; padding-top: 1rem; color: {MUTED};
                       font-size: 0.85rem; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.5rem; }}

        @media (max-width: 640px) {{
            .plm-teams {{ gap: 0.4rem; }}
            .plm-bar {{ height: 40px; }}
            .plm-seg {{ font-size: 0.8rem; }}
            .plm-vs {{ min-height: 28px; font-size: 1.1rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Crests
# ---------------------------------------------------------------------------

def team_slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")


def team_initials(team: str) -> str:
    words = [w for w in team.replace("'", "").split() if w.lower() != "and"]
    if len(words) == 1:
        return words[0][:3].upper()
    return "".join(w[0] for w in words[:3]).upper()


@st.cache_data(show_spinner=False)
def _crest_data_uri(team: str) -> str | None:
    """A base64 data URI for assets/crests/<slug>.<ext>, or None if no
    such file exists. A data URI embeds straight into the HTML, so no
    static-file serving needs configuring."""
    slug = team_slug(team)
    for ext, mime in _CREST_MIME.items():
        path = CRESTS_DIR / f"{slug}.{ext}"
        if path.exists():
            return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return None


def crest_html(team: str | None, size: int = 48) -> str:
    """The team's crest, or a coloured circle with its initials if there's
    no crest file (e.g. a newly promoted team), or a neutral placeholder
    circle if no team is chosen yet."""
    if not team:
        return (f"<div style='width:{size}px;height:{size}px;border-radius:50%;border:2px dashed {BORDER};"
                f"flex-shrink:0;'></div>")
    uri = _crest_data_uri(team)
    if uri:
        return (f"<img src='{uri}' alt='{html.escape(team, quote=True)} crest' "
                f"style='width:{size}px;height:{size}px;object-fit:contain;flex-shrink:0;' />")
    # md5, not hash(): Python's hash() is randomised per process.
    color = _FALLBACK_PALETTE[int(hashlib.md5(team.encode()).hexdigest(), 16) % len(_FALLBACK_PALETTE)]
    return (f"<div role='img' aria-label='{html.escape(team, quote=True)}' style='width:{size}px;height:{size}px;"
            f"border-radius:50%;background:{color};color:{INK};display:inline-flex;align-items:center;"
            f"justify-content:center;font-weight:800;font-size:{size * 0.32:.0f}px;flex-shrink:0;'>"
            f"{html.escape(team_initials(team))}</div>")


# ---------------------------------------------------------------------------
# Page pieces
# ---------------------------------------------------------------------------

def hero_html(title: str, tagline: str) -> str:
    return (f"<div class='plm-hero'><div class='plm-kicker'>⚽ Match outcome model</div>"
            f"<h1>{html.escape(title)}</h1><p>{html.escape(tagline)}</p></div>")


def team_head_html(team: str | None, role: str) -> str:
    name = html.escape(team) if team else f"<span style='color:{MUTED};font-weight:500;'>Choose a team</span>"
    return (f"<div class='plm-team-head'>{crest_html(team, 48)}<div><div class='plm-role'>{html.escape(role)}</div>"
            f"<div class='plm-name'>{name}</div></div></div>")


def vs_html() -> str:
    return "<div class='plm-vs' aria-hidden='true'>VS</div>"


def empty_state_html(title: str, body: str) -> str:
    return (f"<div class='plm-empty'><div class='plm-empty-icon'>⚽</div><strong>{html.escape(title)}</strong>"
            f"{html.escape(body)}</div>")


def match_card_html(home: str, away: str, when: str, proba, classes, extra_lines: list[str] | None = None) -> str:
    """The main prediction card: both teams, the most likely outcome
    highlighted, and one stacked bar showing all three probabilities
    (a draw is almost never the single most likely outcome, so the full
    split matters more than the pick)."""
    names = {"H": f"{home} win", "D": "Draw", "A": f"{away} win"}
    p = dict(zip(classes, proba))
    top = max(p, key=p.get)

    segments = []
    for c in classes:
        label = f"{p[c]:.0%}" if p[c] >= 0.09 else ""
        segments.append(f"<div class='plm-seg' style='width:{p[c] * 100:.2f}%;background:{OUTCOME_COLORS[c]};' "
                        f"title='{html.escape(names[c], quote=True)}: {p[c]:.1%}'>{label}</div>")
    legend = "".join(
        f"<span><i class='plm-dot' style='background:{OUTCOME_COLORS[c]};'></i>"
        f"{html.escape(names[c])} <strong>{p[c]:.1%}</strong></span>" for c in classes)
    aria = html.escape(", ".join(f"{names[c]} {p[c]:.0%}" for c in classes), quote=True)

    if top == "D":
        sentence = f"The model's most likely outcome is a draw, at {p['D']:.0%}."
    else:
        winner, loser = (home, away) if top == "H" else (away, home)
        other = "A" if top == "H" else "H"
        sentence = (f"The model gives <strong>{html.escape(winner)}</strong> a {p[top]:.0%} chance of winning, "
                    f"{html.escape(loser)} {p[other]:.0%}, and a draw {p['D']:.0%}.")

    extras = "".join(f"<p class='plm-note'>{line}</p>" for line in (extra_lines or []))
    return f"""
    <div class='plm-card'>
      <div class='plm-teams'>
        <div class='plm-side'>{crest_html(home, 72)}<div class='plm-name'>{html.escape(home)}</div><div class='plm-role'>Home</div></div>
        <div class='plm-mid'><div class='plm-vs-big'>VS</div>{html.escape(when)}</div>
        <div class='plm-side'>{crest_html(away, 72)}<div class='plm-name'>{html.escape(away)}</div><div class='plm-role'>Away</div></div>
      </div>
      <div class='plm-pick'>Most likely
        <span class='plm-pill' style='background:{OUTCOME_COLORS[top]};'>{html.escape(names[top])} &middot; {p[top]:.0%}</span>
      </div>
      <div class='plm-bar' role='img' aria-label='{aria}'>{''.join(segments)}</div>
      <div class='plm-legend'>{legend}</div>
      <p class='plm-sentence'>{sentence}</p>
      {extras}
    </div>
    """


def form_pills_html(form_rows: list[dict]) -> str:
    if not form_rows:
        return f"<span style='color:{MUTED};'>No previous matches on record.</span>"
    pills = []
    for r in form_rows:
        title = html.escape(f"{r['outcome']}: {r['venue']} vs {r['opponent']}, {r['score']}", quote=True)
        pills.append(f"<span title='{title}' aria-label='{title}' style='background:{FORM_COLORS[r['outcome']]};'>"
                     f"{r['outcome']}</span>")
    return f"<div class='plm-form'>{''.join(pills)}</div>"


def footer_html(repo_url: str) -> str:
    return (f"<div class='plm-footer'><span>Predictions are for education and analytics, not betting advice.</span>"
            f"<span><a href='{repo_url}' target='_blank' rel='noopener'>Source code on GitHub</a></span></div>")


def themed_figure(figsize):
    """Matplotlib doesn't inherit Streamlit's theme, so match it by hand."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    return fig, ax
