"""README 용 성능 개선 과정 그림(docs/figures/score_journey.png)을 만든다. 제출 기록 숫자만 쓴다.

    python tools/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "score_journey.png"
BLUE, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

_HAVE = {f.name for f in font_manager.fontManager.ttflist}
KO_FONT = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR") if f in _HAVE), "DejaVu Sans")

plt.rcParams.update({
    "font.family": KO_FONT,
    "axes.unicode_minus": False, "font.size": 10.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
})

# (제출 번호, 버전, Public, 기각) - experiments/submissions.md
SUBS = [
    (1, "baseline", 0.3616, False), (2, "v2", 0.4981, False), (3, "v3", 0.7408, False),
    (4, "v4", 0.8289, False), (5, "v6", 0.8979, False), (6, "v7", 0.8998, False),
    (7, "v8", 0.8987, True), (8, "v11", 0.9040, False), (9, "v12", 0.9066, False),
    (10, "v13", 0.9066, False), (11, "v14", 0.9096, False), (12, "v15", 0.9228, False),
    (13, "v16", 0.9333, False), (14, "v17", 0.9452, False), (15, "v18", 0.9478, False),
    (16, "v19", 0.9460, True), (17, "v20", 0.9570, False), (18, "v21a", 0.9586, False),
    (19, "v21b", 0.9652, False),
]
# 위 패널: 큰 변화마다 한 줄 설명
TOP_NOTES = {
    "v2": "후보 검색 + 순위 학습",
    "v3": "한글·영문 발음 키",
    "v4": "노이즈 증강 학습",
    "v6": "LLM 영문 음역",
}
# 아래 패널: 구간 설명 (x 시작, x 끝, 글)
ZOOM_SPANS = [
    (7.5, 10.5, "앙상블·신경망 재판별\n이후 3일 정체"),
    (10.5, 15.5, "train 선박 제외 +\n오타에 강한 퍼지 검색"),
    (15.5, 19.5, "카탈로그를 한글로 음역해\n한글끼리 비교"),
]
ZOOM_NOTES = {
    "v19": "재학습\n(기각)", "v20": "한글 음역표", "v21a": "영문 퍼지",
    "v21b": "음역 n-best", "v15": "퍼지 검색", "v14": "train 선박 제외",
}


def plot_line(ax, subs):
    xs, ys = [s[0] for s in subs], [s[2] for s in subs]
    ax.plot(xs, ys, color=BLUE, linewidth=2, zorder=3)
    for n, _, y, rej in subs:
        ax.scatter(n, y, s=60, zorder=4, linewidth=2,
                   facecolor=SURFACE if rej else BLUE, edgecolor=MUTED if rej else SURFACE)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[1] for s in subs], fontsize=8.5)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)


def main():
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10.5, 9), gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.38})

    plot_line(top, SUBS)
    top.set_ylim(0.3, 1.0)
    top.set_title("Public 점수: 공식 베이스라인 0.362 → 최종 0.965 (제출 19회)")
    top.axhspan(0.895, 0.97, color=BLUE, alpha=0.07, linewidth=0)
    top.text(19.5, 0.885, "아래에서 확대", ha="right", va="top", fontsize=8.5, color=BLUE)
    for n, v, y, _ in SUBS:
        if v in TOP_NOTES:
            top.annotate(f"{TOP_NOTES[v]}\n{y:.3f}", (n, y), textcoords="offset points", xytext=(9, -6),
                         ha="left", va="top", fontsize=8.5, color=INK2, linespacing=1.15)
    top.annotate("0.362", (1, 0.3616), textcoords="offset points", xytext=(8, -2), ha="left", va="center",
                 fontsize=8.5, color=INK2)

    tail = [s for s in SUBS if s[0] >= 8]
    plot_line(bot, tail)
    bot.set_ylim(0.895, 0.99)
    bot.set_title("확대: 0.904 → 0.965")
    for i, (x0, x1, text) in enumerate(ZOOM_SPANS):
        if i % 2 == 0:
            bot.axvspan(x0, x1, color=GRID, alpha=0.45, linewidth=0)
        bot.text((x0 + x1) / 2, 0.988, text, ha="center", va="top", fontsize=9, color=INK2, linespacing=1.2)
    for n, v, y, rej in tail:
        below = rej or v in ("v14",)
        label = f"{ZOOM_NOTES[v]}\n{y:.4f}" if v in ZOOM_NOTES else f"{y:.4f}"
        bot.annotate(label, (n, y), textcoords="offset points", xytext=(0, -11 if below else 9), ha="center",
                     va="top" if below else "bottom", fontsize=8, color=MUTED if rej else INK2, linespacing=1.1)
    bot.scatter([], [], s=60, facecolor=BLUE, edgecolor=SURFACE, label="채택")
    bot.scatter([], [], s=60, facecolor=SURFACE, edgecolor=MUTED, linewidth=2, label="기각 (점수 하락)")
    bot.legend(loc="lower right", frameon=False, fontsize=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
