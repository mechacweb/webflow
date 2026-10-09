"""Shared edit decision list: cuts, captions, motion events (all times in OUTPUT seconds)."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
W, H = 1080, 1920

# Source segments kept (src_start, src_end) -> concatenated
SEGMENTS = [(2.85, 51.40), (52.80, 64.70)]
LEN_A = SEGMENTS[0][1] - SEGMENTS[0][0]
DURATION = sum(e - s for s, e in SEGMENTS)
OUTRO_TAIL = 1.2                    # extra frozen/animated seconds at the end
TOTAL = DURATION + OUTRO_TAIL


def o(src):
    """source time -> output time"""
    acc = 0.0
    for s, e in SEGMENTS:
        if src < e + 0.001:
            return acc + max(0.0, src - s)
        acc += e - s
    return acc


# ---------------------------------------------------------------- captions
FIX = {
    "C@3.06": "SI C",
    "fugaliser": "VULGARISER",
    "'intéfaces,": "'INTERFACES,",
    "putain": "P*TAIN",
    "merde": "M*RDE",
    "soit@53.42": "SOIS",
}
HIGHLIGHT = {
    "première", "fois", "boss", "boss,", "domaines", "agentique,", "'interfaces,", "conception",
    "'abonner", "'informatique", "carrière,", "carrière", "véritables", "compétences", "compétences,",
    "business", "important", "très", "vulgariser", "bête", "voisin", "passer", "'énergie", "énergie",
    "spirituelle", "profonde,", "prêche,", "'évangélisation", "branché,", "ensemble", "swipe",
    "opportunité", "p*tain", "m*rde", "prochaine", "vidéos", "construire", "postures,",
}
RED = {"p*tain", "m*rde", "important", "swipe"}


def load_words():
    data = json.load(open(os.path.join(HERE, "words.json")))
    words = []
    for seg in data:
        for j, w in enumerate(seg["words"]):
            raw = w["w"].strip()
            key = f"{raw}@{w['s']:.2f}"
            txt = FIX.get(key, FIX.get(raw, raw))
            words.append({"raw": raw, "txt": txt, "s": o(w["s"]), "e": o(w["e"]), "src": w["s"], "first": j == 0})
    # glue elisions ("C" + "'est", "t" + "'abonner") into one token
    merged = []
    for w in words:
        if merged and w["txt"].startswith("-"):
            merged[-1]["txt"] += w["txt"]; merged[-1]["e"] = w["e"]; merged[-1]["raw"] += w["raw"]
        elif merged and w["txt"].startswith("'") and len(merged[-1]["raw"]) <= 2 \
                and w["s"] - merged[-1]["e"] < 0.15 and not merged[-1]["txt"].endswith((",", ".")):
            m = merged[-1]
            m["txt"] = m["txt"] + w["txt"]; m["e"] = w["e"]; m["raw"] = m["raw"] + w["raw"]
        else:
            merged.append(dict(w))
    for w in merged:
        w["txt"] = w["txt"].upper().replace("SI C'EST'EST", "SI C'EST")
        base = re.sub(r"^[A-Z]'", "", w["raw"].lower()).strip()
        k = w["txt"].lower()
        core = k.split("'")[-1]
        w["hl"] = any(x in HIGHLIGHT for x in (k, core, base, "'" + core))
        w["red"] = any(x.strip(",") in RED for x in (k, core))
        w["clean"] = w["txt"].rstrip(",.")
    return merged


def caption_groups(words):
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        chars = sum(len(x["clean"]) + 1 for x in cur)
        brk = (nxt is None or nxt["first"] or w["txt"].endswith((",", ".")) or len(cur) >= 3 or chars > 14
               or (nxt and nxt["s"] - w["e"] > 0.35) or (nxt and chars + len(nxt["clean"]) > 17))
        if brk:
            groups.append(cur); cur = []
    out = []
    for g in groups:
        out.append({"words": g, "s": g[0]["s"], "e": g[-1]["e"]})
    for a, b in zip(out, out[1:]):       # hold until next group (no flicker) if gap small
        if b["s"] - a["e"] < 0.6:
            a["e"] = b["s"]
        else:
            a["e"] += 0.25
    out[-1]["e"] += 0.3
    return out


# ---------------------------------------------------------------- motion design events
# punch-in jump cuts (sentence starts): (time, zoom level after cut, transition type)
CUTS = [
    (o(8.16), 1.16, "flash"), (o(11.82), 1.0, "whip"), (o(16.18), 1.12, "glitch"),
    (o(21.54), 1.0, "zoomblur"), (o(24.08), 1.14, "whip"), (o(28.24), 1.0, "flash"),
    (o(30.86), 1.20, "glitch"), (o(34.52), 1.06, "zoomblur"), (o(36.68), 1.0, "whip"),
    (o(40.60), 1.14, "glitch"), (o(44.66), 1.04, "zoomblur"), (o(46.72), 1.18, "flash"),
    (o(48.22), 1.04, "whip"), (LEN_A, 1.12, "flash"), (o(56.34), 1.0, "glitch"),
    (o(57.88), 1.16, "whip"), (o(60.82), 1.04, "zoomblur"),
]
# smooth zoom pulses + shake on impact words: (time, strength)
IMPACTS = [
    (o(9.30), 0.10), (o(10.34), 0.13), (o(12.74), 0.05), (o(17.44), 0.06), (o(23.30), 0.07),
    (o(25.12), 0.06), (o(30.86), 0.12), (o(35.44), 0.06), (o(35.80), 0.09), (o(36.18), 0.14),
    (o(40.04), 0.07), (o(42.16), 0.10), (o(46.04), 0.14), (o(49.38), 0.09), (o(53.86), 0.10),
    (o(59.34), 0.08), (o(60.02), 0.12), (o(60.54), 0.12),
]
SHAKES = [(o(36.18), 0.6, 1.0), (o(42.16), 0.5, 0.8), (o(46.04), 0.7, 1.0), (o(60.02), 0.8, 1.0),
          (o(10.34), 0.4, 0.6)]

# big kinetic titles: (start, end, text, icon, style)
TITLES = [
    (o(3.90), o(5.60), "PREMIÈRE FOIS ?", "eyes", "pop"),
    (o(9.20), o(11.70), "LE BOSS", "crown", "gold"),
    (o(21.0), o(21.9), "INFORMATIQUE", "laptop", "pop"),
    (o(22.8), o(24.0), "CARRIÈRE", "rocket", "pop"),
    (o(24.5), o(26.3), "VRAIES COMPÉTENCES", "brain", "pop"),
    (o(30.8), o(32.6), "BUSINESS", "money", "gold"),
    (o(35.3), o(37.0), "TRÈS TRÈS IMPORTANT", "warning", "alert"),
    (o(38.0), o(39.0), "JE VULGARISE TOUT", "bulb", "pop"),
    (o(39.2), o(40.8), "DORMIR MOINS BÊTE", "sleep", "pop"),
    (o(41.6), o(43.6), "QUELQUE CHOSE VA SE PASSER", "bolt", "glitch"),
    (o(44.9), o(47.0), "ÉNERGIE SPIRITUELLE", "fire", "fire"),
    (o(48.5), o(49.3), "LA PRÊCHE", "pray", "pop"),
    (o(49.3), o(51.4), "ÉVANGÉLISATION", "mega", "fire"),
    (o(53.3), o(55.0), "RESTE BRANCHÉ", "plug", "pop"),
    (o(57.8), o(58.9), "SWIPE", "point", "swipe"),
    (o(59.1), o(60.0), "OPPORTUNITÉ", "gem", "pop"),
    (o(60.0), o(60.85), "#$@%!", "skull", "alert"),
]
# stacked skill cards (glass cards sliding in on the left)
CARDS = [(o(11.98), "SYSTÈMES AGENTIQUES", "robot"), (o(13.88), "INTERFACES", "desktop"),
         (o(15.02), "CONCEPTION", "palette")]
CARDS_END = o(16.30)
SUB_BUTTON = (o(16.70), o(19.40), o(17.55))   # start, end, click time
OUTRO = (o(61.0), TOTAL)
BLEEP = (o(60.02), o(60.82))

# music structure (output seconds)
DROP1 = o(8.16)            # "c'est que je suis le boss"
BREAK = o(36.68)           # breakdown
DROP2 = o(44.66)           # "c'est une énergie vraiment spirituelle"
MUSIC_END = o(64.50)
