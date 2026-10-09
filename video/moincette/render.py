"""Motion-design renderer: grade -> camera moves -> transitions -> film look -> kinetic overlays."""
import os, sys, subprocess, math, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import timeline as T

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
W, H, FPS = T.W, T.H, T.FPS
NF = int(round(T.TOTAL * FPS))
rng = np.random.default_rng(3)

GRADE = ("hqdn3d=3:3:6:6,eq=gamma=1.72:contrast=1.15:saturation=1.25,"
         "colorbalance=rs=-0.06:gs=0.0:bs=0.05:rm=0.09:gm=0.02:bm=-0.06:rh=0.05:gh=0.02:bh=-0.03,"
         "curves=master='0/0.02 0.2/0.17 0.5/0.55 0.85/0.9 1/0.97'")

# ------------------------------------------------------------------ helpers
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease_out_back(x, s=1.9):
    x = clamp(x) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2
def ease_out_expo(x): x = clamp(x); return 1 if x >= 1 else 1 - 2 ** (-10 * x)
def ease_in_out(x): x = clamp(x); return 0.5 - 0.5 * math.cos(math.pi * x)
def ease_in(x): x = clamp(x); return x * x * x


@functools.lru_cache(None)
def font(name, size):
    return ImageFont.truetype(os.path.join(A, name), size)


@functools.lru_cache(None)
def icon(name, size):
    im = Image.open(os.path.join(A, "icons", name + ".png")).convert("RGBA").resize((size, size), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def hexc(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def text_rgba(text, fnt, fill="#FFFFFF", stroke=8, stroke_fill="#000000", grad=None,
              shadow=(0, 8, 10, 170), glow=None, pad=40):
    """Render text to float RGBA (premultiplied=False). grad=(top_hex, bottom_hex)."""
    bb = fnt.getbbox(text, stroke_width=stroke)
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    org = (pad - bb[0], pad - bb[1])
    mask_fill = Image.new("L", (w, h), 0); ImageDraw.Draw(mask_fill).text(org, text, font=fnt, fill=255)
    mask_all = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_all).text(org, text, font=fnt, fill=255, stroke_width=stroke, stroke_fill=255)
    out = np.zeros((h, w, 4), np.float32)
    if shadow:
        dx, dy, bl, al = shadow
        sh = mask_all.filter(ImageFilter.GaussianBlur(bl))
        sh = np.roll(np.roll(np.asarray(sh, np.float32) / 255.0, dy, 0), dx, 1) * (al / 255.0)
        out[..., 3] = sh
    if glow:
        gcol, gbl, gal = glow
        g = np.asarray(mask_all.filter(ImageFilter.GaussianBlur(gbl)), np.float32) / 255.0 * gal
        c = np.array(hexc(gcol), np.float32) / 255
        out = over(out, np.dstack([np.broadcast_to(c, (h, w, 3)), np.clip(g * 1.6, 0, 1)]))
    st = np.asarray(mask_all, np.float32) / 255.0
    sc = np.array(hexc(stroke_fill), np.float32) / 255
    if stroke:
        out = over(out, np.dstack([np.broadcast_to(sc, (h, w, 3)), st]))
    fm = np.asarray(mask_fill, np.float32) / 255.0
    if grad:
        top, bot = (np.array(hexc(c), np.float32) / 255 for c in grad)
        ys = np.linspace(0, 1, h)[:, None, None]
        col = top * (1 - ys) + bot * ys
        col = np.broadcast_to(col, (h, w, 3))
    else:
        col = np.broadcast_to(np.array(hexc(fill), np.float32) / 255, (h, w, 3))
    out = over(out, np.dstack([col, fm]))
    return out


def over(dst, src):
    """Porter-Duff over for straight-alpha float RGBA of same size."""
    sa = src[..., 3:4]; da = dst[..., 3:4]
    oa = sa + da * (1 - sa)
    rgb = (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    return np.dstack([rgb, oa])


def fit_to(a, h, w):
    out = np.zeros((h, w, 4), np.float32)
    ah, aw = a.shape[:2]
    sy, sx = max(0, (ah - h) // 2), max(0, (aw - w) // 2)
    dy, dx = max(0, (h - ah) // 2), max(0, (w - aw) // 2)
    hh, ww = min(h, ah), min(w, aw)
    out[dy:dy + hh, dx:dx + ww] = a[sy:sy + hh, sx:sx + ww]
    return out


def rounded_rect(w, h, r, color, alpha=1.0, border=None):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = hexc(color)
    d.rounded_rectangle([0, 0, w - 1, h - 1], r, fill=c + (int(255 * alpha),),
                        outline=(hexc(border[0]) + (border[2],)) if border else None,
                        width=border[1] if border else 0)
    return np.asarray(im).astype(np.float32) / 255.0


def transform(rgba, scale=1.0, rot=0.0):
    if abs(scale - 1) < 1e-3 and abs(rot) < 1e-2:
        return rgba
    h, w = rgba.shape[:2]
    s = max(scale, 0.01)
    nw, nh = int(w * s * 1.0 + abs(math.sin(math.radians(rot))) * h * s) + 2, int(h * s + abs(math.sin(math.radians(rot))) * w * s) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    pm = rgba.copy(); pm[..., :3] *= pm[..., 3:4]
    o = cv2.warpAffine(pm, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    o[..., :3] /= np.maximum(o[..., 3:4], 1e-6)
    return o


def blit(frame, rgba, cx, cy, alpha=1.0, scale=1.0, rot=0.0, add=False):
    """frame: float32 HxWx3 in 0..1, modified in place. (cx,cy)=center."""
    if alpha <= 0.003 or scale <= 0.01:
        return
    p = transform(rgba, scale, rot)
    h, w = p.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    sub = p[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = sub[..., 3:4] * alpha
    reg = frame[fy0:fy1, fx0:fx1]
    if add:
        reg += sub[..., :3] * a
    else:
        reg *= (1 - a); reg += sub[..., :3] * a


# ------------------------------------------------------------------ static film-look layers
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.42) / (H * 0.62)) ** 2)
VIGNETTE = np.clip(1 - 0.55 * np.clip(_r - 0.45, 0, None) ** 1.4, 0.35, 1)[..., None].astype(np.float32)
GRAIN = [cv2.GaussianBlur(rng.standard_normal((H // 2, W // 2)).astype(np.float32), (0, 0), 0.6) for _ in range(6)]
GRAIN = [cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] for g in GRAIN]
del yy, xx, _r


def light_leak(t):
    """warm drifting light leak (screen blend), strength varies; small & cheap at 1/8 res."""
    h8, w8 = H // 8, W // 8
    y, x = np.mgrid[0:h8, 0:w8].astype(np.float32)
    out = np.zeros((h8, w8, 3), np.float32)
    for k, (col, sp, ph) in enumerate((((1.0, 0.45, 0.1), 0.13, 0.0), ((1.0, 0.2, 0.35), 0.09, 2.0))):
        cx = w8 * (0.5 + 0.75 * math.sin(t * sp * 2 * math.pi + ph))
        cy = h8 * (0.25 + 0.5 * math.sin(t * sp * 1.3 * math.pi + ph * 1.7))
        d = ((x - cx) ** 2 + (y - cy) ** 2) / (w8 * 0.55) ** 2
        out += np.exp(-d)[..., None] * np.array(col, np.float32)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


def leak_strength(t):
    base = 0.10 + 0.06 * math.sin(t * 0.7)
    for ct, z, kind in T.CUTS:
        dt = t - ct
        if -0.1 < dt < 0.6: base += 0.35 * math.exp(-max(dt, 0) / 0.18)
    return base


# ------------------------------------------------------------------ camera
FACE = (0.47 * W, 0.38 * H)


def smooth_noise(t, seed, freq):
    v = 0
    for k in range(3):
        f = freq * (1.7 ** k)
        v += math.sin(t * f * 2 * math.pi + seed * 12.9898 + k * 78.233) / (1.6 ** k)
    return v


def camera(t):
    z = 1.0; last = 0.0
    for ct, zz, kind in T.CUTS:
        if t >= ct: z, last = zz, ct
    if t < T.CUTS[0][0]:
        z = 1.0 + 0.25 * (1 - ease_out_expo(t / 0.7))            # intro zoom-in
    z += 0.018 * min(t - last, 4.0)                                # slow ken-burns push
    for it, s in T.IMPACTS:
        dt = t - it
        if -0.06 < dt < 0.9:
            a = clamp((dt + 0.06) / 0.08) if dt < 0.02 else math.exp(-(dt - 0.02) / 0.22)
            z += s * a
    # whip / zoomblur transitions push the zoom too
    for ct, zz, kind in T.CUTS:
        dt = t - ct
        if kind == "zoomblur" and -0.2 < dt < 0.15:
            z += 0.25 * (1 - abs(dt) / 0.2) if dt < 0 else 0.2 * (1 - dt / 0.15)
    dx = 5 * smooth_noise(t, 1, 0.35); dy = 5 * smooth_noise(t, 2, 0.3); rot = 0.25 * smooth_noise(t, 3, 0.25)
    for st, amp, dur in T.SHAKES:
        dt = t - st
        if 0 <= dt < dur:
            k = amp * (1 - dt / dur) ** 2
            dx += 26 * k * smooth_noise(t, 4, 9); dy += 26 * k * smooth_noise(t, 5, 8); rot += 2.2 * k * smooth_noise(t, 6, 7)
    for ct, zz, kind in T.CUTS:
        dt = t - ct
        if kind == "whip" and -0.12 < dt < 0.12:
            dx += (dt / 0.12) * -W * 0.45 * (1 - abs(dt) / 0.12) * 2
    if t > T.OUTRO[0]:
        z += 0.06 * ease_in_out((t - T.OUTRO[0]) / 1.6)
    return z, dx, dy, rot


def apply_camera(img, t):
    z, dx, dy, rot = camera(t)
    M = cv2.getRotationMatrix2D(FACE, rot, z)
    M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def transitions(img, t):
    """img uint8 -> uint8"""
    for ct, zz, kind in T.CUTS:
        dt = t - ct
        if kind == "whip" and -0.12 < dt < 0.12:
            k = int(10 + 120 * (1 - abs(dt) / 0.12))
            ker = np.zeros((1, k), np.float32); ker[:] = 1.0 / k
            img = cv2.filter2D(img, -1, ker)
        elif kind == "zoomblur" and -0.17 < dt < 0.12:
            amt = 0.08 * (1 - abs(dt) / (0.17 if dt < 0 else 0.12))
            acc = img.astype(np.float32)
            for i in range(1, 6):
                s = 1 + amt * i / 5
                M = cv2.getRotationMatrix2D(FACE, 0, s)
                acc += cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            img = (acc / 6).astype(np.uint8)
        elif kind == "glitch" and -0.05 < dt < 0.22:
            img = glitch(img, t, 1 - abs(dt - 0.05) / 0.2)
    return img


def glitch(img, t, k):
    k = clamp(k)
    r = np.random.default_rng(int(t * 1000))
    out = img.copy()
    sh = int(18 * k)
    out[..., 0] = np.roll(img[..., 0], sh, 1); out[..., 2] = np.roll(img[..., 2], -sh, 1)
    for _ in range(int(10 * k)):
        y = r.integers(0, H - 40); h = r.integers(8, 90); off = r.integers(-80, 80)
        out[y:y + h] = np.roll(out[y:y + h], int(off * k), 1)
    return out


def chroma(img, amt):
    if amt < 0.5: return img
    s = int(amt)
    o = img.copy(); o[..., 0] = np.roll(img[..., 0], s, 1); o[..., 2] = np.roll(img[..., 2], -s, 1)
    return o


def impact_amount(t):
    a = 0.0
    for it, s in T.IMPACTS:
        dt = t - it
        if 0 <= dt < 0.3: a = max(a, s * 60 * (1 - dt / 0.3))
    return a


# ------------------------------------------------------------------ captions
WORDS = T.load_words()
GROUPS = T.caption_groups(WORDS)
CF = font("MontBlack.ttf", 84)
YELLOW, ACC, RED = "#FFD23F", "#FF4D1A", "#E11D48"


@functools.lru_cache(None)
def word_img(txt, color, boxed, boxcol):
    t = text_rgba(txt, CF, fill=color, stroke=9, shadow=(0, 7, 8, 190), pad=26)
    if boxed:
        h, w = t.shape[:2]
        box = rounded_rect(w - 18, h - 30, 22, boxcol, 1.0)
        bg = np.zeros_like(t); bh, bw = box.shape[:2]
        y0, x0 = (h - bh) // 2 + 2, (w - bw) // 2
        bg[y0:y0 + bh, x0:x0 + bw] = box
        t2 = text_rgba(txt, CF, fill="#FFFFFF", stroke=0, shadow=(0, 4, 4, 120), pad=26 + 9)
        t2 = fit_to(t2, h, w)
        t = over(bg, t2)
    return t


def draw_captions(frame, t):
    hide = any(a - 0.05 <= t <= b for a, b, *_ in T.TITLES if _[2] in ("gold",)) or t > T.OUTRO[0]
    for g in GROUPS:
        if not (g["s"] - 0.04 <= t < g["e"]) or hide:
            continue
        p = (t - g["s"] + 0.04) / 0.16
        sc = 0.55 + 0.45 * ease_out_back(p) if p < 1 else 1.0
        imgs = []
        for w in g["words"]:
            active = w["s"] - 0.03 <= t < max(w["e"], w["s"] + 0.12)
            spoken = t >= w["s"] - 0.03
            col = YELLOW if w["hl"] else "#FFFFFF"
            im = word_img(w["clean"], col, active, RED if w["red"] else ACC)
            pop = 1.0
            if active:
                q = (t - w["s"] + 0.03) / 0.12
                pop = 1.0 + 0.12 * (ease_out_back(q) if q < 1 else 1)
            imgs.append((im, pop, 1.0 if spoken else 0.55))
        gap = -22
        widths = [im.shape[1] * pop for im, pop, _ in imgs]
        total = sum(widths) + gap * (len(imgs) - 1)
        maxw = W - 80
        fit = min(1.0, maxw / total)
        x = W / 2 - total * sc * fit / 2
        cy = H * 0.705 + (1 - min(1, p)) * 40
        for (im, pop, al), wd in zip(imgs, widths):
            ww = wd * sc * fit
            blit(frame, im, x + ww / 2, cy, alpha=min(1, p * 3) * al if al < 1 else min(1, p * 3), scale=sc * pop * fit)
            x += ww + gap * sc * fit
        break


# ------------------------------------------------------------------ titles
STYLE = {
    "pop":    dict(fill="#FFFFFF", grad=None, stroke="#000000", glow=("#FF8A3D", 18, 0.55)),
    "gold":   dict(fill=None, grad=("#FFF3B0", "#F5A300"), stroke="#2A1600", glow=("#FFB800", 24, 0.8)),
    "alert":  dict(fill=None, grad=("#FFFFFF", "#FFC2C2"), stroke="#B00020", glow=("#FF1E1E", 26, 0.9)),
    "glitch": dict(fill="#FFFFFF", grad=None, stroke="#000000", glow=("#00E5FF", 20, 0.7)),
    "fire":   dict(fill=None, grad=("#FFE259", "#FF4E00"), stroke="#2B0A00", glow=("#FF5A00", 26, 0.9)),
    "swipe":  dict(fill="#FFFFFF", grad=None, stroke="#000000", glow=("#FF3B3B", 18, 0.7)),
}


@functools.lru_cache(None)
def title_img(txt, style):
    st = STYLE[style]
    size = 150 if len(txt) <= 9 else (118 if len(txt) <= 15 else 96)
    f = font("Anton.ttf", size)
    lines = [txt]
    if len(txt) > 17:
        words = txt.split(); mid = len(words) // 2
        best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        lines = [" ".join(words[:best]), " ".join(words[best:])]
    ims = [text_rgba(l, f, fill=st["fill"] or "#FFFFFF", grad=st["grad"], stroke=10, stroke_fill=st["stroke"],
                     glow=st["glow"], shadow=(0, 10, 14, 200), pad=40) for l in lines]
    if len(ims) == 1:
        return ims[0]
    w = max(i.shape[1] for i in ims); hh = sum(i.shape[0] for i in ims) - 60
    out = np.zeros((hh, w, 4), np.float32); y = 0
    for i in ims:
        x = (w - i.shape[1]) // 2
        out[y:y + i.shape[0], x:x + i.shape[1]] = over(out[y:y + i.shape[0], x:x + i.shape[1]], i)
        y += i.shape[0] - 60
    return out


def particles(frame, t, t0, seed, color, n=26, cx=W / 2, cy=H * 0.17, spread=520):
    dt = t - t0
    if not (0 <= dt < 0.9): return
    r = np.random.default_rng(seed)
    ang = r.uniform(0, 2 * np.pi, n); spd = r.uniform(0.4, 1.0, n); sz = r.uniform(5, 13, n)
    e = ease_out_expo(dt / 0.9)
    al = 1 - dt / 0.9
    for a, s, z in zip(ang, spd, sz):
        x = cx + math.cos(a) * s * spread * e
        y = cy + math.sin(a) * s * spread * 0.6 * e + 120 * dt * dt
        star = spark(int(z), color)
        blit(frame, star, x, y, alpha=al, add=True)


@functools.lru_cache(None)
def spark(sz, color):
    s = sz * 4 + 1
    im = np.zeros((s, s), np.float32); c = s // 2
    cv2.line(im, (c, 0), (c, s - 1), 1.0, 2); cv2.line(im, (0, c), (s - 1, c), 1.0, 2)
    cv2.circle(im, (c, c), max(2, sz // 2), 1.0, -1)
    im = cv2.GaussianBlur(im, (0, 0), 1.2)
    col = np.array(hexc(color), np.float32) / 255
    return np.dstack([np.broadcast_to(col, (s, s, 3)), np.clip(im * 1.5, 0, 1)])


GLOWCOL = {"pop": "#FFB36B", "gold": "#FFD54A", "alert": "#FF4040", "glitch": "#4FF0FF", "fire": "#FF7A1A", "swipe": "#FF6B6B"}


def draw_titles(frame, t):
    for i, (a, b, txt, ic, style) in enumerate(T.TITLES):
        if not (a - 0.02 <= t < b + 0.18):
            continue
        p = (t - a) / 0.22
        out = clamp((t - b) / 0.18)
        img = title_img(txt, style)
        cy = H * 0.165
        cx = W / 2
        sc = (0.4 + 0.6 * ease_out_back(p, 2.4)) * (1 - 0.3 * ease_in(out))
        if style in ("gold", "fire"):
            sc = (1.6 - 0.6 * ease_out_expo(p)) * (1 - 0.3 * ease_in(out))  # slam in
        al = clamp(p * 2.5) * (1 - out)
        rot = 0
        if style == "alert":
            cx += 10 * math.sin(t * 70) * (1 - clamp((t - a) / 0.6)); rot = 2 * math.sin(t * 45) * (1 - clamp((t - a) / 0.6))
        if style == "swipe":
            cx -= W * 1.1 * ease_in(out); out = 0; al = clamp(p * 2.5)
        th, tw = img.shape[:2]
        # icon
        isz = 230 if style in ("gold", "fire") else 190
        ip = (t - a - 0.06) / 0.3
        isc = ease_out_back(ip, 2.8) * (1 - ease_in(clamp((t - b) / 0.18))) if style != "swipe" else ease_out_back(ip, 2.8)
        bob = 10 * math.sin((t - a) * 5)
        irot = 14 * math.sin((t - a) * 9) * math.exp(-(t - a) * 2.5) + 5 * math.sin((t - a) * 3)
        icy = cy - th * sc / 2 - isz * 0.36 + bob
        if th > 200:                      # two-line title: push block down so the icon stays in frame
            push = 70
            cy += push; icy += push
        icx = cx + (tw * sc / 2 - 30) * (0.55 if len(txt) < 12 else 0.42)
        if style == "glitch" and (t - a) < 0.3:
            k = 1 - (t - a) / 0.3
            for dxo, col in ((-14 * k, (1, 0, 0.3)), (14 * k, (0, 1, 1))):
                tint = img.copy(); tint[..., :3] = col
                blit(frame, tint, cx + dxo, cy, alpha=0.5 * al, scale=sc, add=True)
        blit(frame, img, cx, cy, alpha=al, scale=sc, rot=rot)
        blit(frame, icon(ic, isz), icx, icy, alpha=clamp(ip * 3) * (1 - out if style != "swipe" else 1), scale=max(0.01, isc), rot=irot)
        if style in ("gold", "fire", "pop", "alert"):
            particles(frame, t, a, i * 31 + 5, GLOWCOL[style], n=30 if style != "pop" else 14, cy=cy)


# ------------------------------------------------------------------ glass skill cards
CARD_F = font("MontBlack.ttf", 46)


@functools.lru_cache(None)
def card_text(txt):
    return text_rgba(txt, CARD_F, fill="#FFFFFF", stroke=0, shadow=(0, 3, 4, 140), pad=10)


@functools.lru_cache(None)
def check_img():
    s = 64; im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse([0, 0, s - 1, s - 1], fill=hexc("#22C55E") + (255,))
    d.line([(16, 33), (28, 45), (48, 20)], fill=(255, 255, 255, 255), width=8, joint="curve")
    return np.asarray(im).astype(np.float32) / 255


def glass(frame, x0, y0, w, h, alpha):
    x0, y0 = int(x0), int(y0)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb: return
    reg = frame[ya:yb, xa:xb]
    bl = cv2.GaussianBlur(reg, (0, 0), 18) * 0.55 + 0.06
    m = rounded_rect(w, h, 34, "#FFFFFF", 1.0)[ya - y0:yb - y0, xa - x0:xb - x0, 3:4] * alpha
    reg[:] = reg * (1 - m) + bl * m
    border = rounded_rect(w, h, 34, "#000000", 0.0, border=("#FFFFFF", 3, 110))
    blit(frame, border, x0 + w / 2, y0 + h / 2, alpha=alpha)


def draw_cards(frame, t):
    if not (T.CARDS[0][0] - 0.05 <= t < T.CARDS_END + 0.3):
        return
    out = clamp((t - T.CARDS_END) / 0.3)
    for i, (a, txt, ic) in enumerate(T.CARDS):
        if t < a - 0.05: continue
        p = ease_out_expo((t - a + 0.05) / 0.45)
        cw, ch = 135 + card_text(txt).shape[1] + 90, 118
        x0 = 40 - (1 - p) * 700 - ease_in(out) * 800
        y0 = 120 + i * 140
        al = clamp(p * 2) * (1 - out)
        glass(frame, x0, y0, cw, ch, al)
        ip = (t - a - 0.05) / 0.3
        blit(frame, icon(ic, 96), x0 + 70, y0 + ch / 2, alpha=al, scale=max(0.01, ease_out_back(ip, 2.5)),
             rot=8 * math.sin((t - a) * 6) * math.exp(-(t - a) * 2))
        blit(frame, card_text(txt), x0 + 135 + card_text(txt).shape[1] / 2, y0 + ch / 2, alpha=al)
        cp = (t - a - 0.25) / 0.25
        if cp > 0:
            blit(frame, check_img(), x0 + cw - 50, y0 + ch / 2, alpha=al, scale=max(0.01, ease_out_back(cp, 3)))


# ------------------------------------------------------------------ subscribe button + cursor
BTN_F = font("MontBlack.ttf", 54)


@functools.lru_cache(None)
def button(state):
    col = "#FF0033" if state == 0 else "#3A3A3A"
    txt = "S'ABONNER" if state == 0 else "ABONNÉ"
    tx = text_rgba(txt, BTN_F, fill="#FFFFFF", stroke=0, shadow=None, pad=6)
    w, h = tx.shape[1] + 210, 130
    b = rounded_rect(w, h, 65, col, 1.0)
    sh = np.zeros((h + 40, w + 40, 4), np.float32)
    sh[20:20 + h, 20:20 + w, 3] = cv2.GaussianBlur(b[..., 3], (0, 0), 10) * 0.6
    sh[25:25 + h, 20:20 + w] = over(sh[25:25 + h, 20:20 + w], b)
    out = sh
    x = 20 + 150; y = 20 + (h - tx.shape[0]) // 2
    out[y:y + tx.shape[0], x:x + tx.shape[1]] = over(out[y:y + tx.shape[0], x:x + tx.shape[1]], tx)
    return out


@functools.lru_cache(None)
def cursor():
    s = 90; im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    pts = [(10, 6), (10, 70), (27, 55), (40, 82), (52, 76), (40, 50), (62, 50)]
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=4)
    return np.asarray(im).astype(np.float32) / 255


def draw_sub(frame, t):
    a, b, c = T.SUB_BUTTON
    if not (a <= t < b + 0.25): return
    p = (t - a) / 0.3; out = clamp((t - b) / 0.25)
    state = 0 if t < c + 0.05 else 1
    img = button(state)
    squish = 1.0
    if c - 0.02 < t < c + 0.18:
        squish = 1 - 0.12 * math.sin(math.pi * (t - c + 0.02) / 0.2)
    sc = ease_out_back(p, 2.2) * squish * (1 - 0.4 * ease_in(out))
    cx, cy = W / 2, H * 0.56
    blit(frame, img, cx, cy, alpha=clamp(p * 3) * (1 - out), scale=max(0.01, sc))
    bw = img.shape[1]
    ring = 22 * math.sin((t - c) * 30) * math.exp(-max(0, t - c) * 3) if t > c else 6 * math.sin(t * 8)
    blit(frame, icon("bell", 92), cx - bw * sc / 2 + 105 * sc, cy - 4, alpha=clamp(p * 3) * (1 - out), scale=max(0.01, sc), rot=ring)
    # cursor path: from bottom-right to button, then click
    k = ease_in_out((t - a - 0.15) / (c - a - 0.2))
    px = W * 0.95 + (cx + 120 - W * 0.95) * k; py = H * 0.82 + (cy + 25 - H * 0.82) * k
    cs = 1.0 - (0.18 if c - 0.03 < t < c + 0.1 else 0)
    blit(frame, cursor(), px + 30, py + 35, alpha=clamp((t - a) / 0.2) * (1 - out), scale=cs)
    if t >= c:
        particles(frame, t, c, 99, "#FFFFFF", n=18, cx=cx, cy=cy, spread=300)
        # click ripple
        dt = t - c
        if dt < 0.4:
            rr = np.zeros((241, 241), np.float32)
            cv2.circle(rr, (120, 120), int(20 + 100 * ease_out_expo(dt / 0.4)), 1.0, 6)
            blit(frame, np.dstack([np.ones((241, 241, 3), np.float32), rr]), px + 4, py + 4, alpha=1 - dt / 0.4, add=True)


# ------------------------------------------------------------------ outro
@functools.lru_cache(None)
def outro_imgs():
    t1 = text_rgba("À LA PROCHAINE", font("Anton.ttf", 132), grad=("#FFFFFF", "#FFD23F"), stroke=10,
                   stroke_fill="#000000", glow=("#FF8A3D", 22, 0.7), shadow=(0, 10, 14, 200))
    t2 = text_rgba("abonne-toi pour la suite", font("Marker.ttf", 76), fill="#FFFFFF", stroke=6,
                   stroke_fill="#000000", shadow=(0, 6, 8, 180))
    return t1, t2


def draw_outro(frame, t):
    a = T.OUTRO[0]
    if t < a: return
    dt = t - a
    frame *= 1 - 0.45 * ease_in_out(dt / 0.6)
    t1, t2 = outro_imgs()
    p = dt / 0.35
    blit(frame, t1, W / 2, H * 0.46, alpha=clamp(p * 3), scale=(1.5 - 0.5 * ease_out_expo(p)))
    ip = (dt - 0.1) / 0.35
    blit(frame, icon("wave", 240), W / 2, H * 0.46 - 230, alpha=clamp(ip * 3), scale=max(0.01, ease_out_back(ip, 2.5)),
         rot=22 * math.sin(dt * 9))
    q = (dt - 0.35) / 0.4
    blit(frame, t2, W / 2, H * 0.46 + 140 + (1 - ease_out_expo(q)) * 40, alpha=clamp(q * 2), rot=-3)
    particles(frame, t, a, 777, "#FFD54A", n=36, cy=H * 0.46, spread=600)
    # end fade to black
    f = clamp((t - (T.TOTAL - 0.45)) / 0.45)
    frame *= 1 - f


def draw_progress(frame, t):
    p = clamp(t / T.DURATION)
    h = 9
    frame[0:h, :] *= 0.4
    xe = int(W * p)
    if xe > 0:
        g = np.linspace(0, 1, xe, dtype=np.float32)[None, :, None]
        c = np.array([1.0, 0.82, 0.25], np.float32) * (1 - g) + np.array([1.0, 0.3, 0.1], np.float32) * g
        frame[0:h, :xe] = c


def flash_amount(t):
    a = 0.0
    for ct, z, kind in T.CUTS:
        dt = t - ct
        if kind == "flash" and -0.03 < dt < 0.25:
            a = max(a, 0.9 * math.exp(-max(dt, 0) / 0.07))
    if t < 0.12: a = max(a, 1 - t / 0.12)
    return a


# ------------------------------------------------------------------ dust & scratches (imperfections)
def dust(frame, t, fi):
    r = np.random.default_rng(fi * 7 + 1)
    if r.random() < 0.35:
        for _ in range(r.integers(1, 4)):
            x, y = r.integers(0, W), r.integers(0, H); rad = r.integers(1, 4)
            cv2.circle(frame, (int(x), int(y)), int(rad), (0.92, 0.9, 0.85) if r.random() < 0.5 else (0.05, 0.05, 0.05), -1)
    if r.random() < 0.06:
        x = int(r.integers(0, W))
        cv2.line(frame, (x, 0), (x + int(r.integers(-30, 30)), H), (0.75, 0.72, 0.68), 1)


# ------------------------------------------------------------------ main
def reader():
    parts = []
    for i, (s, e) in enumerate(T.SEGMENTS):
        parts.append(f"[0:v]trim={s}:{e},setpts=PTS-STARTPTS[v{i}]")
    fc = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(T.SEGMENTS))) + \
        f"concat=n={len(T.SEGMENTS)}:v=1:a=0,fps={FPS},{GRADE},scale={W}:{H}:flags=lanczos,format=rgb24[v]"
    cmd = ["ffmpeg", "-v", "error", "-i", os.path.join(HERE, "in.mov"), "-filter_complex", fc, "-map", "[v]",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)


def render_frame(src_u8, fi):
    t = fi / FPS
    img = apply_camera(src_u8, t)
    img = transitions(img, t)
    img = chroma(img, impact_amount(t) * 0.35)
    f = img.astype(np.float32) * (1 / 255.0)
    # exposure flicker + light leak (screen)
    f *= 1 + 0.015 * math.sin(fi * 2.1) * math.sin(fi * 0.37)
    ls = leak_strength(t)
    if ls > 0.01:
        L = light_leak(t) * ls
        f = 1 - (1 - f) * (1 - np.clip(L, 0, 1))
    f *= VIGNETTE
    f += GRAIN[fi % len(GRAIN)] * 0.035
    dust(f, t, fi)
    # overlays
    draw_cards(f, t)
    draw_sub(f, t)
    draw_titles(f, t)
    draw_captions(f, t)
    draw_outro(f, t)
    fl = flash_amount(t)
    if fl > 0: f = f * (1 - fl) + fl
    draw_progress(f, t)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def main(out_path, start=0, end=NF, preview_every=0):
    rd = reader()
    enc = None
    if out_path:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    last = None
    fsz = W * H * 3
    for fi in range(NF):
        buf = rd.stdout.read(fsz) if rd else b""
        if len(buf) == fsz:
            last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        if fi < start: continue
        if fi >= end: break
        if preview_every and (fi - start) % preview_every: continue
        out = render_frame(last, fi)
        if enc:
            enc.stdin.write(out.tobytes())
        else:
            cv2.imwrite(os.path.join(HERE, "frames", f"p_{fi:04d}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        if fi % 150 == 0:
            print("frame", fi, "/", NF, flush=True)
    if enc:
        enc.stdin.close(); enc.wait()
    rd.kill()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        frames = [int(x) for x in sys.argv[2].split(",")]
        rd_frames = sorted(frames)
        rd = reader(); fsz = W * H * 3; last = None; want = set(rd_frames)
        for fi in range(max(rd_frames) + 1):
            buf = rd.stdout.read(fsz)
            if len(buf) == fsz: last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            if fi in want:
                out = render_frame(last, fi)
                cv2.imwrite(os.path.join(HERE, "frames", f"p_{fi:04d}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        rd.kill()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "video_noaudio.mp4"))
