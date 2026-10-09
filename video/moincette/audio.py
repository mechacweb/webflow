"""Procedural dynamic music + sound design, aligned on the edit timeline."""
import numpy as np, wave, os
from scipy.signal import butter, sosfilt, fftconvolve
import timeline as T

SR = 44100
N = int((T.TOTAL + 1.0) * SR)
rng = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x)


def env(n, a=0.002, d=0.2):
    t = np.arange(n) / SR
    e = np.exp(-t / d)
    na = max(1, int(a * SR))
    e[:na] *= np.linspace(0, 1, na)
    return e


def add(buf, sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i < 0:
        sig = sig[-i:]; i = 0
    n = min(len(sig), buf.shape[0] - i)
    if n <= 0:
        return
    if sig.ndim == 1:
        l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
        buf[i:i + n, 0] += sig[:n] * gain * l * 1.414
        buf[i:i + n, 1] += sig[:n] * gain * r * 1.414
    else:
        buf[i:i + n] += sig[:n] * gain


def reverb(x, secs=1.6, wet=0.3, damp=4000):
    n = int(secs * SR)
    ir = rng.standard_normal((n, 2)) * np.exp(-np.arange(n) / (secs * SR / 6.0))[:, None]
    ir = np.stack([lp(ir[:, 0], damp), lp(ir[:, 1], damp)], 1)
    ir /= np.sqrt((ir ** 2).sum(0))
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    out = np.stack([fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return x * (1 - wet) + out * wet


def note(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def saw(f, n, phase=0.0):
    t = np.arange(n) / SR
    return 2 * ((t * f + phase) % 1.0) - 1


# ------------------------------------------------------------------ instruments
def kick():
    n = int(0.45 * SR); t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    k = np.sin(ph) * np.exp(-t / 0.22)
    k += 0.5 * hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.004)
    return np.tanh(k * 1.6) * 0.9


def clap():
    n = int(0.35 * SR); t = np.arange(n) / SR
    nz = bp(rng.standard_normal(n), 900, 5000)
    e = np.zeros(n)
    for off in (0, 0.011, 0.022):
        i = int(off * SR); e[i:] += np.exp(-(t[:n - i]) / 0.008) * 0.6
    e += np.exp(-np.maximum(t - 0.03, 0) / 0.09) * (t > 0.03)
    return nz * e * 0.9


def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    return hp(rng.standard_normal(n), 7000, 3) * env(n, 0.0005, 0.08 if open_ else 0.015) * 0.5


def snare():
    n = int(0.25 * SR); t = np.arange(n) / SR
    s = bp(rng.standard_normal(n), 1500, 8000) * np.exp(-t / 0.06)
    s += np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05) * 0.5
    return s * 0.7


def supersaw(m_list, n, voices=5, det=0.12):
    out = np.zeros(n)
    for m in m_list:
        for v in range(voices):
            d = (v - (voices - 1) / 2) / ((voices - 1) / 2) * det
            out += saw(note(m + d), n, rng.random())
    return out / (len(m_list) * voices)


def pluck(m, dur=0.22):
    n = int(dur * SR); t = np.arange(n) / SR
    x = 0.6 * saw(note(m), n) + 0.4 * saw(note(m + 12.05), n)
    # one-pole lowpass with decaying cutoff
    fc = 300 + 5000 * np.exp(-t / 0.05)
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.zeros(n); s = 0.0
    for i in range(n):
        s = (1 - a[i]) * x[i] + a[i] * s; y[i] = s
    return y * np.exp(-t / 0.12)


def sub(m, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    s = np.sin(2 * np.pi * note(m) * t) + 0.3 * np.sin(4 * np.pi * note(m) * t)
    e = np.minimum(1, t / 0.005) * np.minimum(1, (dur - t) / 0.02)
    return np.tanh(s * 1.5) * e * 0.7


# ------------------------------------------------------------------ music
BPM = 120.0
BEAT = 60 / BPM
BAR = 4 * BEAT
T0 = T.DROP1 - 3 * BAR      # bar grid aligned so DROP1 is on a downbeat
PROG = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [48, 52, 55, 60]), (55, [55, 59, 62])]  # Am F C G


def section(t):
    if t < T.DROP1: return "intro"
    if t < T.BREAK: return "drop"
    if t < T.DROP2: return "break"
    if t < T.MUSIC_END: return "drop2"
    return "end"


def build_music():
    drums = np.zeros((N, 2)); bass = np.zeros((N, 2)); pads = np.zeros((N, 2)); arps = np.zeros((N, 2))
    fx = np.zeros((N, 2))
    K, C, HC, HO, SN = kick(), clap(), hat(), hat(True), snare()
    first = int(np.floor((0 - T0) / BAR))
    nbars = int((T.MUSIC_END - T0) / BAR) + 2
    pumps = []
    arp_cache = {}
    for b in range(first, nbars):
        bt = T0 + b * BAR
        if bt > T.MUSIC_END + 0.01:
            break
        root, chord = PROG[b % 4]
        sec = section(bt + 0.01)
        # pad (always, filtered in intro/break)
        n = int(BAR * SR)
        p = supersaw([c for c in chord] + [chord[0] - 12], n)
        att = np.minimum(1, np.arange(n) / (0.05 * SR))
        p *= att * np.minimum(1, (n - np.arange(n)) / (0.03 * SR))
        if sec in ("intro", "break"):
            p = lp(p, 900, 2) * 1.4
        else:
            p = lp(p, 3500, 2)
        add(pads, p, bt, 0.22, 0)
        for beat in range(4):
            t = bt + beat * BEAT
            if t >= T.MUSIC_END: break
            if sec in ("drop", "drop2"):
                add(drums, K, t, 0.95); pumps.append(t)
                if beat in (1, 3): add(drums, C, t, 0.55, 0.05)
                add(drums, HC, t + BEAT / 2, 0.35, 0.3)
                add(drums, HO, t + BEAT / 2, 0.12, 0.3)
                for s16 in (0.25, 0.75):
                    add(drums, HC, t + s16 * BEAT, 0.12 + 0.1 * rng.random(), -0.3)
                if sec == "drop2" and beat == 3 and b % 2 == 1:
                    for r in range(4): add(drums, SN, t + r * BEAT / 4, 0.25 + 0.08 * r)
                # bass on offbeats + root
                add(bass, sub(root - 24, BEAT * 0.45), t + BEAT / 2, 0.75)
                if beat == 0: add(bass, sub(root - 24, BEAT * 0.3), t, 0.55)
            elif sec == "intro":
                add(drums, HC, t + BEAT / 2, 0.15, 0.3)
                if beat == 0 and bt >= 0: add(drums, K, t, 0.35)
            elif sec == "break":
                add(drums, HC, t + BEAT / 2, 0.12, 0.3)
            # arp (16ths)
            for s in range(4):
                tt = t + s * BEAT / 4
                if tt >= T.MUSIC_END: break
                m = chord[(beat * 4 + s) % len(chord)] + 12 * (1 if s % 2 == 0 else 2)
                if m not in arp_cache: arp_cache[m] = pluck(m)
                g = {"intro": 0.10, "drop": 0.16, "break": 0.13, "drop2": 0.2}.get(sec, 0)
                add(arps, arp_cache[m], tt, g, -0.4 if s % 2 else 0.4)
    # snare roll into each drop
    for drop, dur in ((T.DROP1, 2 * BEAT * 2), (T.DROP2, 4 * BEAT * 2)):
        steps = int(dur / (BEAT / 4))
        for i in range(steps):
            frac = i / steps
            add(drums, SN, drop - dur + i * BEAT / 4, 0.08 + 0.35 * frac ** 2)
    # sidechain pump on bass/pads/arp
    pump = np.ones(N)
    pn = int(0.35 * SR); shape = 1 - 0.75 * np.exp(-np.arange(pn) / (0.09 * SR))
    for t in pumps:
        i = int(t * SR); j = min(N, i + pn)
        if i < N: pump[i:j] = np.minimum(pump[i:j], shape[:j - i])
    pads *= pump[:, None]; bass *= pump[:, None]; arps *= pump[:, None]
    # filter sweep in break
    arps = reverb(arps, 1.2, 0.35)
    pads = reverb(pads, 2.2, 0.35)
    mix = drums + bass + pads + arps
    # risers + impacts at drops
    for drop, ln in ((T.DROP1, 2.6), (T.DROP2, 3.8)):
        add(fx, riser(ln), drop - ln, 0.35)
        add(fx, impact(1.8), drop, 0.6)
    add(fx, impact(2.5), T.MUSIC_END, 0.7)
    add(fx, reverse_cymbal(1.2), T.MUSIC_END - 1.2, 0.25)
    mix += fx
    # final tail: fade after music end
    t = np.arange(N) / SR
    fade = np.clip(1 - (t - (T.MUSIC_END + 0.3)) / 1.5, 0, 1)
    fade *= np.clip(t / 0.4, 0, 1)
    mix *= fade[:, None] ** 1.5
    mix = np.tanh(mix * 1.1) * 0.85
    return mix


# ------------------------------------------------------------------ sfx
def riser(ln):
    n = int(ln * SR); t = np.arange(n) / SR; fr = t / ln
    nz = rng.standard_normal(n)
    out = np.zeros(n); chunk = 2048
    for i in range(0, n, chunk):
        f = 400 + 9000 * fr[i] ** 2
        out[i:i + chunk] = bp(nz[i:i + chunk + 0], max(100, f * 0.6), min(SR / 2 - 100, f * 1.4), 1)
    f0 = 200 * 2 ** (3 * fr)
    tone = np.sin(2 * np.pi * np.cumsum(f0) / SR) * 0.3
    return (out * 0.8 + tone) * fr ** 1.6


def reverse_cymbal(ln):
    n = int(ln * SR)
    c = hp(rng.standard_normal(n), 5000) * np.exp(-np.arange(n) / (0.35 * SR))
    return c[::-1] * 0.8


def impact(ln=1.5):
    n = int(ln * SR); t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(30 + 90 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.45)
    crack = lp(rng.standard_normal(n), 2500) * np.exp(-t / 0.12) * 0.6
    return reverb(np.tanh((boom + crack) * 1.8), 2.0, 0.35)


def whoosh(ln=0.45, up=True):
    n = int(ln * SR); t = np.arange(n) / SR; fr = t / ln
    nz = rng.standard_normal(n); out = np.zeros(n); chunk = 1024
    for i in range(0, n, chunk):
        f = 300 + 5000 * (fr[i] if up else 1 - fr[i]) ** 1.5
        out[i:i + chunk] = bp(nz[i:i + chunk], f * 0.5, min(f * 1.8, 20000), 1)
    e = np.sin(np.pi * fr) ** 2
    return out * e * 1.2


def pop():
    n = int(0.12 * SR); t = np.arange(n) / SR
    f = 300 + 900 * np.exp(-t / 0.015)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.04) * 0.8


def click():
    n = int(0.05 * SR); t = np.arange(n) / SR
    return (hp(rng.standard_normal(n), 2000) * np.exp(-t / 0.003) +
            np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.01) * 0.5) * 0.8


def ding(f=1320, ln=1.4):
    n = int(ln * SR); t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (0.6 / r)) for r, a in
            ((1, 1), (2.76, 0.4), (5.4, 0.2), (8.9, 0.1)))
    return reverb(s * 0.5, 1.5, 0.3)


def shimmer():
    out = np.zeros(int(1.2 * SR))
    for i, m in enumerate([81, 84, 88, 93]):
        d = ding(note(m), 1.0)[:, 0]
        j = int(i * 0.05 * SR); out[j:j + len(d)] += d[:len(out) - j] * 0.5
    return out


def glitch():
    n = int(0.3 * SR); out = np.zeros(n); i = 0
    while i < n:
        ln = int(rng.uniform(0.01, 0.04) * SR)
        kind = rng.integers(3)
        t = np.arange(ln) / SR
        if kind == 0: s = np.sign(np.sin(2 * np.pi * rng.uniform(200, 1500) * t))
        elif kind == 1: s = np.round(rng.standard_normal(ln) * 2) / 2
        else: s = np.zeros(ln)
        out[i:i + ln] = s[:n - i] * 0.35; i += ln
    return hp(out, 150)


def thump():
    n = int(0.4 * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * np.cumsum(40 + 60 * np.exp(-t / 0.03)) / SR) * np.exp(-t / 0.15)


def shutter():
    n = int(0.12 * SR); t = np.arange(n) / SR
    a = hp(rng.standard_normal(n), 1500) * (np.exp(-t / 0.006) + 0.7 * np.exp(-np.maximum(0, t - 0.06) / 0.006) * (t > 0.06))
    return a * 0.7


def bleep(ln):
    n = int(ln * SR); t = np.arange(n) / SR
    e = np.minimum(1, np.minimum(t, ln - t) / 0.005)
    return np.sin(2 * np.pi * 1000 * t) * e * 0.35


def build_sfx():
    s = np.zeros((N, 2))
    for t, z, kind in T.CUTS:
        if kind == "flash":
            add(s, shutter(), t, 0.5); add(s, thump(), t, 0.5)
        elif kind == "whip":
            add(s, whoosh(0.35), t - 0.2, 0.45, 0.4)
        elif kind == "glitch":
            add(s, glitch(), t - 0.03, 0.4)
        else:
            add(s, whoosh(0.5, True), t - 0.38, 0.4, -0.3)
    for t, st in T.IMPACTS:
        if st >= 0.09: add(s, thump(), t, 0.55 * st / 0.14)
    for (a, b, txt, icon, style) in T.TITLES:
        add(s, pop(), a, 0.4, 0.2)
        if style == "gold": add(s, shimmer(), a, 0.3)
        if style == "glitch": add(s, glitch(), a, 0.35)
        if style == "fire": add(s, whoosh(0.6, True), a - 0.3, 0.35); add(s, impact(1.2), a, 0.25)
        if style == "swipe": add(s, whoosh(0.3, False), a + 0.1, 0.5, -0.8)
        if style == "alert": add(s, thump(), a, 0.5)
    for (t, txt, icon) in T.CARDS:
        add(s, whoosh(0.25, True), t - 0.15, 0.25, -0.6); add(s, click(), t + 0.05, 0.35)
    a, b, c = T.SUB_BUTTON
    add(s, pop(), a, 0.4); add(s, click(), c, 0.6); add(s, ding(1568, 1.6), c + 0.08, 0.35)
    add(s, bleep(T.BLEEP[1] - T.BLEEP[0]), T.BLEEP[0], 1.0)
    add(s, whoosh(0.5, True), T.OUTRO[0] - 0.3, 0.35)
    return s


def write(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    m = build_music(); m /= np.abs(m).max() / 0.9
    write(os.path.join(HERE, "music.wav"), m)
    s = build_sfx(); s /= max(1e-9, np.abs(s).max() / 0.9)
    write(os.path.join(HERE, "sfx.wav"), s)
    print("ok", N / SR)
