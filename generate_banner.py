#!/usr/bin/env python3
"""
Driven by the unknown — animated ASCII banner for a GitHub profile README.

Timeline of one loop (~8.9 s, 16 fps):
  1. ASCII characters fly in from the right and assemble the text letter by
     letter (D -> r -> i -> ... -> n), each letter fading in its drop shadow.
  2. Short hold with a subtle character shimmer.
  3. An ASCII sword drops from the top edge and stabs into "un".
     A four-point cross flare bursts at the crossguard.
  4. A few characters around the blade are knocked loose and drift away slowly
     (~3 s) while the flare fades.
  5. Everything fades out and the loop restarts.

Usage:
  pip install pillow numpy scipy
  python generate_banner.py                      # -> assets/driven-by-the-unknown.gif
  python generate_banner.py --preview            # also writes a contact sheet of key frames
  python generate_banner.py --font path/to/Font.ttf --mono path/to/Mono-Bold.ttf
"""
import argparse, math, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# ----------------------------------------------------------------- config ---
W, H = 1200, 480          # banner size (px). GitHub shows it at ~830 px wide.
CW, CH = 4, 6             # character cell size (px)
COLS, ROWS = W // CW, H // CH
SS = 4                    # supersampling for vector masks
FPS = 16
LINE1, LINE2 = "Driven by", "the unknown"
FONT_PX = 205             # display font size

EDGE_CH = "@%#&$"         # dense characters for outlines
FILL_CH = "=+*x%o:"       # lighter characters for the body
SHADOW_CH = ".:;'"

# timeline (seconds)
T_FIRST = 0.15            # first letter starts flying
T_LETTER = 0.14           # stagger between letters
T_FLY = (0.9, 1.4)        # flight time range per character
T_HOLD_END = 5.0          # sword starts falling
T_DROP = 0.35             # fall duration
T_SCATTER = 3.0           # debris drift duration
T_FADE = 0.5              # fade out before looping
T_IMPACT = T_HOLD_END + T_DROP
T_END = T_IMPACT + T_SCATTER + T_FADE
N_FRAMES = int(round(T_END * FPS))


def find_mono(user_path):
    cands = [user_path] if user_path else []
    cands += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "C:/Windows/Fonts/consolab.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "DejaVuSansMono-Bold.ttf",
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    sys.exit("No monospace font found; pass one with --mono")


# --------------------------------------------------------------- sprites ---
class Sprites:
    """Pre-rendered character bitmaps, blitted with numpy (fast)."""
    def __init__(self, mono_path):
        self.font = ImageFont.truetype(mono_path, 6)
        self.cache = {}
        self.pad = 12
        self.buf = None

    def get(self, ch):
        s = self.cache.get(ch)
        if s is None:
            im = Image.new("L", (8, 9), 0)
            ImageDraw.Draw(im).text((0, 0), ch, font=self.font, fill=255)
            s = np.asarray(im, dtype=np.float32) / 255.0
            self.cache[ch] = s
        return s

    def new_frame(self):
        p = self.pad
        self.buf = np.zeros((H + 2 * p, W + 2 * p), np.float32)

    def blit(self, x, y, ch, b):
        if b <= 0.01:
            return
        x = int(round(x)) + self.pad
        y = int(round(y)) + self.pad
        s = self.get(ch)
        h, w = s.shape
        if x < 0 or y < 0 or x + w > self.buf.shape[1] or y + h > self.buf.shape[0]:
            return
        reg = self.buf[y:y + h, x:x + w]
        np.maximum(reg, s * b, out=reg)

    def frame(self):
        p = self.pad
        return self.buf[p:p + H, p:p + W]


# ----------------------------------------------------------------- masks ---
BW, BH = COLS * CW * SS, ROWS * CH * SS


def to_cells(mask_img):
    a = np.asarray(mask_img.resize((COLS * 4, ROWS * 4), Image.BOX), dtype=np.float32) / 255
    cov = a.reshape(ROWS, 4, COLS, 4).mean((1, 3))
    bl = ndimage.gaussian_filter(cov, 1.0)
    return cov, np.gradient(bl, axis=1), np.gradient(bl, axis=0)


# spotlight: a hot, blooming key light placed right of centre (near the sword)
SPOT = (0.66, 0.52)        # centre as fraction of W, H
SPOT_R = (0.30, 0.55)      # falloff radii as fraction of W, H


def spot(x_px, y_px):
    dx = (x_px / W - SPOT[0]) / SPOT_R[0]
    dy = (y_px / H - SPOT[1]) / SPOT_R[1]
    return math.exp(-(dx * dx + dy * dy) * 1.1)


def light(c, r=ROWS * 0.5):
    """Brightness multiplier for a cell: deep shadows far away, near-white in the hot spot."""
    return 0.32 + 1.05 * spot(c * CW, r * CH)


def cell_chars(cov, gx, gy, rnd, thr=0.09, boost=1.0):
    out = {}
    for r, c in zip(*np.nonzero(cov > thr)):
        v = cov[r, c]
        e = math.hypot(gx[r, c], gy[r, c])
        edge = e > 0.09 or v < 0.5
        rim = max(0.0, -gx[r, c]) * 2.2           # rim light on right-facing edges
        b = min(1.0, ((0.80 + 0.20 * v) * light(c, r) + rim * (0.4 + spot(c * CW, r * CH))) * boost)
        out[(int(r), int(c))] = [rnd.choice(EDGE_CH if edge else FILL_CH), b, edge]
    return out


def shadow_cells(cov, rnd, dx=-4, dy=3):
    sh = ndimage.shift(ndimage.gaussian_filter(cov, 0.8), (dy, dx), order=0)
    out = {}
    for r, c in zip(*np.nonzero((sh > 0.25) & (cov < 0.1))):
        out[(int(r), int(c))] = [rnd.choice(SHADOW_CH), 0.10 + 0.22 * light(c, r)]
    return out


def build_text(font_path):
    f = ImageFont.truetype(font_path, FONT_PX * SS)
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    w1 = probe.textlength(LINE1, font=f)
    w2 = probe.textlength(LINE2, font=f)
    t = probe.textlength("the ", font=f)
    gapx = BW * 0.075                       # keeps the sword column free of line 1
    x1 = (BW - (w1 + gapx - t + w2)) / 2
    x2 = x1 + w1 + gapx - t
    y1, y2 = BH * 0.02, BH * 0.40

    letters = []                            # (char, line, mask image)
    full = Image.new("L", (BW, BH), 0)
    fd = ImageDraw.Draw(full)
    for li, (txt, x, y) in enumerate([(LINE1, x1, y1), (LINE2, x2, y2)]):
        fd.text((x, y), txt, font=f, fill=255)
        for i, ch in enumerate(txt):
            if ch == " ":
                continue
            m = Image.new("L", (BW, BH), 0)
            # draw the prefix in black then the char so kerning matches the full line
            ax = x + probe.textlength(txt[:i], font=f)
            ImageDraw.Draw(m).text((ax, y), ch, font=f, fill=255)
            letters.append((ch, li, m, ax, probe.textlength(ch, font=f), y))
    return full, letters


# ----------------------------------------------------------------- sword ---
def sword_mask(cx_px, guard_y_px, s):
    """Ornate sword drawn as vector shapes (pixel units, s = px per unit)."""
    big = Image.new("L", (BW, BH), 0)
    d = ImageDraw.Draw(big)
    k = SS * s
    X = lambda v: cx_px * SS + v * k * 0.6
    top = guard_y_px * SS - 32 * k
    Y = lambda v: top + v * k
    # flame pommel
    d.polygon([(X(0), Y(2)), (X(-6), Y(-8)), (X(-2.5), Y(-3)), (X(-4), Y(-13)), (X(-1), Y(-5)),
               (X(0), Y(-16)), (X(1), Y(-5)), (X(4), Y(-13)), (X(2.5), Y(-3)), (X(6), Y(-8))], fill=255)
    d.ellipse([X(-3.5), Y(0), X(3.5), Y(6)], fill=255)
    # wrapped grip
    d.polygon([(X(-2.6), Y(5)), (X(2.6), Y(5)), (X(2.2), Y(26)), (X(-2.2), Y(26))], fill=255)
    for yy in range(7, 26, 4):
        d.line([X(-2.5), Y(yy), X(2.5), Y(yy + 2)], fill=0, width=max(1, int(k * 0.6)))
    # hooked crossguard
    for sg in (-1, 1):
        d.polygon([(X(0), Y(26)), (X(sg * 10), Y(25)), (X(sg * 17), Y(19)), (X(sg * 20), Y(13)),
                   (X(sg * 19), Y(21)), (X(sg * 22), Y(28)), (X(sg * 15), Y(26)), (X(sg * 9), Y(33)),
                   (X(sg * 3), Y(32))], fill=255)
        d.polygon([(X(sg * 3), Y(32)), (X(sg * 9), Y(36)), (X(sg * 7), Y(42)), (X(sg * 10), Y(46)),
                   (X(sg * 4), Y(44)), (X(sg * 2), Y(38))], fill=255)
    d.ellipse([X(-4.5), Y(26), X(4.5), Y(38)], fill=255)
    d.ellipse([X(-1.6), Y(29), X(1.6), Y(34)], fill=0)
    # blade with fuller and chipped edge
    d.polygon([(X(-2.6), Y(38)), (X(-3.6), Y(48)), (X(-3.4), Y(135)), (X(0), Y(150)),
               (X(3.4), Y(135)), (X(3.6), Y(48)), (X(2.6), Y(38))], fill=255)
    d.line([X(0), Y(48), X(0), Y(130)], fill=0, width=max(1, int(k * 0.7)))
    rnd = random.Random(1)
    for _ in range(14):
        yy = rnd.uniform(52, 120)
        d.ellipse([X(-3.6), Y(yy), X(-3.6 + rnd.uniform(.5, 1.5)), Y(yy + rnd.uniform(.5, 2))], fill=0)
    blade_half_px = 3.6 * k * 0.6 / SS
    return big, blade_half_px


# ----------------------------------------------------------------- flare ---
def draw_flare(sp, px, py, strength, rnd):
    if strength <= 0.02:
        return
    # soft halo
    for _ in range(int(520 * strength)):
        rr = abs(rnd.gauss(0, 38 * strength))
        a = rnd.uniform(0, 2 * math.pi)
        b = max(0.0, 1 - rr / (90 * strength)) * 0.7 * min(1, strength * 1.5)
        sp.blit(px + rr * math.cos(a), py + rr * math.sin(a) * 0.8, rnd.choice(".:'"), b)
    # four-point cross: long horizontal, short vertical
    for ux, uy, Lr, chs in [(1, 0, 330, "-=~"), (-1, 0, 330, "-=~"), (0, 1, 120, "|!:"), (0, -1, 120, "|!:")]:
        Lr *= strength
        step = CW if ux else CH
        n = max(1, int(Lr / step))
        for i in range(n):
            t = i / n
            b = (1 - t) ** 1.6 * min(1, strength * 1.3)
            for w in (-1, 0, 1):
                if w and t > 0.35:
                    continue
                x = px + ux * i * step + (0 if ux else w * CW)
                y = py + uy * i * step + (w * CH if ux else 0)
                ch = rnd.choice(chs) if t > 0.15 else rnd.choice("@#*")
                sp.blit(x, y, ch, b * (1 if w == 0 else .55))
    for ux, uy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        n = int(7 * strength)
        for i in range(1, n):
            sp.blit(px + ux * i * CW * 1.3, py + uy * i * CH * 0.75, ".", (1 - i / max(n, 1)) * 0.6)
    core = min(1.0, strength * 1.2)
    for i in range(-2, 3):
        for j in range(-1, 2):
            sp.blit(px + i * CW, py + j * CH, "@", core)


# ---------------------------------------------------------------- easing ---
def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def clamp01(t):
    return max(0.0, min(1.0, t))


# ------------------------------------------------------------------ build ---
def build_scene(font_path):
    rnd = random.Random(5)
    full, letters = build_text(font_path)
    cov, gx, gy = to_cells(full)
    text = cell_chars(cov, gx, gy, rnd)
    shadow = shadow_cells(cov, rnd)

    # assign each cell to the letter that covers it most
    covs = np.stack([to_cells(m)[0] for (_, _, m, *_) in letters])
    # glyphs drawn alone can differ slightly from the full line (contextual forms),
    # so give every cell to the *nearest* letter instead of only exact overlaps
    owner = np.stack([ndimage.gaussian_filter(cv, 2.5) for cv in covs]).argmax(0)
    sh_owner = ndimage.shift(owner, (3, -4), order=0, mode="nearest")

    n_letters = len(letters)
    rng = random.Random(21)
    chars = []          # per text cell: dict with timing + flight path
    for (r, c), (ch, b, edge) in text.items():
        li = int(owner[r, c])
        start = T_FIRST + li * T_LETTER + rng.uniform(0, 0.35)
        chars.append(dict(r=r, c=c, ch=ch, b=b, edge=edge, letter=li, t0=start,
                          dur=rng.uniform(*T_FLY), sx=W + rng.uniform(10, 260),
                          sy=r * CH + rng.uniform(-24, 24)))
    shadows = []
    for (r, c), (ch, b) in shadow.items():
        li = int(sh_owner[r, c])
        shadows.append(dict(r=r, c=c, ch=ch, b=b, t0=T_FIRST + li * T_LETTER + 1.25))

    # sword target: centre of "un"
    l2 = [l for l in letters if l[1] == 1]
    u = [l for l in l2 if l[0] == "u"][0]
    n = l2[l2.index(u) + 1]
    ucx_px = (u[3] + n[3] + n[4]) / 2 / SS
    bb = ImageDraw.Draw(Image.new("L", (10, 10))).textbbox((0, 0), LINE2,
                                                          font=ImageFont.truetype(font_path, FONT_PX * SS))
    top2_px = (u[5] + bb[1]) / SS
    guard_y = top2_px - 4
    s = (guard_y - 14) / 48.0
    smask, blade_half = sword_mask(ucx_px, guard_y, s)
    scov, sgx, sgy = to_cells(smask)
    sword = cell_chars(scov, sgx, sgy, random.Random(9), boost=1.3)
    sword_sh = shadow_cells(scov, random.Random(10), -5, 3)

    # characters knocked loose by the blade
    un0, un1 = u[3] / SS / CW, (n[3] + n[4]) / SS / CW
    ucx = ucx_px / CW
    blade_cols = blade_half / CW
    hr = random.Random(11)
    for ch in chars:
        dc = ch["c"] - ucx
        ch["hidden"] = False
        ch["debris"] = None
        if not (un0 <= ch["c"] <= un1 and ch["r"] * CH >= top2_px - CH):
            continue
        if abs(dc) <= blade_cols:
            ch["hidden"] = True             # behind the blade
            if hr.random() < 0.6:
                ch["debris"] = True
        elif abs(dc) < 10 and hr.random() < 0.75 / (1 + abs(dc) / 5):
            ch["hidden"] = True
            ch["debris"] = True
        if ch["debris"]:
            side = 1 if dc >= 0 else -1
            ang = hr.uniform(-1.0, 0.45)                    # mostly sideways, a bit upward/downward
            spd = hr.uniform(25, 150)                       # total drift distance (px)
            ch["vx"] = side * math.cos(ang) * spd
            ch["vy"] = math.sin(ang) * spd * 0.55 + hr.uniform(4, 30)
            ch["spin"] = hr.random() < 0.5
            ch["life"] = hr.uniform(0.75, 1.0)

    # halftone dust: the light itself shows up as scattered dots on the background
    drng = random.Random(33)
    dust = []
    for r in range(ROWS):
        for c in range(COLS):
            I = spot(c * CW, r * CH)
            if drng.random() < 0.55 * I ** 1.7:
                dust.append((r, c, drng.choice(".:'.`"), I * drng.uniform(0.25, 0.75)))
    return dict(dust=dust, chars=chars, shadows=shadows, sword=sword, sword_sh=sword_sh,
                guard=(ucx_px, guard_y + 2), n_letters=n_letters)


# ----------------------------------------------------------------- render ---
def render_frame(sp, scene, fi):
    t = fi / FPS
    rnd = random.Random(1000 + fi)
    sp.new_frame()

    fade = 1.0
    if t > T_END - T_FADE:
        fade = 1 - clamp01((t - (T_END - T_FADE)) / T_FADE)

    shake = 0.0
    if T_IMPACT <= t < T_IMPACT + 0.2:
        shake = (1 - (t - T_IMPACT) / 0.2) * 2.5
    jx = rnd.uniform(-shake, shake)
    jy = rnd.uniform(-shake, shake)

    # halftone glow, fades in with the text and flares on impact
    glow = clamp01((t - 1.0) / 2.5)
    if t >= T_IMPACT:
        glow *= 1 + 0.6 * math.exp(-(t - T_IMPACT) / 0.9)
    if glow > 0:
        for r, c, ch, b in scene["dust"]:
            sp.blit(c * CW, r * CH, ch, min(1, b * glow) * fade)

    # shadows of settled letters
    for s in scene["shadows"]:
        a = clamp01((t - s["t0"]) / 0.4)
        if a > 0:
            sp.blit(s["c"] * CW + jx, s["r"] * CH + jy, s["ch"], s["b"] * a * fade)

    # sword shadow + sword (falls from above the frame)
    sword_on = t >= T_HOLD_END
    if sword_on:
        p = clamp01((t - T_HOLD_END) / T_DROP)
        dy = -(1 - p * p) * (H * 1.15)             # ease-in: accelerates into the stab
        for (r, c), (ch, b) in scene["sword_sh"].items():
            if t >= T_IMPACT:
                sp.blit(c * CW, r * CH + dy, ch, b * fade)

    # text characters
    shimmer = (fi // 2)
    for ch in scene["chars"]:
        if t < ch["t0"]:
            continue
        k = clamp01((t - ch["t0"]) / ch["dur"])
        tx, ty = ch["c"] * CW, ch["r"] * CH
        glyph = ch["ch"]
        if k >= 1 and random.Random(hash((ch["r"], ch["c"], shimmer))).random() < 0.03:
            glyph = random.Random(hash((ch["c"], ch["r"], shimmer))).choice(EDGE_CH if ch["edge"] else FILL_CH)
        if k < 1:
            e = ease_out_cubic(k)
            x = ch["sx"] + (tx - ch["sx"]) * e
            y = ch["sy"] + (ty - ch["sy"]) * e
            b = ch["b"] * (0.35 + 0.65 * e)
            sp.blit(x, y, glyph, b * fade)
            speed = (1 - k) ** 2
            if speed > 0.15:                     # motion streak
                for tr in range(1, 3):
                    sp.blit(x + tr * CW * 1.6, y, "-", b * 0.35 * speed / tr * fade)
            continue
        if t >= T_IMPACT and ch["hidden"]:
            if not ch["debris"]:
                continue
            q = clamp01((t - T_IMPACT) / T_SCATTER)
            travel = 1 - math.exp(-3.0 * q)      # quick kick, slow drift
            travel /= (1 - math.exp(-3.0))
            x = tx + ch["vx"] * travel
            y = ty + ch["vy"] * travel + 10 * q * q
            life = clamp01(1 - (q / ch["life"]) ** 1.8)
            g = glyph
            if ch["spin"] and (fi % 4) < 2:
                g = rnd.choice("/\\|-")
            sp.blit(x, y, g, min(1.0, ch["b"] * 1.15) * life ** 0.7 * fade)
            continue
        sp.blit(tx + jx, ty + jy, glyph, ch["b"] * fade)

    if sword_on:
        for (r, c), (ch, b, edge) in scene["sword"].items():
            y = r * CH + dy
            if y < -CH or y > H:
                continue
            g = ch
            if random.Random(hash((r, c, shimmer))).random() < 0.03:
                g = random.Random(hash((c, r, shimmer))).choice(EDGE_CH if edge else FILL_CH)
            sp.blit(c * CW, y, g, b * fade)
        if t >= T_IMPACT:
            q = t - T_IMPACT
            strength = 1.3 * math.exp(-q / 1.1)
            if q < 0.08:
                strength *= 0.6 + 5 * q          # tiny ramp-up on the hit
            gx, gy = scene["guard"]
            draw_flare(sp, gx, gy, strength * fade, rnd)

    f = sp.frame()
    bloom = ndimage.gaussian_filter(f, 7)        # soft glow around bright areas
    return np.clip(f * 1.1 + bloom * 1.6, 0, 1)


def save_gif(frames, path):
    levels = 32
    pal = []
    for i in range(levels):
        v = int(round(255 * (i / (levels - 1)) ** 0.9))
        pal += [v, v, v]
    pal += [0, 0, 0] * (256 - levels)
    ims = []
    for f in frames:
        idx = np.round(f * (levels - 1)).astype(np.uint8)
        im = Image.fromarray(idx, "P")
        im.putpalette(pal)
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / FPS),
                loop=0, optimize=False, disposal=1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default=os.path.join(here, "fonts", "Aviorte_Personal_Use_only.ttf"))
    ap.add_argument("--mono", default=None, help="small monospace font used for the ASCII cells")
    ap.add_argument("--out", default=os.path.join(here, "assets", "driven-by-the-unknown.gif"))
    ap.add_argument("--preview", action="store_true", help="also write a contact sheet of key frames")
    a = ap.parse_args()
    if not os.path.exists(a.font):
        sys.exit(f"Display font not found: {a.font}")

    sp = Sprites(find_mono(a.mono))
    scene = build_scene(a.font)
    frames = []
    for fi in range(N_FRAMES):
        frames.append(render_frame(sp, scene, fi))
        if fi % 20 == 0:
            print(f"frame {fi}/{N_FRAMES}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_gif(frames, a.out)
    print("wrote", a.out, f"({os.path.getsize(a.out) / 1e6:.2f} MB, {N_FRAMES} frames @ {FPS} fps)")

    if a.preview:
        picks = [(0.9, "fly-in"), (2.6, "building"), (4.8, "complete"), (T_IMPACT - 0.1, "sword drop"),
                 (T_IMPACT + 0.15, "impact"), (T_IMPACT + 1.5, "debris drift"), (T_IMPACT + 2.9, "end of drift")]
        sheet = Image.new("L", (W, H * len(picks)), 0)
        for i, (tt, _) in enumerate(picks):
            fi = min(N_FRAMES - 1, int(tt * FPS))
            sheet.paste(Image.fromarray((frames[fi] * 255).astype(np.uint8)), (0, i * H))
        p = os.path.splitext(a.out)[0] + "-keyframes.png"
        sheet.save(p)
        print("wrote", p)


if __name__ == "__main__":
    main()
