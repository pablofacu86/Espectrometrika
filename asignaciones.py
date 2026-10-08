"""Indicative assignment of spectral regions to functional groups / protons / overtones,
with simple structure drawings (the vibrating atoms/bonds, or the protons that resonate, in red).

IMPORTANT: every value here is a TYPICAL, INDICATIVE range. Real positions depend on the matrix,
solvent, pH, temperature, hydrogen bonding and neighbouring groups. Always check the literature.
Database units: NIR in nm; FT-MIR and Raman in cm-1; 1H NMR in ppm.
"""
import numpy as np
import pandas as pd

ROJO, NEGRO = "#D62828", "#1b1b1b"

# ----------------------------------------------------------------------------------------------
# STRUCTURES (screen coordinates: x to the right, y DOWN). Hydrogens that can be highlighted are
# explicit atoms labelled "H".
# ----------------------------------------------------------------------------------------------
_E = {}


def _s(key, atoms, bonds):
    _E[key] = {"atoms": {n: (l, x, y) for n, l, x, y in atoms},
               "bonds": [(a, b, (o[0] if o else 1)) for a, b, *o in bonds]}


_s("alcohol", [("R", "R", 0, 0), ("O", "O", 1, .6), ("H", "H", 2, 0)], [("R", "O"), ("O", "H")])
_s("phenol", [("R", "Ar", 0, 0), ("O", "O", 1, .6), ("H", "H", 2, 0)], [("R", "O"), ("O", "H")])
_s("water", [("H1", "H", 0, 0), ("O", "O", 1, .6), ("H2", "H", 2, 0)], [("H1", "O"), ("O", "H2")])
_s("acid", [("R", "R", 0, .6), ("C", "C", 1, 0), ("Od", "O", 1, -1), ("O", "O", 2, .6), ("H", "H", 3, 0)],
   [("R", "C"), ("C", "Od", 2), ("C", "O"), ("O", "H")])
_s("ester", [("R", "R", 0, .6), ("C", "C", 1, 0), ("Od", "O", 1, -1), ("O", "O", 2, .6), ("Rp", "R'", 3, 0)],
   [("R", "C"), ("C", "Od", 2), ("C", "O"), ("O", "Rp")])
_s("amide", [("R", "R", 0, .6), ("C", "C", 1, 0), ("Od", "O", 1, -1), ("N", "N", 2, .6),
             ("H1", "H", 3, 0), ("H2", "H", 2, 1.6)],
   [("R", "C"), ("C", "Od", 2), ("C", "N"), ("N", "H1"), ("N", "H2")])
_s("peptide", [("Ca", "C", 0, .6), ("C", "C", 1, 0), ("Od", "O", 1, -1), ("N", "N", 2, .6), ("H", "H", 2, 1.6),
               ("Cb", "C", 3, 0)],
   [("Ca", "C"), ("C", "Od", 2), ("C", "N"), ("N", "H"), ("N", "Cb")])
_s("amine", [("R", "R", 0, 0), ("N", "N", 1, .6), ("H1", "H", 2, 0), ("H2", "H", 1, 1.6)],
   [("R", "N"), ("N", "H1"), ("N", "H2")])
_s("nitrile", [("R", "R", 0, 0), ("C", "C", 1, 0), ("N", "N", 2, 0)], [("R", "C"), ("C", "N", 3)])
_s("aldehyde", [("R", "R", 0, .6), ("C", "C", 1, 0), ("O", "O", 1, -1), ("H", "H", 2, .6)],
   [("R", "C"), ("C", "O", 2), ("C", "H")])
_s("ketone", [("R", "R", 0, .6), ("C", "C", 1, 0), ("O", "O", 1, -1), ("Rp", "R'", 2, .6)],
   [("R", "C"), ("C", "O", 2), ("C", "Rp")])
_s("ether", [("R", "R", 0, 0), ("O", "O", 1, .6), ("Rp", "R'", 2, 0)], [("R", "O"), ("O", "Rp")])
_s("methoxy", [("R", "R", 0, 0), ("O", "O", 1, .6), ("C", "C", 2, 0), ("H1", "H", 2, -1), ("H2", "H", 3, .5),
               ("H3", "H", 2, 1)],
   [("R", "O"), ("O", "C"), ("C", "H1"), ("C", "H2"), ("C", "H3")])
_s("methyl", [("R", "R", 0, 0), ("C", "C", 1, .6), ("H1", "H", 1, -.4), ("H2", "H", 2, 0), ("H3", "H", 1, 1.6)],
   [("R", "C"), ("C", "H1"), ("C", "H2"), ("C", "H3")])
_s("methylene", [("R1", "R", 0, 0), ("C", "C", 1, .6), ("R2", "R'", 2, 0), ("H1", "H", .4, 1.5), ("H2", "H", 1.6, 1.5)],
   [("R1", "C"), ("C", "R2"), ("C", "H1"), ("C", "H2")])
_s("alkene_cis", [("R1", "R", 0, 0), ("C1", "C", 1, .6), ("C2", "C", 2, .6), ("R2", "R'", 3, 0),
                  ("H1", "H", 1, 1.6), ("H2", "H", 2, 1.6)],
   [("R1", "C1"), ("C1", "C2", 2), ("C2", "R2"), ("C1", "H1"), ("C2", "H2")])
_s("alkene_trans", [("R1", "R", 0, 0), ("C1", "C", 1, .6), ("C2", "C", 2, .6), ("R2", "R'", 3, 1.2),
                    ("H1", "H", 1, 1.6), ("H2", "H", 2, -.4)],
   [("R1", "C1"), ("C1", "C2", 2), ("C2", "R2"), ("C1", "H1"), ("C2", "H2")])
_s("alkyne", [("R", "R", 0, 0), ("C1", "C", 1, 0), ("C2", "C", 2, 0), ("H", "H", 3, 0)],
   [("R", "C1"), ("C1", "C2", 3), ("C2", "H")])
# benzene: hexagon with a substituent R and 5 H
_h = np.sqrt(3) / 2
_ar = [("c%d" % i, "C", np.cos(np.pi / 3 * i - np.pi / 2) * 1.0, np.sin(np.pi / 3 * i - np.pi / 2) * 1.0) for i in range(6)]
_ar += [("R", "R", 0, -2.0)]
_ar += [("h%d" % i, "H", np.cos(np.pi / 3 * i - np.pi / 2) * 1.95, np.sin(np.pi / 3 * i - np.pi / 2) * 1.95) for i in range(1, 6)]
_s("aromatic", _ar, [("c%d" % i, "c%d" % ((i + 1) % 6), 2 if i % 2 == 0 else 1) for i in range(6)]
   + [("c0", "R")] + [("c%d" % i, "h%d" % i) for i in range(1, 6)])
_s("acetyl", [("Me", "C", 0, .6), ("H1", "H", -.8, 0), ("H2", "H", 0, 1.6), ("H3", "H", -.8, 1.3),
              ("C", "C", 1, 0), ("O", "O", 1, -1), ("R", "R", 2, .6)],
   [("Me", "C"), ("C", "O", 2), ("C", "R"), ("Me", "H1"), ("Me", "H2"), ("Me", "H3")])
_s("alpha_ch2", [("R", "R", 0, 0), ("Ca", "C", 1, .6), ("H1", "H", .6, 1.6), ("H2", "H", 1.7, 1.5),
                 ("C", "C", 2, 0), ("O", "O", 2, -1), ("Rp", "R'", 3, .6)],
   [("R", "Ca"), ("Ca", "C"), ("C", "O", 2), ("C", "Rp"), ("Ca", "H1"), ("Ca", "H2")])
_s("allylic", [("R", "R", 0, 0), ("Ca", "C", 1, .6), ("H1", "H", .5, 1.6), ("H2", "H", 1.6, 1.6),
               ("C1", "C", 2, 0), ("C2", "C", 3, .6), ("Rp", "R'", 4, 0)],
   [("R", "Ca"), ("Ca", "C1"), ("C1", "C2", 2), ("C2", "Rp"), ("Ca", "H1"), ("Ca", "H2")])
_s("bisallylic", [("C1", "C", 0, 0), ("C2", "C", 1, .6), ("Cm", "C", 2, 0), ("H1", "H", 1.5, 1.0), ("H2", "H", 2.0, -1.0),
                  ("C3", "C", 3, .6), ("C4", "C", 4, 0)],
   [("C1", "C2", 2), ("C2", "Cm"), ("Cm", "C3"), ("C3", "C4", 2), ("Cm", "H1"), ("Cm", "H2")])
_s("vinyl_H", [("R", "R", 0, 0), ("C1", "C", 1, .6), ("C2", "C", 2, .6), ("R2", "R'", 3, 0),
               ("H1", "H", 1, 1.6), ("H2", "H", 2, 1.6)],
   [("R", "C1"), ("C1", "C2", 2), ("C2", "R2"), ("C1", "H1"), ("C2", "H2")])
_s("glycerol", [("C1", "C", 0, 0), ("C2", "C", 1, .6), ("C3", "C", 2, 0),
                ("H1", "H", -.5, -.95), ("H2", "H", .55, -.95), ("H4", "H", 1.45, -.95), ("H5", "H", 2.55, -.95),
                ("H3", "H", .3, 1.55), ("O1", "RCOO", -1.5, .45), ("O2", "RCOO", 3.5, .45), ("O3", "RCOO", 1.9, 1.6)],
   [("C1", "C2"), ("C2", "C3"), ("C1", "H1"), ("C1", "H2"), ("C3", "H4"), ("C3", "H5"), ("C2", "H3"),
    ("C1", "O1"), ("C3", "O2"), ("C2", "O3")])
_s("sugar", [("C1", "C", 0, 0), ("C2", "C", 1, .6), ("C3", "C", 2.2, .3), ("C4", "C", 2.2, -1.0), ("C5", "C", 1, -1.6),
             ("O5", "O", 0, -1.2), ("H1", "H", -.9, .5), ("O1", "O", 0, 1.1), ("H2", "H", 1, 1.6),
             ("H3", "H", 3.2, .6), ("H4", "H", 3.2, -1.3), ("H5", "H", 1, -2.6)],
   [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C5", "O5"), ("O5", "C1"), ("C1", "H1"),
    ("C1", "O1"), ("C2", "H2"), ("C3", "H3"), ("C4", "H4"), ("C5", "H5")])
_s("choline", [("N", "N", 1, 0), ("R", "R", 0, .8), ("M1", "C", 2, .8), ("M2", "C", 2, -.9), ("M3", "C", 0, -.9),
               ("Ha", "H", 3, 1.5), ("Hb", "H", 3, .2), ("Hc", "H", 3, -1.7), ("Hd", "H", 3, -.4),
               ("He", "H", -1, -.2), ("Hf", "H", -1, -1.5)],
   [("N", "R"), ("N", "M1"), ("N", "M2"), ("N", "M3"), ("M1", "Ha"), ("M1", "Hb"), ("M2", "Hc"), ("M2", "Hd"),
    ("M3", "He"), ("M3", "Hf")])
_s("silicate", [("Si1", "Si", 0, 0), ("O", "O", 1, .6), ("Si2", "Si", 2, 0)], [("Si1", "O"), ("O", "Si2")])
_s("clayOH", [("Al", "Al", 0, 0), ("O", "O", 1, .6), ("H", "H", 2, 0)], [("Al", "O"), ("O", "H")])
_s("carbonate", [("C", "C", 1, 0), ("O1", "O", 1, -1.1), ("O2", "O", 0, .7), ("O3", "O", 2, .7)],
   [("C", "O1", 2), ("C", "O2"), ("C", "O3")])
_s("phosphate", [("P", "P", 1, 0), ("O", "O", 1, -1), ("O1", "O", 0, .7), ("O2", "O", 2, .7), ("R", "R", 1, 1.3)],
   [("P", "O", 2), ("P", "O1"), ("P", "O2"), ("P", "R")])
_s("thiol", [("R", "R", 0, 0), ("S", "S", 1, .6), ("H", "H", 2, 0)], [("R", "S"), ("S", "H")])
_s("disulfide", [("R", "R", 0, 0), ("S1", "S", 1, .6), ("S2", "S", 2, 0), ("Rp", "R'", 3, .6)],
   [("R", "S1"), ("S1", "S2"), ("S2", "Rp")])
_s("polyene", [("R", "R", 0, .6), ("C1", "C", 1, 0), ("C2", "C", 2, .6), ("C3", "C", 3, 0), ("C4", "C", 4, .6),
               ("C5", "C", 5, 0), ("C6", "C", 6, .6), ("Rp", "R'", 7, 0)],
   [("R", "C1"), ("C1", "C2", 2), ("C2", "C3"), ("C3", "C4", 2), ("C4", "C5"), ("C5", "C6", 2), ("C6", "Rp")])
_s("formate", [("H", "H", 0, .6), ("C", "C", 1, 0), ("O1", "O", 1, -1), ("O2", "O", 2, .6)],
   [("H", "C"), ("C", "O1", 2), ("C", "O2")])
_s("alkyl_halide", [("R", "R", 0, 0), ("C", "C", 1, .6), ("X", "Cl", 2, 0)], [("R", "C"), ("C", "X")])
_s("imidazole", [("N1", "N", 0, 0), ("C2", "C", 1, -.5), ("N3", "N", 2, 0), ("C4", "C", 1.6, 1.0), ("C5", "C", .4, 1.0),
                 ("H2", "H", 1, -1.5), ("H4", "H", 2.2, 1.8), ("H5", "H", -.2, 1.8)],
   [("N1", "C2"), ("C2", "N3", 2), ("N3", "C4"), ("C4", "C5", 2), ("C5", "N1"), ("C2", "H2"), ("C4", "H4"), ("C5", "H5")])

ESTRUCTURAS = _E


def svg_estructura(clave, atomos_rojos=(), enlaces_rojos=(), px=34, titulo=None):
    """SVG of a structure; atoms/bonds named in the highlight lists are drawn in red, all else black."""
    e = _E[clave]
    at = e["atoms"]
    xs = [v[1] for v in at.values()]; ys = [v[2] for v in at.values()]
    pad = 0.9
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    W, H = (x1 - x0) * px, (y1 - y0) * px
    P = lambda x, y: ((x - x0) * px, (y - y0) * px)
    rojos = set(atomos_rojos)
    enl_r = {frozenset(b) for b in enlaces_rojos}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}">',
           f'<rect width="100%" height="100%" fill="white" rx="6"/>']
    rad = 0.3
    for a, b, o in e["bonds"]:
        la, xa, ya = at[a]; lb, xb, yb = at[b]
        dx, dy = xb - xa, yb - ya
        L = float(np.hypot(dx, dy)) or 1.0
        ux, uy = dx / L, dy / L
        sa = rad if True else 0
        sb = rad if True else 0
        xa2, ya2, xb2, yb2 = xa + ux * sa, ya + uy * sa, xb - ux * sb, yb - uy * sb
        col = ROJO if frozenset((a, b)) in enl_r else NEGRO
        w = 2.6 if col == ROJO else 1.6
        nx, ny = -uy, ux
        offs = {1: [0], 2: [-.09, .09], 3: [-.16, 0, .16]}[o]
        for f in offs:
            p1 = P(xa2 + nx * f, ya2 + ny * f); p2 = P(xb2 + nx * f, yb2 + ny * f)
            out.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                       f'stroke="{col}" stroke-width="{w}" stroke-linecap="round"/>')
    for n, (lab, x, y) in at.items():
        px_, py_ = P(x, y)
        col = ROJO if n in rojos else NEGRO
        wt = "700" if n in rojos else "400"
        out.append(f'<text x="{px_:.1f}" y="{py_ + 5:.1f}" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" '
                   f'font-size="{px * 0.5:.0f}" font-weight="{wt}" fill="{col}">{lab}</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------------------------
# DATABASE. (lo, hi, group / assignment, vibration or type of H, structure, red atoms, red bonds)
# ----------------------------------------------------------------------------------------------
def _e(lo, hi, grupo, modo, est, ra=(), rb=()):
    return dict(lo=lo, hi=hi, grupo=grupo, modo=modo, estructura=est, rojos=tuple(ra), enlaces=tuple(rb))


# --- FT-MIR (cm-1) -------------------------------------------------------------------------------
MIR = [
    _e(3600, 3200, "Alcohols / phenols / water (H-bonded)", "O–H stretching (broad)", "alcohol", ["O", "H"], [("O", "H")]),
    _e(3600, 3500, "Free O–H / clay minerals (structural OH)", "O–H stretching", "clayOH", ["O", "H"], [("O", "H")]),
    _e(3500, 3300, "Amines / amides (N–H)", "N–H stretching", "amide", ["N", "H1"], [("N", "H1")]),
    _e(3330, 3270, "Terminal alkyne", "≡C–H stretching", "alkyne", ["C2", "H"], [("C2", "H")]),
    _e(3100, 3000, "Aromatic / alkene C–H", "=C–H stretching", "alkene_cis", ["C1", "H1"], [("C1", "H1")]),
    _e(3000, 2950, "Methyl (–CH₃)", "C–H asymmetric stretching", "methyl", ["C", "H1", "H2", "H3"], [("C", "H1"), ("C", "H2"), ("C", "H3")]),
    _e(2935, 2915, "Methylene (–CH₂–), lipids / long chains", "C–H asymmetric stretching", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(2875, 2845, "Methyl / methylene", "C–H symmetric stretching", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(2830, 2695, "Aldehyde", "C–H stretching (Fermi doublet)", "aldehyde", ["C", "H"], [("C", "H")]),
    _e(2260, 2210, "Nitrile", "C≡N stretching", "nitrile", ["C", "N"], [("C", "N")]),
    _e(2260, 2100, "Alkyne", "C≡C stretching", "alkyne", ["C1", "C2"], [("C1", "C2")]),
    _e(1760, 1735, "Ester (triglycerides, esters)", "C=O stretching", "ester", ["C", "Od"], [("C", "Od")]),
    _e(1740, 1720, "Aldehyde", "C=O stretching", "aldehyde", ["C", "O"], [("C", "O")]),
    _e(1725, 1700, "Carboxylic acid", "C=O stretching", "acid", ["C", "Od"], [("C", "Od")]),
    _e(1725, 1705, "Ketone", "C=O stretching", "ketone", ["C", "O"], [("C", "O")]),
    _e(1690, 1630, "Amide I (proteins)", "C=O stretching (mainly)", "peptide", ["C", "Od"], [("C", "Od")]),
    _e(1680, 1620, "Alkene", "C=C stretching", "alkene_cis", ["C1", "C2"], [("C1", "C2")]),
    _e(1660, 1620, "Water", "H–O–H bending", "water", ["H1", "O", "H2"], [("H1", "O"), ("O", "H2")]),
    _e(1600, 1585, "Aromatic ring", "C=C ring stretching", "aromatic", ["c0", "c1", "c2", "c3", "c4", "c5"],
       [("c0", "c1"), ("c2", "c3"), ("c4", "c5")]),
    _e(1580, 1510, "Amide II (proteins)", "N–H bending + C–N stretching", "peptide", ["N", "H", "C"], [("N", "H"), ("C", "N")]),
    _e(1520, 1470, "Aromatic ring", "C=C ring stretching", "aromatic", ["c1", "c2", "c3", "c4"], [("c1", "c2"), ("c3", "c4")]),
    _e(1470, 1440, "Methylene / methyl", "C–H bending (scissoring / asymmetric)", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(1385, 1365, "Methyl", "C–H symmetric bending ('umbrella')", "methyl", ["C", "H1", "H2", "H3"], [("C", "H1"), ("C", "H2"), ("C", "H3")]),
    _e(1420, 1395, "Carboxylate / acid", "C–O–H in-plane bending / COO⁻ symmetric", "acid", ["C", "O", "H"], [("C", "O"), ("O", "H")]),
    _e(1350, 1250, "Amide III / C–N", "C–N stretching + N–H bending", "peptide", ["C", "N", "H"], [("C", "N"), ("N", "H")]),
    _e(1300, 1200, "Esters / phenols / ethers", "C–O stretching (asymmetric)", "ester", ["C", "O"], [("C", "O")]),
    _e(1250, 1210, "Phosphates / phosphodiesters", "P=O stretching", "phosphate", ["P", "O"], [("P", "O")]),
    _e(1180, 1140, "Esters (triglycerides)", "C–O stretching", "ester", ["C", "O"], [("C", "O")]),
    _e(1160, 1000, "Carbohydrates / alcohols / ethers", "C–O and C–C stretching", "sugar", ["C1", "O1", "C5", "O5"], [("C1", "O1"), ("C5", "O5")]),
    _e(1100, 1000, "Silicates / clays (soil)", "Si–O stretching", "silicate", ["Si1", "O", "Si2"], [("Si1", "O"), ("O", "Si2")]),
    _e(1100, 1030, "Alcohols (primary / secondary)", "C–O stretching", "alcohol", ["R", "O"], [("R", "O")]),
    _e(1100, 1000, "Phosphates", "P–O stretching", "phosphate", ["P", "O1", "O2"], [("P", "O1"), ("P", "O2")]),
    _e(975, 955, "trans double bond (trans fats)", "=C–H out-of-plane bending", "alkene_trans", ["C1", "C2", "H1", "H2"], [("C1", "H1"), ("C2", "H2")]),
    _e(900, 675, "Aromatic ring substitution pattern", "C–H out-of-plane bending", "aromatic", ["h1", "c1", "h2", "c2"], [("c1", "h1"), ("c2", "h2")]),
    _e(730, 715, "Long aliphatic chains", "CH₂ rocking", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(850, 550, "Alkyl halides", "C–Cl stretching", "alkyl_halide", ["C", "X"], [("C", "X")]),
]

# --- NIR (nm): overtones and combination bands ---------------------------------------------------
NIR = [
    _e(1140, 1230, "Methyl / methylene / aromatic C–H", "C–H 2nd overtone", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(1350, 1400, "Alcohols / free O–H, C–H combination", "O–H 1st overtone / C–H combination", "alcohol", ["O", "H"], [("O", "H")]),
    _e(1400, 1470, "Water, O–H (also phenols, cellulose)", "O–H 1st overtone", "water", ["H1", "O", "H2"], [("H1", "O"), ("O", "H2")]),
    _e(1480, 1550, "Amides / amines / H-bonded O–H", "N–H 1st overtone", "amide", ["N", "H1", "H2"], [("N", "H1"), ("N", "H2")]),
    _e(1640, 1780, "Methyl / methylene / aromatic C–H (lipids, hydrocarbons)", "C–H 1st overtone", "methyl", ["C", "H1", "H2", "H3"], [("C", "H1"), ("C", "H2"), ("C", "H3")]),
    _e(1890, 1960, "Water (and moisture)", "O–H stretching + bending combination", "water", ["H1", "O", "H2"], [("H1", "O"), ("O", "H2")]),
    _e(1900, 2000, "Carbonyl (esters / acids), weak", "C=O stretching 2nd overtone", "ester", ["C", "Od"], [("C", "Od")]),
    _e(2030, 2100, "Amides / proteins", "N–H stretching + amide II combination", "peptide", ["N", "H"], [("N", "H")]),
    _e(2080, 2150, "Carbohydrates / alcohols", "O–H bending + C–O stretching combination", "sugar", ["C1", "O1"], [("C1", "O1")]),
    _e(2150, 2220, "Clay minerals (kaolinite / Al–OH), proteins", "Al–OH bending + O–H stretching / amide combination", "clayOH", ["Al", "O", "H"], [("Al", "O"), ("O", "H")]),
    _e(2250, 2300, "Cellulose / carbohydrates / Mg–OH", "O–H + C–C / C–H combination", "sugar", ["C2", "H2"], [("C2", "H2")]),
    _e(2300, 2380, "Methylene / methyl (lipids), carbonates (≈2340)", "C–H stretching + deformation combination", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(2330, 2350, "Carbonates (soil, calcite)", "CO₃²⁻ combination", "carbonate", ["C", "O1", "O2", "O3"], [("C", "O1"), ("C", "O2"), ("C", "O3")]),
    _e(2380, 2500, "Cellulose / hydrocarbons", "C–H + C–C combination", "methylene", ["C", "H1"], [("C", "H1")]),
]

# --- Raman (cm-1) --------------------------------------------------------------------------------
RAMAN = [
    _e(3100, 3000, "Aromatic / alkene C–H", "=C–H stretching", "alkene_cis", ["C1", "H1"], [("C1", "H1")]),
    _e(3000, 2800, "Alkyl chains (lipids, polymers)", "C–H stretching (CH₂ ≈2850/2880, CH₃ ≈2930)", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(2600, 2540, "Thiols", "S–H stretching", "thiol", ["S", "H"], [("S", "H")]),
    _e(2260, 2200, "Nitrile", "C≡N stretching", "nitrile", ["C", "N"], [("C", "N")]),
    _e(2250, 2100, "Alkyne", "C≡C stretching (strong in Raman)", "alkyne", ["C1", "C2"], [("C1", "C2")]),
    _e(1780, 1700, "Esters / ketones / acids", "C=O stretching", "ester", ["C", "Od"], [("C", "Od")]),
    _e(1680, 1640, "Alkene / amide I", "C=C stretching (cis ≈1655, trans ≈1670) / amide I", "alkene_cis", ["C1", "C2"], [("C1", "C2")]),
    _e(1620, 1580, "Aromatic rings", "ring C=C stretching", "aromatic", ["c0", "c1", "c2", "c3", "c4", "c5"], [("c0", "c1"), ("c2", "c3"), ("c4", "c5")]),
    _e(1540, 1500, "Carotenoids / polyenes", "C=C stretching (conjugated)", "polyene", ["C1", "C2", "C3", "C4"], [("C1", "C2"), ("C3", "C4")]),
    _e(1470, 1420, "Methylene / methyl", "CH₂ / CH₃ deformation", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(1310, 1290, "Lipids, long chains", "CH₂ twisting", "methylene", ["C", "H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(1275, 1255, "cis unsaturation", "=C–H bending", "alkene_cis", ["C1", "C2", "H1", "H2"], [("C1", "H1"), ("C2", "H2")]),
    _e(1260, 1230, "Amide III (proteins)", "C–N stretching + N–H bending", "peptide", ["C", "N", "H"], [("C", "N"), ("N", "H")]),
    _e(1170, 1140, "Carotenoids / polyenes", "C–C stretching", "polyene", ["C2", "C3", "C4", "C5"], [("C2", "C3"), ("C4", "C5")]),
    _e(1130, 1060, "Carbohydrates / lipids", "C–O and C–C stretching", "sugar", ["C1", "O1", "C5", "O5"], [("C1", "O1"), ("C5", "O5")]),
    _e(1090, 1080, "Carbonates", "CO₃²⁻ symmetric stretching", "carbonate", ["C", "O1", "O2", "O3"], [("C", "O1"), ("C", "O2"), ("C", "O3")]),
    _e(1010, 990, "Aromatic ring (phenylalanine, polystyrene)", "ring breathing", "aromatic", ["c0", "c1", "c2", "c3", "c4", "c5"], [("c0", "c1"), ("c1", "c2"), ("c2", "c3"), ("c3", "c4"), ("c4", "c5"), ("c5", "c0")]),
    _e(1000, 950, "Phosphates / silicates", "P–O / Si–O symmetric stretching", "phosphate", ["P", "O1", "O2"], [("P", "O1"), ("P", "O2")]),
    _e(900, 800, "Ethers / sugars", "C–O–C stretching", "ether", ["R", "O", "Rp"], [("R", "O"), ("O", "Rp")]),
    _e(550, 480, "Disulfide bridges", "S–S stretching", "disulfide", ["S1", "S2"], [("S1", "S2")]),
    _e(750, 600, "Alkyl halides / C–S", "C–Cl / C–S stretching", "alkyl_halide", ["C", "X"], [("C", "X")]),
]

# --- 1H NMR (ppm, typical for D2O / CDCl3 / DMSO; strongly matrix dependent) -----------------------
RMN = [
    _e(0.75, 1.05, "Terminal –CH₃ (lipids, fatty acids; Val/Leu/Ile methyls)", "¹H of CH₃", "methyl", ["H1", "H2", "H3"], [("C", "H1"), ("C", "H2"), ("C", "H3")]),
    _e(1.15, 1.40, "Chain –(CH₂)ₙ– (lipids); lactate CH₃ ≈1.33", "¹H of CH₂", "methylene", ["H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(1.40, 1.55, "Alanine CH₃ (≈1.48), β-CH₂ of amino acids", "¹H of CH₃ / CH₂", "methyl", ["H1", "H2", "H3"], [("C", "H1"), ("C", "H2"), ("C", "H3")]),
    _e(1.50, 1.70, "β-CH₂ to a carbonyl (lipids, ≈1.6)", "¹H of CH₂", "methylene", ["H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(1.90, 2.10, "Acetyl CH₃–C(=O) (acetate ≈1.92, N-acetyl ≈2.0); allylic CH₂ (≈2.0)", "¹H of CH₃ / allylic CH₂", "acetyl", ["H1", "H2", "H3"], [("Me", "H1"), ("Me", "H2"), ("Me", "H3")]),
    _e(2.20, 2.45, "α-CH₂ to carbonyl (≈2.3; fatty-acid chains)", "¹H of α-CH₂", "alpha_ch2", ["H1", "H2"], [("Ca", "H1"), ("Ca", "H2")]),
    _e(2.50, 2.70, "Citrate (AB system), succinate (≈2.41), Met S–CH₃", "¹H of CH₂ / S–CH₃", "methylene", ["H1", "H2"], [("C", "H1"), ("C", "H2")]),
    _e(2.70, 2.85, "Bis-allylic CH₂ of polyunsaturated fatty acids (≈2.77)", "¹H of =CH–CH₂–CH=", "bisallylic", ["H1", "H2"], [("Cm", "H1"), ("Cm", "H2")]),
    _e(2.95, 3.10, "Creatine / creatinine N–CH₃ (≈3.03–3.05); Lys ε-CH₂", "¹H of N–CH₃", "choline", ["Ha", "Hb", "Hc", "Hd", "He", "Hf"], [("M1", "Ha"), ("M1", "Hb"), ("M2", "Hc"), ("M2", "Hd"), ("M3", "He"), ("M3", "Hf")]),
    _e(3.15, 3.30, "Choline / N⁺(CH₃)₃ (≈3.2), glucose H2 (≈3.24)", "¹H of N⁺–CH₃", "choline", ["Ha", "Hb", "Hc", "Hd", "He", "Hf"], [("M1", "Ha"), ("M1", "Hb"), ("M2", "Hc"), ("M2", "Hd"), ("M3", "He"), ("M3", "Hf")]),
    _e(3.30, 3.45, "Methoxy O–CH₃ (≈3.3–3.9; methanol ≈3.34)", "¹H of O–CH₃", "methoxy", ["H1", "H2", "H3"], [("C", "H1"), ("C", "H2"), ("C", "H3")]),
    _e(3.40, 4.00, "Sugar ring H (3.2–4.0), CH–OH, ethanol CH₂ (≈3.65)", "¹H of C–H–O (carbohydrates)", "sugar", ["H2", "H3", "H4", "H5"], [("C2", "H2"), ("C3", "H3"), ("C4", "H4"), ("C5", "H5")]),
    _e(4.05, 4.40, "Glycerol sn-1,3 CH₂–O–C(=O) in triglycerides (≈4.1–4.3)", "¹H of glycerol CH₂", "glycerol", ["H1", "H2", "H4", "H5"], [("C1", "H1"), ("C1", "H2"), ("C3", "H4"), ("C3", "H5")]),
    _e(4.50, 4.90, "Anomeric H of β-glucose (≈4.64); residual water HDO (≈4.7–4.8, artefact)", "¹H anomeric / solvent", "sugar", ["H1"], [("C1", "H1")]),
    _e(5.15, 5.30, "α-glucose anomeric H (≈5.23); glycerol sn-2 CH (≈5.26)", "¹H anomeric", "sugar", ["H1"], [("C1", "H1")]),
    _e(5.28, 5.45, "Olefinic –CH=CH– of unsaturated fatty acids (≈5.3–5.4); sucrose anomeric (≈5.4)", "¹H olefinic", "vinyl_H", ["H1", "H2"], [("C1", "H1"), ("C2", "H2")]),
    _e(5.80, 6.60, "Conjugated olefinic / vinyl H", "¹H vinyl", "vinyl_H", ["H1", "H2"], [("C1", "H1"), ("C2", "H2")]),
    _e(6.50, 7.10, "Aromatic H (phenols, Tyr ≈6.9, flavonoids)", "¹H aromatic", "aromatic", ["h1", "h2", "h3", "h4", "h5"], [("c1", "h1"), ("c2", "h2"), ("c3", "h3"), ("c4", "h4"), ("c5", "h5")]),
    _e(7.10, 8.00, "Aromatic H (Phe ≈7.3–7.4, Trp, benzoates), His imidazole", "¹H aromatic / heteroaromatic", "aromatic", ["h1", "h2", "h3", "h4", "h5"], [("c1", "h1"), ("c2", "h2"), ("c3", "h3"), ("c4", "h4"), ("c5", "h5")]),
    _e(7.80, 8.60, "Formate (≈8.46), adenine / purines, amide N–H (in H₂O)", "¹H of formate / heteroaromatic / N–H", "formate", ["H"], [("H", "C")]),
    _e(9.40, 10.20, "Aldehyde –CHO", "¹H of CHO", "aldehyde", ["H"], [("C", "H")]),
    _e(10.0, 13.0, "Carboxylic acid O–H / phenolic O–H (exchangeable, solvent-dependent)", "¹H of COOH / OH", "acid", ["H"], [("O", "H")]),
]

TECNICAS = {
    "FT-MIR (cm⁻¹)": ("mir", MIR, "cm⁻¹"),
    "NIR (nm)": ("nir_nm", NIR, "nm"),
    "NIR (cm⁻¹)": ("nir_cm", NIR, "cm⁻¹"),
    "Raman (cm⁻¹)": ("raman", RAMAN, "cm⁻¹"),
    "¹H NMR (ppm)": ("rmn", RMN, "ppm"),
}
AVISO = ("Indicative values only: typical ranges for each group. Real positions shift with the matrix, solvent, pH, "
         "temperature, hydrogen bonding and neighbouring groups — always confirm against the literature or a reference spectrum.")


def sugerir_tecnica(eje, tipo_senal=None):
    """Best guess of the technique from the axis range (the user can override it)."""
    eje = np.asarray(eje, dtype=float)
    lo, hi = float(np.nanmin(eje)), float(np.nanmax(eje))
    if tipo_senal == "NMR" or (hi <= 16 and lo >= -3):
        return "¹H NMR (ppm)"
    if lo >= 600 and hi <= 2700 and (hi - lo) > 150 and hi < 3000:
        return "NIR (nm)" if tipo_senal != "Raman" else "Raman (cm⁻¹)"
    if lo >= 3800 and hi <= 14000:
        return "NIR (cm⁻¹)"
    if tipo_senal == "Raman":
        return "Raman (cm⁻¹)"
    return "FT-MIR (cm⁻¹)"


def buscar(tecnica, regiones, max_por_region=3):
    """regiones: list of (lo, hi) in the axis units of the data. Returns a DataFrame of candidate
    assignments (best first within each region)."""
    cod, base, _ = TECNICAS[tecnica]
    filas = []
    for k, (a, b) in enumerate(regiones):
        a, b = float(min(a, b)), float(max(a, b))
        if cod == "nir_cm":                       # cm-1 -> nm
            a, b = 1e7 / max(b, 1e-9), 1e7 / max(a, 1e-9)
        cand = []
        for i, e in enumerate(base):
            lo, hi = min(e["lo"], e["hi"]), max(e["lo"], e["hi"])
            ov = min(b, hi) - max(a, lo)
            if ov <= 0:
                continue
            frac = ov / max(min(b - a, hi - lo), 1e-9)
            cand.append((frac, -(hi - lo), i))
        cand.sort(reverse=True)
        for frac, _, i in cand[:max_por_region]:
            e = base[i]
            rng = f"{min(e['lo'], e['hi']):g}–{max(e['lo'], e['hi']):g}"
            filas.append({"Region": f"{regiones[k][0]:.4g} – {regiones[k][1]:.4g}", "Typical range": rng,
                          "Group / assignment": e["grupo"], "Vibration / proton type": e["modo"],
                          "_k": k, "_i": i, "_base": cod})
    return pd.DataFrame(filas)


def entrada(tecnica, i):
    return TECNICAS[tecnica][1][i]


def etiqueta_corta(texto, n=26):
    t = texto.split("(")[0].strip()
    return t if len(t) <= n else t[:n - 1] + "…"


def regiones_desde_mascara(eje, mascara, max_regiones=20, hueco=3):
    """Turns a variable-selection mask into (lo, hi) regions: neighbouring selected variables (up to
    'hueco' points apart) are merged; the widest 'max_regiones' are kept (sorted along the axis)."""
    idx = np.where(np.asarray(mascara, dtype=bool))[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > hueco)[0] + 1
    bloques = np.split(idx, cortes)
    regs = [(float(eje[b[0]]), float(eje[b[-1]]), len(b)) for b in bloques]
    regs = sorted(regs, key=lambda r: -r[2])[:max_regiones]
    return sorted([(min(a, b), max(a, b)) for a, b, _ in regs])
