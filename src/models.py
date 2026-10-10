"""Accounting-quality and distress models, each returning all of its components.

- Altman Z-Score: bankruptcy risk. Original Z for manufacturers, Z'' for everyone
  else; not applicable to financials, REITs and regulated utilities (src/profiles.py).
- Piotroski F-Score: 9 binary tests of fundamental strength, year t vs year t-1.
  A criterion that can't be evaluated is reported as such (never as a fail).
- Beneish M-Score: 8-variable probability-of-manipulation model, threshold -1.78;
  not applicable to financials and REITs.

Inputs are the canonical annual frame from statements.py (one row per fiscal year,
oldest -> newest). Missing inputs produce "N/D", never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import profiles
from .sector import MANUFACTURING, UNKNOWN, SectorInfo
from .statements import FIELD_LABELS, consecutive, value

ND = "N/D"

NOT_APPLICABLE = "Non applicabile"


@dataclass
class ModelResult:
    name: str
    value: float | None = None
    label: str = ND  # zone / verdict, or "N/D" / "Non applicabile"
    variant: str = ""
    fiscal_year: str = ""  # fiscal year end the result refers to
    components: list = field(default_factory=list)  # list of dicts, rendered as a table
    missing: list = field(default_factory=list)  # human-readable missing inputs
    notes: list = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.value is not None


def _missing_labels(row: pd.Series, names: list, suffix: str = "") -> list:
    return [FIELD_LABELS.get(n, n) + suffix for n in names if value(row, n) is None]


def _fy(ts: pd.Timestamp) -> str:
    return ts.strftime("%d/%m/%Y")


# ---------------------------------------------------------------- Altman ----

ALTMAN_ORIGINAL = {
    "name": "Z originale (manifatturiere quotate)",
    "weights": {"X1": 1.2, "X2": 1.4, "X3": 3.3, "X4": 0.6, "X5": 1.0},
    "distress": 1.81, "safe": 2.99,
}
ALTMAN_Z2 = {
    "name": "Z'' (non manifatturiere / servizi)",
    "weights": {"X1": 6.56, "X2": 3.26, "X3": 6.72, "X4": 1.05},
    "distress": 1.10, "safe": 2.60,
}


def altman_zone(z: float, variant: dict) -> str:
    if z > variant["safe"]:
        return "Sicura"
    if z >= variant["distress"]:
        return "Grigia"
    return "Distress"


def altman_z(annual: pd.DataFrame, sector: SectorInfo, market_value_equity: float | None = None) -> ModelResult:
    """market_value_equity: market cap *at the fiscal year end* (needed only by the
    original Z). Using today's market cap against a year-old balance sheet would mix
    two different dates."""
    res = ModelResult("Altman Z-Score")
    if sector.is_financial:
        res.label = NOT_APPLICABLE
        res.notes.append(f"Il modello non è pensato per {sector.label.lower()}: il bilancio è fatto di attività "
                         "e passività finanziarie, quindi capitale circolante e leva hanno un altro significato.")
        return res
    if not profiles.get(sector.profile)["altman"]:
        res.label = NOT_APPLICABLE
        res.notes.append(f"L'Altman Z-Score è {profiles.na_reason(sector.profile)}: è stato stimato su aziende "
                         "industriali, con capitale circolante e debito di natura diversa.")
        return res
    if annual.empty:
        res.missing.append("Bilanci annuali")
        return res

    manufacturing = sector.category == MANUFACTURING
    variant = ALTMAN_ORIGINAL if manufacturing else ALTMAN_Z2
    res.variant = variant["name"]
    if sector.category == UNKNOWN:
        res.notes.append("Settore non determinato: uso Z'' (la variante più generale).")

    row = annual.iloc[-1]
    res.fiscal_year = _fy(annual.index[-1])
    needed = ["total_assets", "working_capital", "retained_earnings", "ebit", "total_liabilities"]
    needed += ["revenue"] if manufacturing else ["equity"]
    res.missing = _missing_labels(row, needed)
    if manufacturing and market_value_equity is None:
        res.missing.append("Capitalizzazione di mercato a fine esercizio")
    ta, tl = value(row, "total_assets"), value(row, "total_liabilities")
    if res.missing or not ta or not tl:
        return res

    x = {
        "X1": ("Capitale circolante / Totale attivo", value(row, "working_capital") / ta),
        "X2": ("Utili non distribuiti / Totale attivo", value(row, "retained_earnings") / ta),
        "X3": ("EBIT / Totale attivo", value(row, "ebit") / ta),
    }
    if manufacturing:
        x["X4"] = ("Capitalizzazione di mercato / Totale passività", market_value_equity / tl)
        x["X5"] = ("Ricavi / Totale attivo", value(row, "revenue") / ta)
    else:
        x["X4"] = ("Patrimonio netto contabile / Totale passività", value(row, "equity") / tl)

    z = 0.0
    for key, weight in variant["weights"].items():
        desc, ratio = x[key]
        z += weight * ratio
        res.components.append({"Componente": key, "Formula": desc, "Valore": ratio, "Peso": weight,
                               "Contributo": weight * ratio})
    res.value = z
    res.label = altman_zone(z, variant)
    if not manufacturing and (x["X1"][1] < 0 or x["X2"][1] < 0):
        res.notes.append("Capitale circolante o utili non distribuiti negativi pesano molto nella Z''. "
                         "Non sempre indicano difficoltà: succede anche a chi incassa in anticipo (abbonamenti, "
                         "software) o ha riacquistato molte azioni proprie. Verifica liquidità e flussi di cassa "
                         "prima di trarre conclusioni.")
    res.notes.append(f"Soglie {variant['name']}: distress < {variant['distress']:.2f}, "
                     f"zona grigia {variant['distress']:.2f}-{variant['safe']:.2f}, sicura > {variant['safe']:.2f}.")
    return res


# ------------------------------------------------------------- Piotroski ----

def piotroski_f(annual: pd.DataFrame, sector: SectorInfo | None = None) -> ModelResult:
    """Piotroski (2000). Ratios use *beginning-of-year* total assets as in the
    paper, so the full test needs three consecutive balance sheets (t, t-1, t-2)."""
    res = ModelResult("Piotroski F-Score")
    if len(annual) < 2 or not consecutive(annual.index[-2], annual.index[-1]):
        res.missing.append("Servono due esercizi consecutivi")
        return res

    t, t1 = annual.iloc[-1], annual.iloc[-2]
    t2 = annual.iloc[-3] if len(annual) >= 3 and consecutive(annual.index[-3], annual.index[-2]) else None
    res.fiscal_year = _fy(annual.index[-1])
    g = lambda row, name: None if row is None else value(row, name)  # noqa: E731

    def ratio(a, b):
        return None if a is None or not b else a / b

    ta_t, ta_t1, ta_t2 = g(t, "total_assets"), g(t1, "total_assets"), g(t2, "total_assets")
    ni_t, ni_t1 = g(t, "net_income"), g(t1, "net_income")
    cfo_t = g(t, "ocf")
    roa_t, roa_t1 = ratio(ni_t, ta_t1), ratio(ni_t1, ta_t2)

    def lev(ltd, ta_end, ta_begin):
        if ltd is None or not ta_end or not ta_begin:
            return None
        return ltd / ((ta_end + ta_begin) / 2)

    ltd_t, ltd_t1 = g(t, "long_term_debt"), g(t1, "long_term_debt")
    lev_t, lev_t1 = lev(ltd_t, ta_t, ta_t1), lev(ltd_t1, ta_t1, ta_t2)
    cr_t = ratio(g(t, "current_assets"), g(t, "current_liabilities"))
    cr_t1 = ratio(g(t1, "current_assets"), g(t1, "current_liabilities"))
    sh_t, sh_t1 = g(t, "shares_outstanding"), g(t1, "shares_outstanding")
    gm_t = ratio(g(t, "gross_profit"), g(t, "revenue"))
    gm_t1 = ratio(g(t1, "gross_profit"), g(t1, "revenue"))
    at_t, at_t1 = ratio(g(t, "revenue"), ta_t1), ratio(g(t1, "revenue"), ta_t2)

    def test(cond_inputs, cond):
        return None if any(v is None for v in cond_inputs) else bool(cond())

    def no_new_leverage():
        if ltd_t == 0 and ltd_t1 == 0:
            return True  # no long-term debt in either year
        return lev_t < lev_t1

    criteria = [
        ("Redditività", "ROA positivo (utile netto / attivo iniziale > 0)", roa_t, None,
         test([roa_t], lambda: roa_t > 0)),
        ("Redditività", "Flusso di cassa operativo positivo", cfo_t, None,
         test([cfo_t], lambda: cfo_t > 0)),
        ("Redditività", "ROA in miglioramento rispetto all'anno prima", roa_t, roa_t1,
         test([roa_t, roa_t1], lambda: roa_t > roa_t1)),
        ("Redditività", "Qualità degli utili: cassa operativa > utile netto", cfo_t, ni_t,
         test([cfo_t, ni_t], lambda: cfo_t > ni_t)),
        ("Leva/liquidità", "Leva (debito LT / attivo medio) non aumentata", lev_t, lev_t1,
         test([lev_t, lev_t1], no_new_leverage)),
        ("Leva/liquidità", "Current ratio in miglioramento", cr_t, cr_t1,
         test([cr_t, cr_t1], lambda: cr_t > cr_t1)),
        ("Leva/liquidità", "Nessuna nuova emissione di azioni", sh_t, sh_t1,
         test([sh_t, sh_t1], lambda: sh_t <= sh_t1)),
        ("Efficienza", "Margine lordo in miglioramento", gm_t, gm_t1,
         test([gm_t, gm_t1], lambda: gm_t > gm_t1)),
        ("Efficienza", "Rotazione dell'attivo in miglioramento", at_t, at_t1,
         test([at_t, at_t1], lambda: at_t > at_t1)),
    ]
    for area, desc, cur, prev, passed in criteria:
        res.components.append({"Area": area, "Criterio": desc, "Anno t": cur, "Anno t-1": prev,
                               "Esito": ND if passed is None else ("Superato" if passed else "Non superato")})
        if passed is None:
            res.missing.append(desc)

    evaluable = sum(1 for *_, p in criteria if p is not None)
    if evaluable == 0:
        return res
    score = sum(1 for *_, p in criteria if p)
    res.value = float(score)
    res.variant = f"{evaluable}/9 criteri valutabili"
    res.label = "Forte" if score >= 7 else ("Debole" if score <= 3 else "Intermedio")
    if evaluable < 9:
        res.label += " (parziale)"
        res.notes.append(f"Solo {evaluable} criteri su 9 valutabili: il punteggio massimo raggiungibile è {evaluable}.")
    if sector is not None and sector.is_financial:
        res.notes.append("Il Piotroski non è pensato per banche e assicurazioni: interpretalo con cautela.")
    return res


# --------------------------------------------------------------- Beneish ----

BENEISH_COEFFS = {
    "DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
    "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327,
}
BENEISH_INTERCEPT = -4.84
BENEISH_THRESHOLD = -1.78

BENEISH_DESCRIPTIONS = {
    "DSRI": "Crediti/ricavi rispetto all'anno prima (crediti che crescono più delle vendite)",
    "GMI": "Margine lordo dell'anno prima / margine lordo attuale (margini in calo)",
    "AQI": "Qualità dell'attivo (quota di attivo non corrente e non materiale)",
    "SGI": "Crescita dei ricavi",
    "DEPI": "Tasso di ammortamento dell'anno prima / attuale (ammortamenti rallentati)",
    "SGAI": "Spese generali/ricavi rispetto all'anno prima",
    "TATA": "Accantonamenti: (utile - cassa operativa) / totale attivo",
    "LVGI": "Leva finanziaria rispetto all'anno prima",
}


BENEISH_NA_WHY = {
    "reit": "i suoi indici (crediti, margine lordo, attivo corrente) non descrivono un'attività immobiliare",
    "utility": "le utility non pubblicano le spese generali (SG&A) né un margine lordo, che il modello richiede",
}


def beneish_m(annual: pd.DataFrame, sector: SectorInfo) -> ModelResult:
    res = ModelResult("Beneish M-Score", variant="Modello a 8 variabili (Beneish, 1999)")
    if sector.is_financial:
        res.label = NOT_APPLICABLE
        res.notes.append(f"Il modello non è pensato per {sector.label.lower()}.")
        return res
    if not profiles.get(sector.profile)["beneish"]:
        res.label = NOT_APPLICABLE
        why = BENEISH_NA_WHY.get(sector.profile, "")
        res.notes.append(f"Il Beneish M-Score è {profiles.na_reason(sector.profile)}" + (f": {why}." if why else "."))
        return res
    if len(annual) < 2 or not consecutive(annual.index[-2], annual.index[-1]):
        res.missing.append("Servono due esercizi consecutivi")
        return res

    t, t1 = annual.iloc[-1], annual.iloc[-2]
    res.fiscal_year = _fy(annual.index[-1])
    both = ["receivables", "revenue", "gross_profit", "current_assets", "net_ppe", "total_assets",
            "d_and_a", "sga", "long_term_debt", "current_liabilities"]
    res.missing = _missing_labels(t, both + ["ocf"], " (anno t)") + _missing_labels(t1, both, " (anno t-1)")
    ni_t = value(t, "net_income_cont_ops")
    if ni_t is None:
        ni_t = value(t, "net_income")
        if ni_t is None:
            res.missing.append("Utile netto (anno t)")
    if res.missing:
        return res

    v0 = {k: value(t, k) for k in both}
    v1 = {k: value(t1, k) for k in both}
    ocf_t = value(t, "ocf")

    def safe(fn):
        try:
            return fn()
        except ZeroDivisionError:
            return None

    idx = {
        "DSRI": safe(lambda: (v0["receivables"] / v0["revenue"]) / (v1["receivables"] / v1["revenue"])),
        "GMI": safe(lambda: (v1["gross_profit"] / v1["revenue"]) / (v0["gross_profit"] / v0["revenue"])),
        "AQI": safe(lambda: (1 - (v0["current_assets"] + v0["net_ppe"]) / v0["total_assets"])
                    / (1 - (v1["current_assets"] + v1["net_ppe"]) / v1["total_assets"])),
        "SGI": safe(lambda: v0["revenue"] / v1["revenue"]),
        "DEPI": safe(lambda: (v1["d_and_a"] / (v1["d_and_a"] + v1["net_ppe"]))
                     / (v0["d_and_a"] / (v0["d_and_a"] + v0["net_ppe"]))),
        "SGAI": safe(lambda: (v0["sga"] / v0["revenue"]) / (v1["sga"] / v1["revenue"])),
        "TATA": safe(lambda: (ni_t - ocf_t) / v0["total_assets"]),
        "LVGI": safe(lambda: ((v0["long_term_debt"] + v0["current_liabilities"]) / v0["total_assets"])
                     / ((v1["long_term_debt"] + v1["current_liabilities"]) / v1["total_assets"])),
    }
    for key, val in idx.items():
        coeff = BENEISH_COEFFS[key]
        res.components.append({"Indice": key, "Significato": BENEISH_DESCRIPTIONS[key], "Valore": val,
                               "Coefficiente": coeff, "Contributo": None if val is None else coeff * val})
    broken = [k for k, v in idx.items() if v is None]
    if broken:
        res.missing.append("Indici non calcolabili (divisione per zero): " + ", ".join(broken))
        return res

    m = BENEISH_INTERCEPT + sum(BENEISH_COEFFS[k] * v for k, v in idx.items())
    res.value = m
    res.label = "Possibile manipolazione" if m > BENEISH_THRESHOLD else "Nessun segnale"
    sgi_excess = BENEISH_COEFFS["SGI"] * (idx["SGI"] - 1)
    if m > BENEISH_THRESHOLD and m - sgi_excess <= BENEISH_THRESHOLD:
        res.notes.append(f"Attenzione: il segnale dipende soprattutto dalla forte crescita dei ricavi "
                         f"(SGI = {idx['SGI']:.2f}). Con ricavi stabili il punteggio sarebbe "
                         f"{m - sgi_excess:.2f}, sotto la soglia: per le società in forte crescita "
                         "questo è un falso positivo frequente del modello.")
    res.notes.append(f"Soglia {BENEISH_THRESHOLD}: sopra la soglia il profilo contabile somiglia a quello delle "
                     "società che hanno manipolato i conti. È un campanello d'allarme statistico, non una prova.")
    return res
