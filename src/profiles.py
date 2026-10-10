"""Which accounting models apply to each type of business.

Mirrors the profiles of PATTY (~/equity-analyst/src/profiles.py), limited to what the
screening uses: Altman Z and Beneish M were estimated on industrial companies. A REIT
grows by issuing shares and borrowing against buildings, a regulated utility spends
more than its cash flow on networks by design: applying the models to them produces
alarms - or missing data - that say nothing about the company.
"""

from __future__ import annotations

PROFILES = {
    "generic": {"label": "azienda industriale o di servizi", "altman": True, "beneish": True},
    "reit": {"label": "REIT", "altman": False, "beneish": False},
    "utility": {"label": "utility regolata", "altman": False, "beneish": False},  # no SG&A: Beneish always N/D
    "financial": {"label": "banca, assicurazione o intermediario finanziario", "altman": False, "beneish": False},
}

NA_REASONS = {
    "reit": "non adatto ai REIT (contano FFO, immobili e debito a lungo termine)",
    "utility": "non adatto alle utility regolate (investimenti e rendimenti fissati dal regolatore)",
    "financial": "non applicabile a banche e assicurazioni",
}


def get(profile: str) -> dict:
    return PROFILES.get(profile, PROFILES["generic"])


def na_reason(profile: str) -> str:
    return NA_REASONS.get(profile, "non applicabile a questo tipo di azienda")
