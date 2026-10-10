"""Classify a company so each model is only applied where it makes sense.

Primary source: the SEC's SIC industry code (the same code academic studies use to
split "manufacturing" from the rest - Altman's original Z was estimated on SIC
2000-3999 manufacturers). Fallback when SIC is unavailable: Yahoo's sector/industry
labels, which are less precise.
"""

from __future__ import annotations

from dataclasses import dataclass

BANK = "bank"
INSURANCE = "insurance"
CAPITAL_MARKETS = "capital_markets"
REIT = "reit"
UTILITY = "utility"
MANUFACTURING = "manufacturing"
NON_MANUFACTURING = "non_manufacturing"
UNKNOWN = "unknown"

# Balance sheets of these businesses are made of financial assets/liabilities:
# working capital, accruals and leverage mean something different, so Altman,
# Beneish and EV-based multiples don't apply.
FINANCIALS = {BANK, INSURANCE, CAPITAL_MARKETS}

CATEGORY_LABELS = {
    BANK: "Banca / istituto di credito",
    INSURANCE: "Assicurazione",
    CAPITAL_MARKETS: "Intermediario finanziario / mercati dei capitali",
    REIT: "REIT (società immobiliare quotata)",
    UTILITY: "Utility (servizi di pubblica utilità regolati)",
    MANUFACTURING: "Manifatturiera",
    NON_MANUFACTURING: "Non manifatturiera / servizi",
    UNKNOWN: "Non determinato",
}

# Yahoo industry keywords that indicate a manufacturer (fallback only)
_YAHOO_MANUFACTURING_KEYWORDS = (
    "Semiconductor", "Auto", "Chemical", "Aerospace", "Machinery", "Steel", "Aluminum",
    "Packaging", "Electronic Components", "Electrical Equipment", "Medical Devices",
    "Medical Instruments", "Drug Manufacturers", "Biotechnology", "Beverages", "Packaged Foods",
    "Confectioners", "Tobacco", "Building Products", "Furnishings", "Paper", "Farm & Heavy",
    "Specialty Industrial", "Computer Hardware", "Consumer Electronics", "Household & Personal",
    "Oil & Gas Refining", "Footwear", "Apparel Manufacturing", "Recreational Vehicles",
    "Scientific & Technical Instruments", "Metal Fabrication", "Tools", "Communication Equipment",
)


@dataclass
class SectorInfo:
    category: str
    source: str  # "SIC (SEC)" / "Yahoo Finance" / "-"
    detail: str  # e.g. "SIC 7372 - Services-Prepackaged Software"

    @property
    def profile(self) -> str:
        return profile_of(self.category)

    @property
    def label(self) -> str:
        return CATEGORY_LABELS[self.category]

    @property
    def is_financial(self) -> bool:
        return self.category in FINANCIALS


def profile_of(category: str) -> str:
    """Which yardsticks fit this business (see src/profiles.py): financial / reit / utility / generic.
    Same profiles as PATTY (~/equity-analyst/src/profiles.py)."""
    if category in FINANCIALS:
        return "financial"
    if category in (REIT, UTILITY):
        return category
    return "generic"


def classify_sic(sic: int) -> str:
    if 6000 <= sic <= 6199:  # depository and non-depository credit institutions
        return BANK
    if 6200 <= sic <= 6299:  # brokers, dealers, exchanges
        return CAPITAL_MARKETS
    if 6300 <= sic <= 6399:  # insurance carriers
        return INSURANCE
    if sic == 6798:
        return REIT
    if 4900 <= sic <= 4949:  # electric, gas, water and combined utilities (not waste management)
        return UTILITY
    if 2000 <= sic <= 3999:
        return MANUFACTURING
    return NON_MANUFACTURING


def classify_yahoo(sector: str | None, industry: str | None) -> str:
    industry = industry or ""
    if not sector and not industry:
        return UNKNOWN
    if "Bank" in industry or industry in ("Credit Services", "Mortgage Finance"):
        return BANK
    if "Insurance" in industry and "Brokers" not in industry:
        return INSURANCE
    if industry in ("Capital Markets", "Asset Management", "Financial Conglomerates"):
        return CAPITAL_MARKETS
    if industry.startswith("REIT"):
        return REIT
    if (sector or "") == "Utilities":
        return UTILITY
    if any(k in industry for k in _YAHOO_MANUFACTURING_KEYWORDS):
        return MANUFACTURING
    return NON_MANUFACTURING


def classify(sic: int | None, sic_description: str | None, yahoo_sector: str | None,
             yahoo_industry: str | None) -> SectorInfo:
    if sic:
        category = classify_sic(int(sic))
        detail = f"SIC {sic} - {sic_description or ''}".strip(" -")
        # Some REITs are filed by the SEC under plain real estate (e.g. INVH: SIC 6510 "Real Estate Operators"):
        # Yahoo's "REIT - ..." industry corrects that, and only that.
        if category not in FINANCIALS and category != REIT and (yahoo_industry or "").startswith("REIT"):
            return SectorInfo(REIT, "SIC (SEC) + Yahoo Finance", f"{detail}; Yahoo: {yahoo_industry}")
        return SectorInfo(category, "SIC (SEC)", detail)
    category = classify_yahoo(yahoo_sector, yahoo_industry)
    detail = " / ".join(x for x in (yahoo_sector, yahoo_industry) if x) or "-"
    return SectorInfo(category, "Yahoo Finance" if category != UNKNOWN else "-", detail)
