"""Type-of-business classification and profiles (aligned with PATTY's src/profiles.py)."""

from src.sector import REIT, UTILITY, classify


def test_reit_recognised_from_yahoo_industry():
    s = classify(None, None, "Real Estate", "REIT - Residential")
    assert s.category == REIT and s.profile == "reit"


def test_reit_filed_as_plain_real_estate_corrected_by_yahoo_industry():
    # INVH: the SEC files it under SIC 6510 "Real Estate Operators", Yahoo says "REIT - Residential"
    s = classify(6510, "Real Estate Operators", "Real Estate", "REIT - Residential")
    assert s.category == REIT and "Yahoo" in s.source
    assert classify(6510, "Real Estate Operators", "Real Estate", "Real Estate Services").category != REIT
    # a bank stays a bank whatever Yahoo says
    assert classify(6021, "National Commercial Banks", None, "REIT - Mortgage").category == "bank"


def test_utilities_from_sic_or_yahoo_sector():
    assert classify(4911, "Electric Services", None, None).category == UTILITY
    assert classify(4953, "Refuse Systems", None, None).category != UTILITY  # waste management
    s = classify(None, None, "Utilities", "Utilities - Regulated Electric")
    assert s.category == UTILITY and s.profile == "utility"


def test_profiles():
    assert classify(None, None, "Technology", "Software - Application").profile == "generic"
    assert classify(None, None, "Financial Services", "Banks - Regional").profile == "financial"
