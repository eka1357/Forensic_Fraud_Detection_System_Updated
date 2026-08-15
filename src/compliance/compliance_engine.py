"""compliance_engine.py — Rule-based compliance mapping for flagged transactions.

Maps risk tiers + transaction patterns to UK regulatory categories:
- Money Laundering Regulations 2017 (MLR 2017)
- FCA SYSC 6.1 (Financial Crime Systems and Controls)
- Proceeds of Crime Act 2002 (POCA 2002)
- Payment Services Regulations 2017 (PSR 2017)

DISCLAIMER: This mapping layer is illustrative for demonstration and portfolio purposes.
It does NOT constitute legal or regulatory compliance advice.
"""

import logging
from typing import List, Dict, Any
from src.config import (
    COMPLIANCE_HIGH_AMOUNT_THRESHOLD,
    COMPLIANCE_UNUSUAL_HOUR_START,
    COMPLIANCE_UNUSUAL_HOUR_END,
)

logger = logging.getLogger(__name__)

# Regulatory category definitions
REGULATIONS = {
    "MLR2017": {
        "name": "Money Laundering Regulations 2017",
        "description": "Suspicious Activity Report (SAR) consideration under the Money Laundering, "
                       "Terrorist Financing and Transfer of Funds Regulations 2017.",
    },
    "FCA_SYSC": {
        "name": "FCA SYSC 6.1 – Financial Crime",
        "description": "Systems and controls requirement under FCA Senior Management Arrangements.",
    },
    "POCA2002": {
        "name": "Proceeds of Crime Act 2002",
        "description": "Potential obligation to file a SAR with the National Crime Agency (NCA) under POCA s.330-332.",
    },
    "PSR2017": {
        "name": "Payment Services Regulations 2017",
        "description": "Strong Customer Authentication and fraud monitoring obligations under PSR 2017.",
    },
}


def _is_high_amount(tx: Dict[str, Any]) -> bool:
    """Check if transaction amount exceeds high-amount threshold (£5,000)."""
    return float(tx.get("amt", 0.0) or 0.0) >= COMPLIANCE_HIGH_AMOUNT_THRESHOLD


def _is_unusual_hour(tx: Dict[str, Any]) -> bool:
    """Check if transaction occurred during off-peak hours (midnight to 5 AM)."""
    try:
        hour = int(tx.get("hour", 12))
        return COMPLIANCE_UNUSUAL_HOUR_START <= hour < COMPLIANCE_UNUSUAL_HOUR_END
    except (ValueError, TypeError):
        return False


def get_compliance_flags(risk_tier: str, transaction: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return applicable regulatory triggers for a transaction.

    Parameters
    ----------
    risk_tier : "Low", "Medium", or "High"
    transaction : dict of transaction features (e.g. amt, hour, category)

    Returns
    -------
    List of dicts, each with keys ``regulation``, ``name``, ``reason``.
    """
    flags: List[Dict[str, str]] = []

    if risk_tier == "High":
        flags.append({
            "regulation": "MLR2017",
            "name": REGULATIONS["MLR2017"]["name"],
            "reason": "Transaction flagged as high-risk by multi-signal ensemble model — "
                      "SAR consideration required under MLR 2017.",
        })
        flags.append({
            "regulation": "POCA2002",
            "name": REGULATIONS["POCA2002"]["name"],
            "reason": "Potential proceeds-of-crime trigger due to high risk score — review for NCA reporting.",
        })

    if risk_tier in ("Medium", "High") and _is_high_amount(transaction):
        amt_val = float(transaction.get("amt", 0.0) or 0.0)
        flags.append({
            "regulation": "FCA_SYSC",
            "name": REGULATIONS["FCA_SYSC"]["name"],
            "reason": f"Elevated-risk transaction with high amount "
                      f"(£{amt_val:,.2f}) requires enhanced due diligence under FCA SYSC 6.1.",
        })

    if risk_tier in ("Medium", "High") and _is_unusual_hour(transaction):
        flags.append({
            "regulation": "PSR2017",
            "name": REGULATIONS["PSR2017"]["name"],
            "reason": "Unusual transaction hour combined with elevated risk — "
                      "Strong Customer Authentication (SCA) & monitoring review under PSR 2017.",
        })

    return flags


def format_flags_text(flags: List[Dict[str, str]]) -> str:
    """Format compliance triggers into a clean human-readable summary."""
    if not flags:
        return "No regulatory triggers"
    return "; ".join(f"{f['name']}: {f['reason']}" for f in flags)
