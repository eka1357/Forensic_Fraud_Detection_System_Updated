"""compliance_engine.py
Rule-based compliance mapping for flagged transactions.

Maps risk tiers + transaction patterns to UK regulatory categories.
This is a transparent lookup table — easy to audit and extend.

DISCLAIMER: This is illustrative for portfolio / interview purposes.
It is NOT legal or compliance advice and should not be used as such.
"""

from typing import List, Dict

# ── Regulatory category definitions ──────────────────────────────────
REGULATIONS = {
    "MLR2017": {
        "name": "Money Laundering Regulations 2017",
        "description": "Suspicious Activity Report (SAR) consideration under the Money Laundering, "
                       "Terrorist Financing and Transfer of Funds Regulations 2017.",
    },
    "FCA_SYSC": {
        "name": "FCA SYSC 6.1 – Financial Crime",
        "description": "Systems and controls requirement under FCA's Senior Management Arrangements.",
    },
    "POCA2002": {
        "name": "Proceeds of Crime Act 2002",
        "description": "Potential obligation to file a SAR with the NCA under POCA s.330-332.",
    },
    "PSR2017": {
        "name": "Payment Services Regulations 2017",
        "description": "Strong Customer Authentication and fraud monitoring obligations.",
    },
}


# ── Pattern detectors ────────────────────────────────────────────────
def _is_high_amount(tx: Dict) -> bool:
    """Transaction amount exceeds a notable threshold."""
    return tx.get("amt", 0) > 5000


def _is_rapid_succession(tx: Dict) -> bool:
    """Placeholder for detecting structuring / smurfing patterns.
    In production this would check transaction velocity per account.
    """
    return False  # not enough context in a single-row call


def _is_unusual_hour(tx: Dict) -> bool:
    """Transaction occurred between midnight and 5 AM."""
    hour = tx.get("hour", 12)
    return 0 <= hour < 5


# ── Main mapping function ───────────────────────────────────────────
def get_compliance_flags(risk_tier: str, transaction: Dict) -> List[Dict]:
    """Return applicable regulatory triggers for a transaction.

    Parameters
    ----------
    risk_tier : "Low", "Medium", or "High"
    transaction : dict of transaction features

    Returns
    -------
    List of dicts, each with keys ``regulation``, ``name``, ``reason``.
    """
    flags: List[Dict] = []

    if risk_tier == "High":
        flags.append({
            "regulation": "MLR2017",
            "name": REGULATIONS["MLR2017"]["name"],
            "reason": "Transaction flagged as high-risk by ensemble model — "
                      "SAR consideration required.",
        })
        flags.append({
            "regulation": "POCA2002",
            "name": REGULATIONS["POCA2002"]["name"],
            "reason": "Potential proceeds-of-crime trigger due to high risk score.",
        })

    if risk_tier in ("Medium", "High") and _is_high_amount(transaction):
        flags.append({
            "regulation": "FCA_SYSC",
            "name": REGULATIONS["FCA_SYSC"]["name"],
            "reason": f"Medium/high-risk transaction with elevated amount "
                      f"(${transaction.get('amt', '?'):.2f}) requires enhanced monitoring.",
        })

    if risk_tier in ("Medium", "High") and _is_unusual_hour(transaction):
        flags.append({
            "regulation": "PSR2017",
            "name": REGULATIONS["PSR2017"]["name"],
            "reason": "Unusual transaction hour combined with elevated risk — "
                      "SCA / fraud-monitoring review.",
        })

    return flags


def format_flags_text(flags: List[Dict]) -> str:
    """Human-readable one-liner for dashboard display."""
    if not flags:
        return "No regulatory triggers"
    return "; ".join(f"{f['name']}: {f['reason']}" for f in flags)


if __name__ == "__main__":
    demo = {"amt": 9500, "hour": 2}
    for tier in ["Low", "Medium", "High"]:
        result = get_compliance_flags(tier, demo)
        print(f"\n{tier} risk →")
        if result:
            for f in result:
                print(f"  [{f['regulation']}] {f['reason']}")
        else:
            print("  No triggers")
