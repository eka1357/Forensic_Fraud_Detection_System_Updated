"""test_compliance.py — Unit tests for the UK regulatory compliance engine."""

import unittest
from src.compliance.compliance_engine import (
    get_compliance_flags,
    format_flags_text,
    _is_high_amount,
    _is_unusual_hour,
)


class TestComplianceEngine(unittest.TestCase):

    def test_high_risk_tier_triggers_mlr_and_poca(self):
        """High risk tier must trigger MLR 2017 and POCA 2002."""
        tx = {"amt": 100.0, "hour": 14}
        flags = get_compliance_flags("High", tx)
        reg_codes = [f["regulation"] for f in flags]
        self.assertIn("MLR2017", reg_codes)
        self.assertIn("POCA2002", reg_codes)

    def test_low_risk_tier_produces_no_flags(self):
        """Low risk tier with standard amount and normal hour produces zero flags."""
        tx = {"amt": 50.0, "hour": 14}
        flags = get_compliance_flags("Low", tx)
        self.assertEqual(len(flags), 0)
        self.assertEqual(format_flags_text(flags), "No regulatory triggers")

    def test_high_amount_trigger_fca_sysc(self):
        """Elevated risk with amount >= 5000 triggers FCA SYSC 6.1."""
        tx = {"amt": 7500.0, "hour": 14}
        flags_med = get_compliance_flags("Medium", tx)
        reg_codes = [f["regulation"] for f in flags_med]
        self.assertIn("FCA_SYSC", reg_codes)
        self.assertNotIn("MLR2017", reg_codes)

    def test_unusual_hour_trigger_psr2017(self):
        """Elevated risk during early morning (0-4 AM) triggers PSR 2017."""
        tx = {"amt": 100.0, "hour": 2}
        flags_med = get_compliance_flags("Medium", tx)
        reg_codes = [f["regulation"] for f in flags_med]
        self.assertIn("PSR2017", reg_codes)

    def test_combined_high_risk_high_amount_unusual_hour(self):
        """High risk + high amount + unusual hour triggers all four regulations."""
        tx = {"amt": 12000.0, "hour": 3}
        flags = get_compliance_flags("High", tx)
        reg_codes = {f["regulation"] for f in flags}
        self.assertEqual(reg_codes, {"MLR2017", "POCA2002", "FCA_SYSC", "PSR2017"})

    def test_format_flags_text(self):
        """Formatting flags returns human-readable semicolon-separated string."""
        flags = [
            {"regulation": "MLR2017", "name": "Money Laundering Regulations 2017", "reason": "Test SAR"},
            {"regulation": "POCA2002", "name": "Proceeds of Crime Act 2002", "reason": "Test POCA"},
        ]
        formatted = format_flags_text(flags)
        self.assertIn("Money Laundering Regulations 2017: Test SAR", formatted)
        self.assertIn("Proceeds of Crime Act 2002: Test POCA", formatted)


if __name__ == "__main__":
    unittest.main()
