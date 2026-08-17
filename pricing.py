"""
Pricing engine for the UBI POC.
Provides compute_premium(base_premium, risk_score, monthly_mileage, is_ev, safe_driver)
which returns a transparent breakdown.
"""
import json
import os
from datetime import datetime

RULES_PATH = os.path.join(os.path.dirname(__file__), 'pricing_rules.json')


def load_rules(path=RULES_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _find_risk_adj(rules, risk_score):
    # risk_bands sorted by descending min_score
    bands = sorted(rules.get('risk_bands', []), key=lambda b: b['min_score'], reverse=True)
    for b in bands:
        if risk_score >= b['min_score']:
            return b['adj_pct']
    return 0


def _find_mileage_adj(rules, monthly_mileage):
    bands = sorted(rules.get('mileage_bands', []), key=lambda b: b['max_km'])
    for b in bands:
        if monthly_mileage <= b['max_km']:
            return b['adj_pct']
    return 0


def compute_premium(risk_score: float, monthly_mileage: float, is_ev: bool=False, safe_driver: bool=False, base_premium: float=None, rules: dict=None):
    """Compute premium breakdown and return a dict with detailed fields.

    risk_score: 0-100 (higher is safer in our UI mapping)
    monthly_mileage: km per month
    is_ev: whether vehicle is electric
    safe_driver: extra discount flag
    base_premium: override base premium (if None, load from rules)
    rules: optional rules dict
    """
    rules = rules or load_rules()
    if base_premium is None:
        base_premium = rules.get('base_premium', 30000)

    # note: in our model higher probability was interpreted as higher risk; for mapping to 0-100
    # we assume risk_score already in 0-100 where higher is better. If using model prob, pass 100*(1-prob)

    risk_adj_pct = _find_risk_adj(rules, risk_score)
    mileage_adj_pct = _find_mileage_adj(rules, monthly_mileage)
    ev_adj_pct = -rules.get('ev_discount_pct', 0) if is_ev else 0
    safe_adj_pct = -rules.get('safe_driving_discount_pct', 0) if safe_driver else 0

    # Apply adjustments multiplicatively on base premium in percentage terms
    premium = base_premium
    premium = premium * (1 + risk_adj_pct/100.0)
    premium = premium * (1 + mileage_adj_pct/100.0)
    premium = premium * (1 + ev_adj_pct/100.0)
    premium = premium * (1 + safe_adj_pct/100.0)

    tax = premium * rules.get('tax_pct', 0) / 100.0
    total_due = premium + tax

    breakdown = {
        'base_premium': round(base_premium,2),
        'risk_adj_pct': risk_adj_pct,
        'mileage_adj_pct': mileage_adj_pct,
        'ev_adj_pct': ev_adj_pct,
        'safe_adj_pct': safe_adj_pct,
        'premium_before_tax': round(premium,2),
        'tax_pct': rules.get('tax_pct', 0),
        'tax': round(tax,2),
        'total_due': round(total_due,2),
        'rules_used': os.path.basename(RULES_PATH),
        'computed_at': datetime.utcnow().isoformat() + 'Z'
    }
    return breakdown


if __name__ == '__main__':
    # quick manual test
    r = compute_premium(risk_score=80, monthly_mileage=1200, is_ev=False, safe_driver=False)
    print(json.dumps(r, indent=2))
