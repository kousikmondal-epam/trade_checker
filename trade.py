"""
Portfolio Trade Order Compliance & Execution Checker
=====================================================

Purpose
-------
Evaluates proposed portfolio trades against a fixed, ordered set of
compliance / execution-readiness rules and returns, for each trade:

    decision : "Approved" | "Rejected" | "Needs Manual Sign-off"
    reason   : short human-readable explanation

The six rules are applied IN ORDER, and the first one a trade fails
(or triggers) short-circuits the rest -- exactly as specified:

    Step 1 - Validate trade data (quantity/price positive, action Buy/Sell)
    Step 2 - Restricted security check
    Step 3 - Sell check (enough holdings to sell)
    Step 4 - Buy affordability check (enough cash)
    Step 5 - Position concentration check (Buy only)
    Step 6 - Otherwise -> Approved

All "exceeds" / "insufficient" comparisons are STRICT: a value exactly
equal to a limit passes (it does not fail / does not require sign-off).

This file is self-contained and runnable. See the bottom of the file
(`if __name__ == "__main__":`) for how it ties everything together.
"""

import csv
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union



@dataclass
class Trade:
    """One proposed trade, as it would appear in trades_test.csv."""
    portfolio_id: str
    security_id: str
    action: Any          # kept as 'Any' on purpose: Step 1 must be able to
    quantity: Any         # reject non-numeric / malformed values too, so we
    price: Any             # do not coerce types before validating them.
    security_type: str = ""

    @staticmethod
    def from_csv_row(row: Dict[str, str]) -> "Trade":
        """
        Build a Trade from a raw CSV row (all strings).

        We try to convert quantity/price to numbers for convenience, but if
        a value is not a valid number we leave it as the original string so
        Step 1 (validation) can correctly reject it as "Invalid trade data."
        """
        def to_number(value: str):
            try:
                return float(value)
            except (TypeError, ValueError):
                return value  # validation step will reject it

        return Trade(
            portfolio_id=row.get("portfolio_id", ""),
            security_id=row.get("security_id", ""),
            action=row.get("action", ""),
            quantity=to_number(row.get("quantity", "")),
            price=to_number(row.get("price", "")),
            security_type=row.get("security_type", ""),
        )


@dataclass
class PortfolioReferenceData:
    """
    Reference data needed to evaluate trades for ONE portfolio.

    cash_balance         : float  - available cash
    restricted_securities: list[str] - security ids this portfolio may not trade
    max_position_pct     : float  - e.g. 0.25 means 25% maximum concentration
    current_holdings     : list[(security_id, quantity, market_price)]
    """
    cash_balance: float
    restricted_securities: List[str] = field(default_factory=list)
    max_position_pct: float = 1.0
    current_holdings: List[Tuple[str, float, float]] = field(default_factory=list)

    def holding_quantity(self, security_id: str) -> float:
        """Quantity currently held of a given security (0 if not held)."""
        for sec_id, qty, _price in self.current_holdings:
            if sec_id == security_id:
                return qty
        return 0.0

    def value_of_other_holdings(self, security_id: str) -> float:
        """Sum of (quantity * market_price) for every holding EXCEPT security_id."""
        total = 0.0
        for sec_id, qty, mkt_price in self.current_holdings:
            if sec_id != security_id:
                total += qty * mkt_price
        return total


Decision = Dict[str, str]  


def evaluate_trade(trade: Trade, portfolio: Union[PortfolioReferenceData, None]) -> Decision:
    """
    Evaluate a single trade against its portfolio's reference data,
    following Steps 1-6 in strict order.

    Parameters
    ----------
    trade     : Trade
    portfolio : PortfolioReferenceData or None
                None is only acceptable if the trade fails Step 1, since
                later steps need real reference data to be evaluated.

    Returns
    -------
    dict with keys "decision" and "reason"
    """

    # ---------------- Step 1 — Validate ----------------
    quantity = trade.quantity
    price = trade.price
    action = trade.action

    is_number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)

    valid_quantity = is_number(quantity) and quantity > 0
    valid_price = is_number(price) and price > 0
    valid_action = isinstance(action, str) and action in ("Buy", "Sell")

    if not (valid_quantity and valid_price and valid_action):
        return {"decision": "Rejected", "reason": "Invalid trade data."}


    if portfolio is None:
        # Cannot evaluate steps 2-5 without reference data for this portfolio.
        raise ValueError(
            f"No portfolio reference data available for portfolio "
            f"'{trade.portfolio_id}' — cannot evaluate steps 2-5."
        )

    # ---------------- Step 2 — Restricted security check ----------------
    if trade.security_id in portfolio.restricted_securities:
        return {
            "decision": "Rejected",
            "reason": "Security is restricted for this portfolio.",
        }

    # ---------------- Step 3 — Sell check ----------------
    if action == "Sell":
        held_quantity = portfolio.holding_quantity(trade.security_id)
        if held_quantity < quantity:   # strict: exactly equal is fine
            return {
                "decision": "Rejected",
                "reason": "Insufficient holdings to sell.",
            }
        # Sell orders never trigger the concentration check (Step 5 is Buy-only).
        return {"decision": "Approved", "reason": ""}

    # ---------------- Step 4 — Buy affordability check ----------------

    cost = quantity * price
    if cost > portfolio.cash_balance:   # strict: exactly equal is affordable
        return {
            "decision": "Rejected",
            "reason": "Insufficient cash for this buy order.",
        }

    # ---------------- Step 5 — Position concentration check (Buy only) --
    post_trade_cash = portfolio.cash_balance - cost
    existing_quantity = portfolio.holding_quantity(trade.security_id)
    post_trade_quantity = existing_quantity + quantity
    post_trade_value_of_this_security = post_trade_quantity * price
    value_of_other_holdings = portfolio.value_of_other_holdings(trade.security_id)

    total_portfolio_value_after = (
        post_trade_cash + post_trade_value_of_this_security + value_of_other_holdings
    )

    # Guard against division by zero in a degenerate reference-data setup.
    if total_portfolio_value_after > 0:
        position_pct = post_trade_value_of_this_security / total_portfolio_value_after
    else:
        position_pct = float("inf")

    if position_pct > portfolio.max_position_pct:   # strict: exactly at cap passes
        return {
            "decision": "Needs Manual Sign-off",
            "reason": "Exceeds maximum position concentration — requires approval.",
        }

    # ---------------- Step 6 — Otherwise ----------------
    return {"decision": "Approved", "reason": ""}



def load_trades_from_csv(path: str) -> List[Trade]:
    """
    Load proposed trades from a CSV file with columns:
        portfolio_id, security_id, action, quantity, price, security_type

    Returns a list of Trade objects, in file order.
    """
    trades: List[Trade] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(Trade.from_csv_row(row))
    return trades



def load_portfolio_reference_data(
    source: Union[str, Dict[str, Any]]
) -> Dict[str, PortfolioReferenceData]:
    """
    Build a {portfolio_id: PortfolioReferenceData} map from either:
      - a path to a JSON file, or
      - an already-parsed dict (same shape the JSON would have).

    Expected JSON / dict shape:
    {
      "POR001": {
        "cash_balance": 10000,
        "restricted_securities": ["XYZCORP"],
        "max_position_pct": 0.25,
        "current_holdings": [
          ["AAPL", 100, 150],
          ["TCS", 50, 310]
        ]
      },
      "POR002": { ... }
    }
    """
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = source

    result: Dict[str, PortfolioReferenceData] = {}
    for portfolio_id, data in raw.items():
        holdings = [
            (sec_id, qty, mkt_price)
            for sec_id, qty, mkt_price in data.get("current_holdings", [])
        ]
        result[portfolio_id] = PortfolioReferenceData(
            cash_balance=data["cash_balance"],
            restricted_securities=list(data.get("restricted_securities", [])),
            max_position_pct=data["max_position_pct"],
            current_holdings=holdings,
        )
    return result


EXAMPLE_PORTFOLIO_REFERENCE_DATA: Dict[str, Any] = {
    "POR001": {
        "cash_balance": 10000,
        "restricted_securities": ["XYZCORP"],
        "max_position_pct": 0.30,
        "current_holdings": [
            ["AAPL", 10, 150],
            ["TCS", 50, 310],
        ],
    },
    "POR002": {
        "cash_balance": 20000,
        "restricted_securities": [],
        "max_position_pct": 0.40,
        "current_holdings": [
            ["GOOGL", 2, 2000],
        ],
    },
    "POR003": {
        "cash_balance": 15000,
        "restricted_securities": ["BADCO"],
        "max_position_pct": 0.25,
        "current_holdings": [
            ["TCS", 5, 300],
            ["INFY", 10, 1500],
            ["RELIANCE", 30, 2400],
        ],
    },
    "POR004": {
        "cash_balance": 50000,
        "restricted_securities": [],
        "max_position_pct": 0.50,
        "current_holdings": [
            ["GOOGL", 1, 2000],
            ["AAPL", 100, 150],
        ],
    },
    "POR005": {
        "cash_balance": 8000,
        "restricted_securities": ["PENNY1"],
        "max_position_pct": 0.20,
        "current_holdings": [
            ["AAPL", 20, 150],
            ["TCS", 10, 250],
        ],
    },
}

EXAMPLE_REFERENCE_DATA_PATH = "portfolio_reference_data.json"



def run_checker(
    trades_csv_path: str,
    reference_data: Union[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Load trades from CSV and portfolio reference data (JSON path or dict),
    evaluate every trade, and return a list of result rows combining the
    original trade fields with the decision and reason.
    """
    trades = load_trades_from_csv(trades_csv_path)
    portfolios = load_portfolio_reference_data(reference_data)

    results = []
    for trade in trades:
        portfolio = portfolios.get(trade.portfolio_id)
        try:
            outcome = evaluate_trade(trade, portfolio)
        except ValueError as exc:
            outcome = {"decision": "Error", "reason": str(exc)}

        results.append(
            {
                "portfolio_id": trade.portfolio_id,
                "security_id": trade.security_id,
                "action": trade.action,
                "quantity": trade.quantity,
                "price": trade.price,
                "security_type": trade.security_type,
                "decision": outcome["decision"],
                "reason": outcome["reason"],
            }
        )
    return results


if __name__ == "__main__":
   
    TRADES_CSV_PATH = "trades_test.csv"

    rows = run_checker(TRADES_CSV_PATH, EXAMPLE_PORTFOLIO_REFERENCE_DATA)

    header = f"{'Portfolio':<10}{'Security':<10}{'Action':<6}{'Qty':>8}{'Price':>8}  {'Decision':<24}Reason"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['portfolio_id']:<10}{r['security_id']:<10}{str(r['action']):<6}"
            f"{str(r['quantity']):>8}{str(r['price']):>8}  {r['decision']:<24}{r['reason']}"
        )
