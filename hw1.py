#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return the LangChain receipt-audit chains once.

    Three independent vision prompts are used later by ``answer_queries``.
    Two primary auditors run in parallel; a ledger auditor is held back as a
    tie-breaker for receipts where the primary auditors disagree.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableLambda, RunnableParallel
    from langchain_deepseek import ChatDeepSeek

    system_prompt = (
        "You are a meticulous supermarket receipt auditor. "
        "Read the image at high fidelity and return only the requested JSON."
    )

    discount_prompt = """Audit this supermarket receipt and return ONLY JSON:
{"subtotal":number|null,"rounding":number|null,"final_payment":number|null,"negative_lines":[{"label":"exact text","amount":positive number}],"discount_total":number}
Rules:
- subtotal is the amount printed on the SUBTOTAL line.
- Only examine the transaction section ABOVE the SUBTOTAL line. NEVER include any negative card balance, remaining value, points, or other text below SUBTOTAL/payment.
- Include EVERY negative monetary line above SUBTOTAL except ROUNDING: discounts, promotions, coupons, member/app offers, and packaging adjustments. Put every amount as a positive number.
- A discount label and amount may be split across adjacent lines; inspect carefully.
- discount_total is the arithmetic sum of negative_lines.
- rounding and final_payment are from the lines at or below SUBTOTAL; final_payment is the actual charged amount from Octopus/Visa/Card/Payment, never subtotal.
- Re-scan specifically for Buy X Save, % OFF, APP, COUPON, MB, member, and packaging adjustment lines.
Return JSON only, with no Markdown fences or explanation."""

    structured_prompt = """Read this supermarket receipt with extreme care. Return ONLY JSON:
{"subtotal":number|null,"rounding":number|null,"final_payment":number|null,"discount_total":number,"positive_total":number,"discount_lines":[{"label":"exact label","amount":positive number}],"arithmetic_ok":boolean}
Rules:
- subtotal is the exact number printed on the SUBTOTAL line (after discounts, before rounding).
- rounding is the signed number printed on ROUNDING, or 0 if absent.
- final_payment is the actual final amount charged after rounding, copied from the payment line such as OCTOPUS, VISA, PAID, CARD, or AMOUNT DEDUCTED. Do not use SUBTOTAL.
- discount_total is the sum of absolute amounts for every negative discount, promotion, coupon, member/app saving, packaging adjustment, or negative adjustment line. EXCLUDE ROUNDING.
- positive_total is the sum of positive product and charge line totals. Include bag charges, deposits, and positive surcharges. Exclude QTY/points/balances/change/SUBTOTAL/ROUNDING/payment lines. If a line prints QTY and one total amount, use that total rather than a unit price.
- The equations positive_total - discount_total = subtotal and subtotal + rounding = final_payment should hold to 0.01.
- If an equation fails, re-read every amount and correct the JSON before answering.
Return JSON only, with no Markdown fences or explanation."""

    ledger_prompt = """Transcribe the monetary ledger of this supermarket receipt and return ONLY JSON:
{"subtotal":number|null,"rounding":number|null,"final_payment":number|null,"monetary_lines":[{"text":"exact printed text","amount":signed number,"section":"before_subtotal" or "after_subtotal"}]}
Rules:
- Walk through every printed line in receipt order and include each monetary amount in monetary_lines. Preserve its printed sign: a line printed with '-' must have a negative amount.
- Do not include QTY numbers, percentages without a currency amount, points, card numbers, dates, or telephone numbers.
- Use section='before_subtotal' only for the product/discount transaction section above the SUBTOTAL line. Use section='after_subtotal' for SUBTOTAL, ROUNDING, payment, change, card balance, and footer information.
- Discount labels and amounts may be split across adjacent lines; inspect carefully. Include every negative line above SUBTOTAL, including packaging adjustments.
- subtotal is the amount printed on SUBTOTAL.
- rounding is the signed amount printed on ROUNDING, or 0 if absent.
- final_payment is the actual final charged amount from OCTOPUS/VISA/PAID/CARD/AMOUNT DEDUCTED, never SUBTOTAL.
Return JSON only, with no Markdown fences or explanation."""

    def make_chain(task_prompt: str):
        model = ChatDeepSeek(
            model="deepseek-v4-flash-vision-exp",
            temperature=0,
            max_retries=2,
            timeout=120,
        )

        def to_messages(payload: dict[str, Any]):
            return [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=[
                        {"type": "text", "text": task_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": payload["image_url"]},
                        },
                    ]
                ),
            ]

        return RunnableLambda(to_messages) | model

    return {
        "primary": RunnableParallel(
            discount_audit=make_chain(discount_prompt),
            structured_audit=make_chain(structured_prompt),
        ),
        "ledger_audit": make_chain(ledger_prompt),
    }

def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run the chains and return the two aggregate HKD answers.

    Receipt-level values are parsed and summed with ``Decimal`` in Python. A
    third audit is requested only when the two primary auditors disagree.
    """
    cent = Decimal("0.01")

    def money(value: Any) -> Decimal | None:
        if value is None or isinstance(value, bool):
            return None
        raw = str(value).strip()
        raw = raw.replace("HK$", "").replace("$", "").replace(",", "").strip()
        if raw.startswith("(") and raw.endswith(")"):
            raw = "-" + raw[1:-1]
        try:
            number = Decimal(raw)
        except (InvalidOperation, ValueError):
            return None
        if not number.is_finite():
            return None
        return number.quantize(cent)

    def record(value: Any) -> dict[str, Any] | None:
        if value is None or isinstance(value, BaseException):
            return None
        text = value if isinstance(value, str) else response_text(value)
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            first = text.find("{")
            last = text.rfind("}")
            if first < 0 or last <= first:
                return None
            try:
                parsed = json.loads(text[first : last + 1])
            except (json.JSONDecodeError, TypeError):
                return None
        return parsed if isinstance(parsed, dict) else None

    def listed_total(
        item: dict[str, Any],
        key: str,
        *,
        before_subtotal_only: bool = False,
    ) -> Decimal | None:
        rows = item.get(key)
        if not isinstance(rows, list):
            return None
        total = Decimal("0.00")
        for row in rows:
            if not isinstance(row, dict):
                continue
            amount = money(row.get("amount"))
            if amount is None:
                continue
            if before_subtotal_only:
                section = str(row.get("section", "")).lower()
                marker = row.get("before_subtotal")
                if marker is False or (section and "before" not in section):
                    continue
            total += abs(amount)
        return total.quantize(cent)

    def pass_values(
        kind: str, item: dict[str, Any] | None
    ) -> dict[str, Decimal] | None:
        if item is None:
            return None

        subtotal = money(item.get("subtotal"))
        rounding = money(item.get("rounding"))
        final_payment = money(item.get("final_payment"))
        if final_payment is None and subtotal is not None and rounding is not None:
            final_payment = (subtotal + rounding).quantize(cent)
        if final_payment is None:
            return None

        if kind == "discount_audit":
            discount = listed_total(item, "negative_lines")
            if subtotal is None or discount is None:
                return None
            without_discount = (subtotal + discount).quantize(cent)
        elif kind == "structured_audit":
            discount = listed_total(item, "discount_lines")
            formula_total = (
                (subtotal + discount).quantize(cent)
                if subtotal is not None and discount is not None
                else None
            )
            positive_total = money(item.get("positive_total"))
            if positive_total is not None and formula_total is not None:
                if abs(positive_total - formula_total) > cent:
                    return None
                without_discount = positive_total
            else:
                without_discount = positive_total or formula_total
            if without_discount is None:
                return None
        else:
            discount = listed_total(
                item, "monetary_lines", before_subtotal_only=True
            )
            if subtotal is None or discount is None:
                return None
            without_discount = (subtotal + discount).quantize(cent)

        if (
            subtotal is None
            or subtotal <= 0
            or without_discount < subtotal
            or without_discount > subtotal * 2
        ):
            return None

        return {
            "paid": final_payment,
            "without": without_discount,
        }

    def same(values: list[Decimal]) -> bool:
        return len({value.quantize(cent) for value in values}) <= 1

    def choose(values: list[Decimal]) -> Decimal:
        if not values:
            return Decimal("0.00")
        counts: dict[Decimal, int] = {}
        for value in values:
            key = value.quantize(cent)
            counts[key] = counts.get(key, 0) + 1
        best_count = max(counts.values())
        winners = [value for value, count in counts.items() if count == best_count]
        if len(winners) == 1:
            return winners[0]
        ordered = sorted(winners)
        return ordered[len(ordered) // 2]

    if not images:
        return {QUERY_1: "HK$0.00", QUERY_2: "HK$0.00"}

    inputs = [{"image_url": image_data_url(path)} for path in images]
    primary = chain["primary"].batch(
        inputs,
        config={"max_concurrency": 3},
        return_exceptions=True,
    )

    all_candidates: list[list[dict[str, Decimal]]] = []
    for result in primary:
        candidates: list[dict[str, Decimal]] = []
        if isinstance(result, dict):
            for kind in ("discount_audit", "structured_audit"):
                values = pass_values(kind, record(result.get(kind)))
                if values is not None:
                    candidates.append(values)
        all_candidates.append(candidates)

    unresolved = []
    for index, candidates in enumerate(all_candidates):
        if len(candidates) < 2:
            unresolved.append(index)
            continue
        if not same([row["paid"] for row in candidates]) or not same(
            [row["without"] for row in candidates]
        ):
            unresolved.append(index)

    if unresolved:
        tie_breaker = chain["ledger_audit"].batch(
            [inputs[index] for index in unresolved],
            config={"max_concurrency": 3},
            return_exceptions=True,
        )
        for index, value in zip(unresolved, tie_breaker):
            parsed = pass_values("ledger_audit", record(value))
            if parsed is not None:
                all_candidates[index].append(parsed)

    paid_total = Decimal("0.00")
    without_total = Decimal("0.00")
    for candidates in all_candidates:
        paid_candidates = [row["paid"] for row in candidates]
        without_candidates = [row["without"] for row in candidates]
        paid_total += choose(paid_candidates)
        without_total += choose(without_candidates)

    return {
        QUERY_1: f"HK${paid_total.quantize(cent):.2f}",
        QUERY_2: f"HK${without_total.quantize(cent):.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
