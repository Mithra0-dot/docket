"""Real risk-overview computations over scored transactions.

The linking identifiers in this module are deliberately synthetic. The source
credit-card dataset contains no device, IP, account, or email identifiers, so
these seeded identifiers are only a reproducible graph-analysis demonstration;
they must never be interpreted as recovered source data.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from itertools import combinations

FRAUD_MISSED_COST = "amount"
FALSE_POSITIVE_COST = 25.0
AMBIGUOUS_REVIEW_COST = 50.0
MIN_COMPONENT_SIZE = 4
MIN_FRAUD_COUNT = 2
MIN_AMBIGUOUS_COUNT = 3


@dataclass(frozen=True)
class RiskRow:
    transaction_id: int
    amount: float
    label: int
    score: float
    tier: str


def synthetic_identifiers(transaction_id: int) -> tuple[str, str]:
    """Assign reproducible, randomly clustered synthetic device and IP IDs.

    A keyed digest makes the grouping independent of the fraud label and
    transaction features. Both identifiers intentionally share a cluster key,
    creating connected components while remaining clearly synthetic.
    """
    digest = blake2b(
        str(transaction_id).encode("ascii"), key=b"docket-stage-5", digest_size=8
    ).digest()
    cluster = int.from_bytes(digest, "big") % 40000
    return f"synthetic-device-{cluster:05d}", f"synthetic-ip-10.5.{cluster // 256}.{cluster % 256}"


def build_graph(rows: list[RiskRow], max_nodes: int = 5000) -> dict[str, object]:
    groups: dict[tuple[str, str], list[RiskRow]] = defaultdict(list)
    for row in rows:
        groups[synthetic_identifiers(row.transaction_id)].append(row)

    flagged = []
    component_by_id: dict[int, int] = {}
    components: list[list[RiskRow]] = []
    for component_id, members in enumerate(groups.values(), start=1):
        if len(members) < MIN_COMPONENT_SIZE:
            continue
        components.append(members)
        for member in members:
            component_by_id[member.transaction_id] = component_id
        fraud_count = sum(member.label for member in members)
        ambiguous_count = sum(member.tier == "ambiguous" for member in members)
        fraud_rate = fraud_count / len(members)
        ambiguous_rate = ambiguous_count / len(members)
        if (fraud_count >= MIN_FRAUD_COUNT and fraud_rate >= 0.5) or (
            ambiguous_count >= MIN_AMBIGUOUS_COUNT and ambiguous_rate >= 0.5
        ):
            flagged.append(
                {
                    "cluster_id": component_id,
                    "size": len(members),
                    "fraud_count": fraud_count,
                    "fraud_rate": round(fraud_rate, 6),
                    "ambiguous_count": ambiguous_count,
                    "ambiguous_rate": round(ambiguous_rate, 6),
                    "transaction_ids": [member.transaction_id for member in members],
                }
            )

    flagged.sort(key=lambda cluster: (-cluster["fraud_rate"], -cluster["size"], cluster["cluster_id"]))
    selected_ids = {
        transaction_id
        for cluster in flagged
        for transaction_id in cluster["transaction_ids"]
    }
    if not selected_ids:
        selected_ids = {
            member.transaction_id
            for members in sorted(components, key=len, reverse=True)[:3]
            for member in members
        }
    selected_ids = set(sorted(selected_ids)[:max_nodes])
    row_by_id = {row.transaction_id: row for row in rows}
    nodes = []
    for transaction_id in sorted(selected_ids):
        row = row_by_id[transaction_id]
        device_id, ip_address = synthetic_identifiers(transaction_id)
        nodes.append(
            {
                "id": transaction_id,
                "label": row.label,
                "score": round(row.score, 8),
                "tier": row.tier,
                "device_id": device_id,
                "ip_address": ip_address,
                "cluster_id": component_by_id.get(transaction_id),
            }
        )
    edges = []
    for members in groups.values():
        visible = [member for member in members if member.transaction_id in selected_ids]
        device_id, ip_address = synthetic_identifiers(visible[0].transaction_id) if visible else (None, None)
        for left, right in combinations(visible, 2):
            edges.append(
                {
                    "source": left.transaction_id,
                    "target": right.transaction_id,
                    "shared_device_id": device_id,
                    "shared_ip_address": ip_address,
                }
            )
    return {
        "synthetic": True,
        "synthetic_notice": "device_id and ip_address are reproducible synthetic linking identifiers; source data has no such fields",
        "nodes": nodes,
        "edges": edges,
        "flagged_clusters": flagged,
        "total_components_at_least_minimum_size": len(components),
        "returned_node_count": len(nodes),
        "returned_edge_count": len(edges),
    }


def cost_comparison(rows: list[RiskRow]) -> dict[str, object]:
    naive = Counter()
    tiered = Counter()
    total_fraud = sum(row.label for row in rows)
    tier_detected_fraud = sum(
        row.label for row in rows if row.tier in ("auto-block", "ambiguous")
    )
    fraud_scores = sorted((row.score for row in rows if row.label == 1), reverse=True)
    equal_recall_threshold = fraud_scores[tier_detected_fraud - 1]
    equal_recall = Counter()
    for row in rows:
        naive_decision = row.score >= 0.5
        if row.label == 1 and not naive_decision:
            naive["false_negative_count"] += 1
            naive["false_negative_cost"] += row.amount
        elif row.label == 0 and naive_decision:
            naive["false_positive_count"] += 1
            naive["false_positive_cost"] += FALSE_POSITIVE_COST

        if row.tier == "ambiguous":
            tiered["ambiguous_count"] += 1
            tiered["review_cost"] += AMBIGUOUS_REVIEW_COST
        elif row.tier == "auto-block" and row.label == 0:
            tiered["false_positive_count"] += 1
            tiered["false_positive_cost"] += FALSE_POSITIVE_COST
        elif row.tier == "auto-clear" and row.label == 1:
            tiered["false_negative_count"] += 1
            tiered["false_negative_cost"] += row.amount

        equal_recall_decision = row.score >= equal_recall_threshold
        if row.label == 1 and not equal_recall_decision:
            equal_recall["false_negative_count"] += 1
            equal_recall["false_negative_cost"] += row.amount
        elif row.label == 0 and equal_recall_decision:
            equal_recall["false_positive_count"] += 1
            equal_recall["false_positive_cost"] += FALSE_POSITIVE_COST

    naive["total_cost"] = naive["false_negative_cost"] + naive["false_positive_cost"]
    tiered["total_cost"] = tiered["false_negative_cost"] + tiered["false_positive_cost"] + tiered["review_cost"]
    equal_recall["total_cost"] = equal_recall["false_negative_cost"] + equal_recall["false_positive_cost"]
    ambiguous_fraud_count = sum(
        row.label for row in rows if row.tier == "ambiguous"
    )
    ambiguous_fraud_amount = sum(
        row.amount for row in rows if row.tier == "ambiguous" and row.label == 1
    )
    review_cost = tiered["review_cost"]
    return {
        "currency": "USD (dataset Amount treated as transaction currency)",
        "assumptions": {
            "false_negative": "transaction Amount",
            "false_positive": FALSE_POSITIVE_COST,
            "ambiguous_review": AMBIGUOUS_REVIEW_COST,
            "ambiguous_outcome": "fraud labels in the reviewed tier count as caught; review cost is charged",
            "review_cost_basis": "conservative analyst handling, QA, and operations overhead per case",
        },
        "naive_flat_0_5": dict(naive),
        "naive_at_current_tiering_recall": {
            **dict(equal_recall),
            "threshold": equal_recall_threshold,
            "recall": (total_fraud - equal_recall["false_negative_count"]) / total_fraud,
        },
        "current_tiering": dict(tiered),
        "ambiguous_tier_value": {
            "fraud_count_caught": ambiguous_fraud_count,
            "fraud_amount_caught": ambiguous_fraud_amount,
            "review_cost": review_cost,
            "net_value_before_other_effects": ambiguous_fraud_amount - review_cost,
            "break_even_review_cost_per_case": ambiguous_fraud_amount / tiered["ambiguous_count"],
        },
        "equal_recall_cost_difference_current_minus_naive": tiered["total_cost"] - equal_recall["total_cost"],
    }
