import pandas as pd

from src import config

GROUP_COLS = ["policy_state", "insured_sex", "age_bracket", "incident_state"]
MIN_GROUP_SIZE = 100      # smaller groups are shown but left out of the ratios
FOUR_FIFTHS = 0.80


def _r(v):
    return None if pd.isna(v) else round(float(v), 3)


def load_audit() -> pd.DataFrame:
    audit = pd.read_csv(config.AUDIT_PATH)
    audit["flagged"] = audit["score"] >= config.THRESHOLD
    audit["age_bracket"] = pd.cut(
        audit["age"], bins=[0, 29, 39, 49, 100],
        labels=["<30", "30-39", "40-49", "50+"],
    ).astype(str)
    return audit


def group_table(audit: pd.DataFrame, col: str) -> pd.DataFrame:
    g = audit.groupby(col).agg(
        n=("flagged", "size"),
        flag_rate=("flagged", "mean"),
        fraud_rate=("fraud", "mean"),
    )
    # share of each group's real fraud that was flagged
    g["catch_rate"] = audit[audit["fraud"] == 1].groupby(col)["flagged"].mean()
    # share of each group's honest claims that were flagged anyway
    g["false_flag_rate"] = audit[audit["fraud"] == 0].groupby(col)["flagged"].mean()
    g["reliable"] = g["n"] >= MIN_GROUP_SIZE
    if col == "age_bracket":
        g = g.reindex(["<30", "30-39", "40-49", "50+"])
    return g


def build_report(audit: pd.DataFrame | None = None) -> dict:
    if audit is None:
        audit = load_audit()

    report = {
        "threshold": config.THRESHOLD,
        "min_group_size": MIN_GROUP_SIZE,
        "overall_flag_rate": _r(audit["flagged"].mean()),
        "groups": {},
    }
    for col in GROUP_COLS:
        g = group_table(audit, col)
        big = g[g["reliable"]]

        flag_ratio = not_flagged_ratio = None
        if len(big) >= 2:
            flag_ratio = _r(big["flag_rate"].min() / big["flag_rate"].max())
            not_flagged_ratio = _r((1 - big["flag_rate"]).min() / (1 - big["flag_rate"]).max())

        rows = []
        for name, r in g.iterrows():
            rows.append({
                "group": str(name),
                "n": int(r["n"]),
                "flag_rate": _r(r["flag_rate"]),
                "fraud_rate": _r(r["fraud_rate"]),
                "catch_rate": _r(r["catch_rate"]),
                "false_flag_rate": _r(r["false_flag_rate"]),
                "reliable": bool(r["reliable"]),
            })

        report["groups"][col] = {
            "rows": rows,
            "flag_rate_ratio": flag_ratio,
            "not_flagged_ratio": not_flagged_ratio,
            "passes_four_fifths": (not_flagged_ratio >= FOUR_FIFTHS)
                                  if not_flagged_ratio is not None else None,
        }
    return report


def main():
    report = build_report()
    print(f"Threshold {report['threshold']} | overall flag rate {report['overall_flag_rate']}")
    for col, info in report["groups"].items():
        print(f"\n=== {col} ===")
        print(pd.DataFrame(info["rows"]).to_string(index=False))
        print(f"flag-rate ratio: {info['flag_rate_ratio']} | "
              f"not-flagged ratio: {info['not_flagged_ratio']} | "
              f"passes 80% rule: {info['passes_four_fifths']}")


if __name__ == "__main__":
    main()