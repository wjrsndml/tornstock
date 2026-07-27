"""04_dividends.py — 权益块(Benefit Block)分红 ROI 排名。

假设与口径:
  - 每只股票只能持有 1 个权益块(持股 ≥ 所需股数即享分红,多持不多得);
  - 现金分红直接计价;物品分红按 ITEM_VALUES 估值(★ 需用户按当前市价校准);
  - Energy / Nerve / Happy 不可变现,只标注不估值;
  - Passive 型(永久加成)不产生现金流,不列入现金 ROI;
  - 总预期收益 = 分红年化 + 该股 3 年价格 CAGR(历史漂移,不保证未来)。

输出 analysis/output/dividends/roi_table.csv + 分档购买建议。

用法: .venv/bin/python analysis/04_dividends.py
"""

import pandas as pd

from common import ensure_out, load_stock

# ── 权益块定义(来源:README §6)─────────────────────────
# (所需股数, 周期天, 分红类型, 分红内容/数量)
BLOCKS = {
    "ASS": (1_000_000, 7, "item", "Six Pack of Alcohol"),
    "BAG": (3_000_000, 7, "item", "Ammunition Pack"),
    "CNC": (7_500_000, 31, "cash", 80_000_000),
    "EVL": (100_000, 7, "points_na", "1,000 Happy(不可变现)"),
    "EWM": (1_000_000, 7, "item", "Box of Grenades"),
    "FHG": (2_000_000, 7, "item", "Feathery Hotel Coupon"),
    "GRN": (500_000, 31, "cash", 4_000_000),
    "CBD": (350_000, 7, "points_na", "50 Nerve(不可变现)"),
    "HRG": (10_000_000, 31, "item", "Random Property"),
    "IOU": (3_000_000, 31, "cash", 12_000_000),
    "LAG": (750_000, 7, "item", "Lawyer Business Card"),
    "LSC": (500_000, 7, "item", "Lottery Voucher"),
    "MCS": (350_000, 7, "points_na", "100 Energy(不可变现)"),
    "MUN": (5_000_000, 7, "item", "Six Pack of Energy Drink"),
    "PRN": (1_000_000, 7, "item", "Erotic DVD"),
    "PTS": (10_000_000, 7, "item", "100 Points"),
    "SYM": (500_000, 7, "item", "Drug Pack"),
    "TCT": (100_000, 31, "cash", 1_000_000),
    "THS": (150_000, 7, "item", "Box of Medical Supplies"),
    "TMI": (6_000_000, 31, "cash", 25_000_000),
    "TSB": (3_000_000, 31, "cash", 50_000_000),
    "TCC": (7_500_000, 31, "item", "Clothing Cache"),
}

# ── 物品估值(★ 粗略估计,请按当前游戏内市价校准后重跑)──
ITEM_VALUES = {
    "Six Pack of Alcohol": 3_000,
    "Ammunition Pack": 8_000_000,
    "Box of Grenades": 10_000_000,
    "Feathery Hotel Coupon": 15_000_000,
    "Random Property": 20_000_000,        # 随机房产期望值,波动极大
    "Lawyer Business Card": 1_500_000,
    "Lottery Voucher": 100_000,
    "Six Pack of Energy Drink": 30_000,
    "Erotic DVD": 400_000,
    "100 Points": 4_500_000,              # 按 ~$45k/point
    "Drug Pack": 3_000_000,               # 随机药品期望
    "Box of Medical Supplies": 1_000_000,
    "Clothing Cache": 10_000_000,
}

TIERS = {"<$1B": 1e9, "$1B-10B": 10e9, ">$10B": float("inf")}


def main() -> None:
    outdir = ensure_out("dividends")
    baseline = pd.read_csv(outdir.parent / "stats" / "baseline.csv", index_col=0)

    rows = []
    for sym, (shares, period, kind, content) in BLOCKS.items():
        price = float(load_stock(sym)["price"].iloc[-1])
        cost = shares * price
        if kind == "cash":
            value = float(content)
        elif kind == "item":
            value = float(ITEM_VALUES[content])
        else:  # 不可变现
            value = 0.0
        payout_per_year = value * (365.25 / period)
        div_yield = payout_per_year / cost
        px_cagr = float(baseline.loc[sym, "cagr"])
        rows.append({
            "stock": sym, "shares": shares, "period_d": period,
            "price": round(price, 2), "cost": cost,
            "payout_type": kind, "payout": str(content),
            "payout_value": value,
            "div_yield_ann": div_yield,
            "price_cagr_3y": px_cagr,
            "total_exp_ann": div_yield + px_cagr,
        })

    df = pd.DataFrame(rows).sort_values("div_yield_ann", ascending=False)
    df.to_csv(outdir / "roi_table.csv", index=False)

    show = df.assign(
        cost_m=(df.cost / 1e6).round(0),
        div_y=(df.div_yield_ann * 100).round(2),
        px_y=(df.price_cagr_3y * 100).round(2),
        tot_y=(df.total_exp_ann * 100).round(2),
    )[["stock", "cost_m", "period_d", "payout", "div_y", "px_y", "tot_y"]]
    show.columns = ["stock", "成本($M)", "周期d", "分红", "分红年化%", "价格CAGR%", "合计年化%"]
    print("=== 权益块 ROI 排名(按分红年化)===")
    print(show.to_string(index=False))

    print("\n=== 分资金档位建议(按分红年化从高到低,买得起就买,一块一股)===")
    for tier, budget in TIERS.items():
        cash_rows = df[df.payout_type.isin(["cash", "item"])].sort_values(
            "div_yield_ann", ascending=False)
        picked, spent = [], 0.0
        for _, r in cash_rows.iterrows():
            if spent + r.cost <= budget:
                picked.append(r)
                spent += r.cost
        if not picked:
            print(f"\n[{tier}] 预算内买不起任何权益块")
            continue
        p = pd.DataFrame(picked)
        weighted_div = (p.div_yield_ann * p.cost).sum() / p.cost.sum()
        print(f"\n[{tier}] 买入 {len(p)} 块, 总成本 ${spent/1e9:.2f}B, "
              f"组合分红年化 {weighted_div*100:.2f}%")
        print("   " + ", ".join(
            f"{r.stock}({r.div_yield_ann*100:.1f}%)" for r in picked))

    print("\n★ 物品分红估值为粗略估计,校准 ITEM_VALUES 后重跑以更新排名")
    print(f"输出: {outdir / 'roi_table.csv'}")


if __name__ == "__main__":
    main()
