"""
決算発表後のドリフト（PEAD）の検証 — J-Quants API V2 の財務情報（/v2/fins/summary）を使う

仮説（先に固定）:
  良い決算・上方修正が出た銘柄は、発表後もしばらく市場（TOPIX）より上がりやすい。悪い場合は逆。
  発表を見てから買う現実的な条件で確かめるため、エントリーは「発表後に最初に買える日の始値」とする。

使うデータ:
  - 株価: sss_data_ext.json（yfinance。分割調整済み）とその中の 1306（TOPIXの代わり）
  - 決算: J-Quants /v2/fins/summary（銘柄ごと。取得結果は jq_fins.json に保存し、2回目以降は再利用）

準備:
  1. J-Quants に登録し、ダッシュボードで API キーを発行する（無料プランでも可。無料は約2年分・12週間遅れ）
  2. Claude Code の環境変数（または API credentials）に JQUANTS_API_KEY を設定する。キーをチャットやコードに貼らない
  3. stock-data 環境の許可ドメインに api.jquants.com があること

使い方:
    python sss_pead.py                 # 取得 → 集計 → pead_report.md
    python sss_pead.py --no-fetch      # 取得済みの jq_fins.json で集計だけ
    python sss_pead.py --codes 7203 6758   # 一部の銘柄だけ（動作確認用）
"""
import argparse, datetime as dt, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://api.jquants.com/v2"
FINS = os.path.join(HERE, "jq_fins.json")
HORIZONS = [1, 5, 20, 60]


# ---------------- J-Quants からの取得 ----------------
def fetch_summary(code, key, sleep):
    import requests
    out, params = [], {"code": code}
    while True:
        r = requests.get(f"{BASE}/fins/summary", headers={"x-api-key": key}, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(10); continue
        if r.status_code != 200:
            raise RuntimeError(f"{code}: HTTP {r.status_code} {r.text[:200]}")
        j = r.json()
        out += j.get("data", [])
        pk = j.get("pagination_key")
        time.sleep(sleep)
        if not pk:
            return out
        params["pagination_key"] = pk


def fetch_all(codes, sleep):
    key = os.environ.get("JQUANTS_API_KEY")
    if not key:
        sys.exit("環境変数 JQUANTS_API_KEY がありません。J-Quants のダッシュボードで発行したキーを設定してください。")
    cache = json.load(open(FINS, encoding="utf-8")) if os.path.exists(FINS) else {}
    for i, c in enumerate(codes, 1):
        if c in cache:
            continue
        try:
            cache[c] = fetch_summary(c, key, sleep)
        except Exception as e:
            print(f"  取得失敗 {c}: {e}")
            continue
        if i % 20 == 0:
            json.dump(cache, open(FINS, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {i}/{len(codes)} 銘柄")
    json.dump(cache, open(FINS, "w", encoding="utf-8"), ensure_ascii=False)
    return cache


# ---------------- 決算イベントと指標 ----------------
def num(x):
    try:
        return float(x) if x not in ("", None) else np.nan
    except (TypeError, ValueError):
        return np.nan


def pct(new, old):
    if np.isnan(new) or np.isnan(old) or old == 0:
        return np.nan
    return (new - old) / abs(old)


def build_events(rows):
    """1銘柄の開示一覧から、開示ごとに ①予想修正率 ②営業利益の前年同期比 ③来期予想の増益率 を計算"""
    rows = sorted(rows, key=lambda r: (r.get("DiscDate", ""), r.get("DiscTime", "") or ""))
    fc = {}        # 決算期末日 → 直近の通期営業利益予想
    actual = {}    # (期の種類, 期末日) → 累計営業利益の実績
    ev = []
    for r in rows:
        doc = r.get("DocType", "") or ""
        fy, per, nfy = r.get("CurFYEn", ""), r.get("CurPerType", ""), r.get("NxtFYEn", "")
        op, fop, nxfop = num(r.get("OP")), num(r.get("FOP")), num(r.get("NxFOP"))
        e = {"date": r.get("DiscDate", ""), "time": (r.get("DiscTime") or "")[:5], "doc": doc,
             "rev": np.nan, "yoy": np.nan, "guide": np.nan}
        if fy and not np.isnan(fop):
            if fy in fc:
                e["rev"] = pct(fop, fc[fy])
            fc[fy] = fop
        if nfy and not np.isnan(nxfop):
            if nfy in fc:
                e["rev"] = pct(nxfop, fc[nfy]) if np.isnan(e["rev"]) else e["rev"]
            fc[nfy] = nxfop
        if "FinancialStatements" in doc and per and fy and not np.isnan(op):
            try:
                target = dt.date.fromisoformat(fy) - dt.timedelta(days=365)
                for (p, f), v in actual.items():
                    if p == per and abs((dt.date.fromisoformat(f) - target).days) <= 20:
                        e["yoy"] = pct(op, v)
            except ValueError:
                pass
            actual[(per, fy)] = op
            if per == "FY" and not np.isnan(nxfop):
                e["guide"] = pct(nxfop, op)
        if e["date"]:
            ev.append(e)
    return ev


def classify(e):
    """先に決めた分類（変更しない）"""
    tags = []
    if e["rev"] >= 0.05: tags.append("上方修正（予想+5%以上）")
    if e["rev"] <= -0.05: tags.append("下方修正（予想−5%以下）")
    if e["yoy"] >= 0.20: tags.append("増益（前年同期比+20%以上）")
    if e["yoy"] <= -0.20: tags.append("減益（前年同期比−20%以下）")
    if e["guide"] >= 0.10: tags.append("来期増益予想（+10%以上）")
    if e["guide"] <= -0.10: tags.append("来期減益予想（−10%以下）")
    good = (e["rev"] >= 0.05) or (e["yoy"] >= 0.20 and not (e["rev"] <= -0.05))
    bad = (e["rev"] <= -0.05) or (e["yoy"] <= -0.20 and not (e["rev"] >= 0.05))
    if good and not bad: tags.append("★好決算（総合）")
    if bad and not good: tags.append("☆悪決算（総合）")
    return tags


# ---------------- 株価とリターン ----------------
def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def repair(v):
    v = v.copy()
    for i in range(1, len(v) - 1):
        if v[i - 1] and v[i] and v[i] / v[i - 1] < 0.2:
            v[i] *= round(v[i - 1] / v[i])
    return v


def entry_index(dates, disc_date, disc_time):
    """発表後に最初に買える日の位置：寄り付き前（9:00前）の発表ならその日の始値、それ以外は翌営業日の始値"""
    i = int(np.searchsorted(dates, disc_date))
    if i < len(dates) and dates[i] == disc_date and disc_time and disc_time < "09:00":
        return i
    return i + 1 if (i < len(dates) and dates[i] == disc_date) else i


def weekly_ci(df, col, base=0.0, n=4000, seed=0):
    w = df.assign(w=[tuple(dt.date.fromisoformat(str(d)).isocalendar()[:2]) for d in df.entry]).groupby("w")[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(w), len(w)); s = w.iloc[i].sum()
        out.append(s["sum"] / s["count"] - base)
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--data", default=os.path.join(HERE, "sss_data_ext.json"))
    ap.add_argument("--codes", nargs="*")
    ap.add_argument("--sleep", type=float, default=1.1, help="1リクエストごとの待ち秒数（無料プランは 12 以上）")
    a = ap.parse_args()

    data = json.load(open(a.data, encoding="utf-8"))
    dates = np.array(data["dates"])
    stocks = {s["code"]: s for s in data["stocks"]}
    codes = a.codes or list(stocks)
    fins = json.load(open(FINS, encoding="utf-8")) if a.no_fetch else fetch_all(codes, a.sleep)

    Bo, Bc = repair(arr(data["benchmark"].get("o") or data["benchmark"]["c"])), repair(arr(data["benchmark"]["c"]))
    rows = []
    for c in codes:
        if c not in stocks or c not in fins:
            continue
        s = stocks[c]; o, cl = arr(s["o"]), arr(s["c"])
        for e in build_events(fins[c]):
            i = entry_index(dates, e["date"], e["time"])
            if i <= 0 or i >= len(dates) or np.isnan(o[i]):
                continue
            rec = {"code": c, "name": s["name"], "disc": e["date"], "entry": str(dates[i]), "doc": e["doc"],
                   "rev": e["rev"], "yoy": e["yoy"], "guide": e["guide"],
                   "gap": o[i] / cl[i - 1] - 1 if cl[i - 1] else np.nan, "tags": classify(e)}
            for H in HORIZONS:
                j = i + H - 1
                if j < len(dates) and not np.isnan(cl[j]):
                    rec[f"r{H}"] = cl[j] / o[i] - 1
                    rec[f"x{H}"] = rec[f"r{H}"] - (Bc[j] / Bo[i] - 1)
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("決算イベントがありませんでした。取得結果（jq_fins.json）を確認してください。")
    df.to_csv(os.path.join(HERE, "pead_events.csv"), index=False)

    L = [f"# 決算発表後のドリフト検証（{df.disc.min()}〜{df.disc.max()}、{df.code.nunique()}銘柄・{len(df)}開示）", "",
         "エントリー＝発表後に最初に買える日の始値。超過＝同じ期間のTOPIX連動ETF（1306）との差。",
         "信頼区間＝エントリー週単位のブートストラップ4,000回の2.5〜97.5%点。配当は含まない。", ""]
    tags = sorted({t for ts in df.tags for t in ts})
    base = {H: df[f"x{H}"].mean() for H in HORIZONS}
    L += ["## 分類ごとの超過リターン（平均）", "",
          "| 分類 | 件数 | 発表直後の窓（参考・買えない） | " + " | ".join(f"{H}日" for H in HORIZONS) + " | 20日：全開示との差（95%信頼区間） | 20日でTOPIXに勝った割合 |",
          "|---|---|---|" + "---|" * len(HORIZONS) + "---|---|"]
    L.append(f"| 全開示 | {len(df)} | {df.gap.mean()*100:+.2f}% | " + " | ".join(f"{base[H]*100:+.2f}%" for H in HORIZONS) + " | — | "
             f"{(df.x20 > 0).mean()*100:.0f}% |")
    for t in tags:
        g = df[df.tags.apply(lambda ts: t in ts)]
        if len(g) < 10:
            continue
        g20 = g.dropna(subset=["x20"])
        lo, hi = weekly_ci(g20, "x20", base=base[20]) if len(g20) >= 10 else (np.nan, np.nan)
        L.append(f"| {t} | {len(g)} | {g.gap.mean()*100:+.2f}% | " + " | ".join(f"{g[f'x{H}'].mean()*100:+.2f}%" for H in HORIZONS)
                 + f" | {(g20.x20.mean()-base[20])*100:+.2f}%（{lo*100:+.2f}〜{hi*100:+.2f}%） | {(g20.x20 > 0).mean()*100:.0f}% |")
    L += ["", "## 年ごとの再現性（★好決算・20日の超過リターン）", "", "| 年 | 件数 | 好決算 | 全開示 |", "|---|---|---|---|"]
    df["year"] = df.disc.str[:4]
    for y, g in df.groupby("year"):
        gg = g[g.tags.apply(lambda ts: "★好決算（総合）" in ts)]
        L.append(f"| {y} | {len(gg)} | {gg.x20.mean()*100:+.2f}% | {g.x20.mean()*100:+.2f}% |")
    L += ["", "※判断の目安: ★好決算の「全開示との差」の信頼区間の下限がプラスで、年ごとにもおおむね全開示を上回るなら、",
          "  SSSに「好決算の翌日以降に買う」ルールを次世代の仮説として加える価値がある。分類の基準（5%・20%・10%）はこの結果を見て変えないこと。"]
    txt = "\n".join(L)
    open(os.path.join(HERE, "pead_report.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
