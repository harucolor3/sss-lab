"""
SSS 紙上売買（実際には発注しない想定売買の記録）

毎営業日の引け後（15:45 JST 以降）に実行する想定:
    python sss_paper.py                  # データ・決算日を取得 → 当日の約定・決済 → 翌日の注文予定を記録
    python sss_paper.py --check-earnings # 決算日が何銘柄分取れるかだけ確認する
    python sss_paper.py --report         # 成績を3つの基準（全銘柄平均・ランダム2銘柄・TOPIX）と比較
    python sss_paper.py --no-fetch       # 既存の sss_data.json / 決算日キャッシュを使う（テスト用）

ルール・配点は paper_config.json で固定する（検証中は変更しないこと）。
記録:
    records/paper_state.json      保有中・翌日の注文予定・損失上限の状態
    records/paper_trades.csv      決済済みの想定売買
    records/signals.csv           毎日の点数上位（シグナル単位の検証用）
    records/earnings_log.json     その日に分かっていた次回決算日（後から公平に再計算するため）
    records/daily/YYYY-MM-DD.md   その日の報告

v1.4（2026-09-25）4回目レビュー対応:
  - 東証の営業日カレンダー（土日・祝日・年末年始）で「翌営業日」「週の最終営業日」を判定
  - ランダム比較は75点以上の候補からランダムに選ぶ（ランキングに意味があるかの比較）
  - 75点以上の全シグナルと全銘柄の点数を当日の状態のまま records/ に保存し、研究評価は当日記録で行う
  - 実運用評価は紙上売買と同じ売買エンジンで判定。月間損益の −5万円基準を PASS/FAIL 表示
  - TOPIX比較は最初に売買できる日の始値から。決算日の取得失敗時はキャッシュ済みの未来日付を保持

v1.1（2026-09-25）フィードバック対応:
  - 株数を「1取引の許容損失」から計算（損切りまでの下げ＋コストで risk_per_trade 以内）
  - 決算回避：保有予定期間に決算がある銘柄は買わない／決算前日の終値で手仕舞い
  - 保有中に株式分割があった場合、買値・株数を分割に合わせて補正
  - --report にランダム2銘柄（同条件）・TOPIX・週単位ブートストラップの95%信頼区間を追加
"""
import argparse, csv, datetime as dt, json, os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "records")
JST = dt.timezone(dt.timedelta(hours=9))
FN = ["トレンド", "対市場", "出来高", "5日騰落", "安定性", "押し目"]


def load_config():
    with open(os.path.join(HERE, "paper_config.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------- 要素の計算（ツール sss-lab.html と同じ定義） ----------------
def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def features(data):
    """全銘柄・全日の要素を計算して DataFrame で返す（t は日付インデックス）"""
    B = arr(data["benchmark"]["c"])
    T = len(data["dates"])
    out = []
    for si, s in enumerate(data["stocks"]):
        o, h, l, c, v = (arr(s[k]) for k in "ohlcv")
        cs = pd.Series(c)
        ma5 = cs.rolling(5).mean().values
        ma25 = cs.rolling(25).mean().values
        ma25p = pd.Series(ma25).shift(5).values
        c20, c5, c1, c2 = (cs.shift(k).values for k in (20, 5, 1, 2))
        b20 = pd.Series(B).shift(20).values
        rel = (c / c20 - 1) - (B / b20 - 1)
        vol20 = pd.Series(v).shift(1).rolling(20, min_periods=15).mean().values
        tv20 = pd.Series(v * c).shift(1).rolling(20, min_periods=15).mean().values
        vr = v / vol20
        mom = c / c5 - 1
        tr = np.fmax(np.fmax(h - l, np.abs(h - c1)), np.abs(l - c1))
        atr = pd.Series(tr).rolling(14, min_periods=10).mean().values / c
        gap = o / c1 - 1
        prev = c1 / c2 - 1
        conds = (c > ma5).astype(int) + (ma5 > ma25).astype(int) + (ma25 > ma25p).astype(int)
        up25 = ma25 > ma25p
        df = pd.DataFrame({
            "t": np.arange(T), "si": si, "code": s["code"], "name": s["name"],
            "sector": s.get("sector", "その他"),
            "conds": conds, "rel": rel, "vr": vr, "mom": mom, "atr": atr, "tv": tv20,
            "gap": gap, "prev": prev, "up25": up25, "close": c,
        })
        need = [c, o, h, l, v, ma5, ma25, ma25p, rel, vr, mom, atr, prev]
        ok = np.all([~np.isnan(x) for x in need], axis=0)
        ok[:60] = False
        out.append(df[ok])
    R = pd.concat(out, ignore_index=True)
    R["f0"] = R.conds.map({0: 0, 1: .2, 2: .6, 3: 1})
    R["f1"] = np.select([R.rel <= -.05, R.rel < 0, R.rel < .03, R.rel < .06], [0, .2, .5, .8], 1)
    R["f2"] = np.select([R.vr < .8, R.vr < 1.2, R.vr < 1.5, R.vr < 2], [0, .25, .5, .75], 1)
    R["f3"] = np.select([R.mom <= -.03, R.mom < 0, R.mom < .03, R.mom < .07, R.mom < .12], [0, .2, .533, 1, .667], .2)
    R["f4"] = np.select([R.atr <= .015, R.atr <= .025, R.atr <= .035], [1, .67, .33], 0)
    R["f5"] = np.where(~R.up25, 0, np.where(R.mom <= -.06, .3, np.where(R.mom <= -.01, 1, np.where(R.mom < 0, .5, 0))))
    R["fl"] = (R.mom >= .2) * 1 + (R.tv < 1e9) * 2 + (R.gap >= .04) * 4 + (R.prev >= .10) * 8
    return R


def score(R, w, pen):
    w = np.array(w, float)
    lo, hi = np.minimum(w, 0).sum(), np.maximum(w, 0).sum()
    raw = R[[f"f{i}" for i in range(6)]].values @ w
    sc = (raw - lo) / ((hi - lo) or 1) * 100
    fl = R.fl.values
    for j in range(4):
        sc = sc - pen[j] * ((fl >> j) & 1)
    return sc


# ---------------- 決算日 ----------------
# ---------------- 東証の営業日カレンダー ----------------
_FALLBACK_HOLIDAYS = {  # jpholiday が使えない場合の予備（2026〜2027年の祝日）
    "2026-01-01", "2026-01-12", "2026-02-11", "2026-02-23", "2026-03-20", "2026-04-29", "2026-05-03",
    "2026-05-04", "2026-05-05", "2026-05-06", "2026-07-20", "2026-08-11", "2026-09-21", "2026-09-22",
    "2026-09-23", "2026-10-12", "2026-11-03", "2026-11-23",
    "2027-01-01", "2027-01-11", "2027-02-11", "2027-02-23", "2027-03-21", "2027-03-22", "2027-04-29",
    "2027-05-03", "2027-05-04", "2027-05-05", "2027-07-19", "2027-08-11", "2027-09-20", "2027-09-23",
    "2027-10-11", "2027-11-03", "2027-11-23"}
try:
    import jpholiday
except ImportError:
    jpholiday = None


def tse_open(d):
    """東証が開いている日か（土日・祝日・12/31・1/1〜1/3 は休み）"""
    if isinstance(d, str):
        d = dt.date.fromisoformat(d)
    if d.weekday() >= 5 or (d.month == 12 and d.day == 31) or (d.month == 1 and d.day <= 3):
        return False
    if jpholiday is not None:
        return not jpholiday.is_holiday(d)
    if d.year > 2027:
        print("警告: jpholiday が無く、2028年以降の祝日を判定できません（pip install jpholiday）", file=sys.stderr)
    return d.isoformat() not in _FALLBACK_HOLIDAYS


def busday(date_str, n):
    """date_str から n 営業日後（東証カレンダー）。n=0 なら当日以降の最初の営業日"""
    d = dt.date.fromisoformat(date_str)
    if n == 0:
        while not tse_open(d):
            d += dt.timedelta(days=1)
        return d.isoformat()
    k = 0
    while k < n:
        d += dt.timedelta(days=1)
        if tse_open(d):
            k += 1
    return d.isoformat()


def fetch_next_earnings(codes, today):
    """yfinance から各銘柄の次回決算発表日を取得。{code: 'YYYY-MM-DD' or None}"""
    import yfinance as yf
    out = {}
    for c in codes:
        nxt = None
        try:
            cal = yf.Ticker(f"{c}.T").calendar
            ds = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if ds:
                ds = [str(pd.Timestamp(d).date()) for d in (ds if isinstance(ds, (list, tuple)) else [ds])]
                fut = sorted(d for d in ds if d >= today)
                nxt = fut[0] if fut else None
        except Exception:
            nxt = None
        out[c] = nxt
    return out


def resolve_earnings(fetched, cache, today, recent_days=60):
    """取得結果とキャッシュから、その日の判断に使う {code: 次回決算日 | "SAFE" | None} を作る。
    次回が未発表でも、直近の決算が60日以内に終わっていれば次回は当分先なので "SAFE"。"""
    new_cache, eff = {}, {}
    t = dt.date.fromisoformat(today)
    for code, nxt in fetched.items():
        old = cache.get(code, {}) if isinstance(cache.get(code), dict) else {}
        last = old.get("last")
        if old.get("next") and old["next"] < today:
            last = old["next"]
        if nxt is None and old.get("next") and old["next"] >= today:
            nxt = old["next"]          # 取得に失敗しても、キャッシュ済みの未来の決算日は保持
        if nxt and nxt < today:
            last, nxt = nxt, None
        new_cache[code] = {"next": nxt, "last": last}
        if nxt:
            eff[code] = nxt
        elif last and (t - dt.date.fromisoformat(last)).days <= recent_days:
            eff[code] = "SAFE"
        else:
            eff[code] = None
    return eff, new_cache


def earnings_block(code, date_t, earn, cfg):
    """date_t の引け後に、翌営業日に買ってよいか。(ok, 理由)"""
    rule = cfg["rules"]
    d = earn.get(code) if earn is not None else None
    if d == "SAFE":
        return True, ""
    if d is None:
        if rule.get("earnings_unknown", "skip") == "skip":
            return False, "決算日不明"
        return True, ""
    entry = busday(date_t, 1)
    last = busday(entry, rule["max_hold_days"] - 1)
    if entry <= d <= last:
        return False, f"保有予定期間に決算（{d}）"
    return True, ""


# ---------------- 紙上売買の1日分 ----------------
def empty_state(cfg):
    return {"last_date": None, "positions": [], "pending": [], "realized": 0.0,
            "month": None, "month_pnl": 0.0, "pause_day": False, "pause_month": None,
            "start_date": cfg["start_date"]}


def is_week_end(dates, t):
    """その週の最終営業日か（金曜が祝日なら木曜が最終営業日）。東証カレンダーで判定"""
    d = dt.date.fromisoformat(dates[t])
    nxt = dt.date.fromisoformat(busday(dates[t], 1))
    return nxt.isocalendar()[:2] != d.isocalendar()[:2]


def position_size(price, cfg, cash):
    """損失額ベースの株数（100株単位）。0 なら買えない"""
    r = cfg["rules"]
    per_share_risk = price * r["stop_loss"] + price * r["cost"]
    by_risk = int(r["risk_per_trade"] // (per_share_risk * 100)) * 100
    by_budget = int(r["budget_per_stock"] // (price * 100)) * 100
    by_cash = int(max(cash, 0) // (price * 100)) * 100
    return min(by_risk, by_budget, by_cash), (by_risk, by_budget, by_cash)


def raw_factor(stock, t):
    """過去期間の検証用：分割調整済み株価を当時の実際の株価に戻す倍率（rawF があるときだけ。通常運用では常に1）"""
    f = stock.get("rawF")
    return f[t] if f else 1.0


def step(data, R, t, state, cfg, earn=None, picker=None, pool="threshold"):
    """日付インデックス t の1日分を処理し、報告用の dict を返す。
    earn: {code: 次回決算日} その日の引け後に分かっている情報。
    picker: 候補の並べ方（None=点数順）。pool: "threshold"=75点以上が候補／"all"=全銘柄が候補（比較用）"""
    dates = data["dates"]
    date = dates[t]
    S = data["stocks"]
    code2si = {s["code"]: i for i, s in enumerate(S)}
    r = cfg["rules"]
    tp, sl, H, cost = r["take_profit"], r["stop_loss"], r["max_hold_days"], r["cost"]
    log = {"date": date, "opened": [], "closed": [], "skipped": [], "holding": [], "paused": "",
           "day_pnl": 0.0, "notes": [], "earn_skipped": []}

    m = date[:7]
    if state["month"] != m:
        state["month"], state["month_pnl"] = m, 0.0

    # 0) 株式分割の補正（データは分割調整済みなので、保有中に分割があると買値とずれる）
    for p in state["positions"]:
        if p["entry_date"] in dates and "F" not in p:
            adj = S[code2si[p["code"]]]["o"][dates.index(p["entry_date"])]
            if adj and abs(adj / p["entry"] - 1) > 0.02:
                f = adj / p["entry"]
                p["entry"] *= f; p["stop"] *= f; p["take"] *= f
                p["shares"] = int(round(p["shares"] / f))
                log["notes"].append(f"{p['code']} {p['name']}：株式分割を検出（×{1/f:.2f}）。買値・株数を補正")

    # 1) 前日に決めた注文を今日の始値で約定
    if state["pause_day"]:
        log["paused"] = "前日に1日の損失の停止基準に達したため、今日は新規購入なし"
    elif state["pause_month"] == m:
        log["paused"] = "今月の損失の停止基準に達したため、今月は新規購入なし"
    if not log["paused"]:
        for od in state["pending"]:
            if len(state["positions"]) >= r["max_positions"]:
                break
            if any(p["code"] == od["code"] for p in state["positions"]):
                continue
            si = code2si.get(od["code"])
            if si is None:
                continue
            op = S[si]["o"][t]
            if op is None:
                continue
            F = raw_factor(S[si], t)
            invested = sum(p["entry"] * p.get("F", 1.0) * p["shares"] for p in state["positions"])
            cash = r["capital"] + state["realized"] - invested
            shares, parts = position_size(op * F, cfg, cash)
            if shares <= 0:
                why = ["許容損失", "1銘柄の上限金額", "資金不足"][int(np.argmin(parts))]
                log["skipped"].append({"code": od["code"], "name": od["name"], "price100": round(op * F * 100), "why": why})
                continue
            p = {"code": od["code"], "name": od["name"], "sector": od["sector"], "score": od["score"],
                 "entry_date": date, "entry": op, "shares": shares,
                 "stop": op * (1 - sl), "take": op * (1 + tp),
                 "risk": round((op * sl + op * cost) * F * shares)}
            if F != 1.0:
                p["F"] = F
            state["positions"].append(p)
            log["opened"].append(p)

    # 2) 保有銘柄の決済判定（ツールと同じ順序）＋決算前の手仕舞い
    still = []
    for p in state["positions"]:
        s = S[code2si[p["code"]]]
        o, h, l, c = s["o"][t], s["h"][t], s["l"][t], s["c"][t]
        if None in (o, h, l, c):
            still.append(p)
            continue
        entry_t = dates.index(p["entry_date"])
        ex, why = None, ""
        if entry_t < t:
            if o <= p["stop"]:
                ex, why = o, "損切り（窓開け）"
            elif o >= p["take"]:
                ex, why = o, "利確（窓開け）"
        if ex is None:
            if l <= p["stop"]:
                ex, why = p["stop"], "損切り"
            elif h >= p["take"]:
                ex, why = p["take"], "利確"
        if ex is None and t - entry_t + 1 >= H:
            ex, why = c, "保有期限"
        if ex is None and r["exit_on_week_end"] and is_week_end(dates, t):
            ex, why = c, "週末の手仕舞い"
        if ex is None and earn is not None:
            d = earn.get(p["code"])
            if d not in (None, "SAFE") and d <= busday(date, 1):
                ex, why = c, f"決算前の手仕舞い（決算 {d}）"
        if ex is not None:
            pnl = ((ex - p["entry"]) * p["shares"] - cost * p["entry"] * p["shares"]) * p.get("F", 1.0)
            log["closed"].append({**p, "exit_date": date, "exit": ex, "why": why, "pnl": pnl,
                                  "ret": ex / p["entry"] - 1 - cost})
            log["day_pnl"] += pnl
        else:
            still.append(p)
    state["positions"] = still
    state["realized"] += log["day_pnl"]
    state["month_pnl"] += log["day_pnl"]
    for p in still:
        c = S[code2si[p["code"]]]["c"][t]
        if c is not None:
            log["holding"].append({**p, "close": c, "unreal": (c - p["entry"]) * p["shares"] * p.get("F", 1.0)})

    # 3) 損失上限
    state["pause_day"] = r["daily_loss_limit"] > 0 and log["day_pnl"] <= -r["daily_loss_limit"]
    if r["monthly_loss_limit"] > 0 and state["month_pnl"] <= -r["monthly_loss_limit"]:
        state["pause_month"] = m

    # 4) 今日の終値で点数を計算し、翌日の注文予定を作る（決算回避を適用）
    today = R[R.t == t].copy()
    today["score"] = score(today, cfg["weights"], cfg["penalties"])
    today = today.sort_values("score", ascending=False)
    held = {p["code"] for p in state["positions"]}
    base = today[~today.code.isin(held)]
    if pool == "threshold":
        base = base[base.score >= cfg["threshold"]]
    cands = base if picker is None else picker(base)
    pend = []
    for x in cands.itertuples():
        ok, why = earnings_block(x.code, date, earn, cfg) if earn is not None else (True, "")
        if not ok:
            log["earn_skipped"].append({"code": x.code, "name": x.name, "why": why})
            continue
        pend.append({"code": x.code, "name": x.name, "sector": x.sector, "score": round(float(x.score), 1)})
    state["pending"] = pend
    log["scored"] = today[["code", "name", "sector", "score", "f0", "f1", "f2", "f3", "f4", "f5", "fl", "close", "mom", "rel"]].to_dict("records")
    log["top10"] = [{"code": x.code, "name": x.name, "sector": x.sector, "score": round(float(x.score), 1),
                     "mom": x.mom, "rel": x.rel, "close": x.close} for x in today.head(10).itertuples()]
    log["pending"] = pend
    # 表示用：今日の終値で「翌朝買えそうか」を見積もる（実際の株数は翌朝の始値で決まる）
    invested = sum(p["entry"] * p.get("F", 1.0) * p["shares"] for p in state["positions"])
    cash = r["capital"] + state["realized"] - invested
    free = r["max_positions"] - len(state["positions"])
    view, n_ok = [], 0
    for od in pend:
        c = S[code2si[od["code"]]]["c"][t]
        c = c * raw_factor(S[code2si[od["code"]]], t) if c else c
        sh, parts = position_size(c, cfg, cash) if c else (0, (0, 0, 0))
        ok = sh > 0
        why = "" if ok else ["許容損失", "1銘柄の上限金額", "資金不足"][int(np.argmin(parts))]
        view.append({**od, "est_shares": sh, "price100": round((c or 0) * 100), "why": why})
        if ok:
            n_ok += 1
            cash -= c * sh
        if n_ok >= max(free, 0) and ok:
            break
    log["pending_view"] = view
    state["last_date"] = date
    log["realized"] = state["realized"]
    log["month_pnl"] = state["month_pnl"]
    log["unreal"] = sum(h["unreal"] for h in log["holding"])
    return log


# ---------------- 記録 ----------------
def yen(x):
    return ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(round(x)):,}円"


def write_records(log, cfg, earn):
    os.makedirs(os.path.join(REC, "daily"), exist_ok=True)
    fp = os.path.join(REC, "paper_trades.csv")
    new = not os.path.exists(fp)
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["entry_date", "exit_date", "code", "name", "sector", "score", "entry", "exit", "shares", "why", "pnl", "ret"])
        for c in log["closed"]:
            w.writerow([c["entry_date"], c["exit_date"], c["code"], c["name"], c["sector"], c["score"],
                        round(c["entry"], 2), round(c["exit"], 2), c["shares"], c["why"], round(c["pnl"]), round(c["ret"], 5)])
    # 当日の状態のまま記録（後から再計算しない）：
    #   signals.csv   = 75点以上の全シグナル（1日の数制限なし）
    #   scores/日付.csv = 全銘柄の点数と要素（研究評価の母集団）
    r = cfg["rules"]
    limit = r["risk_per_trade"] / (r["stop_loss"] + r["cost"]) / 100
    rows = sorted(log["scored"], key=lambda x: -x["score"])
    fp = os.path.join(REC, "signals.csv")
    new = not os.path.exists(fp)
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "rank", "code", "name", "sector", "score", "f_trend", "f_rel", "f_vol", "f_mom", "f_stab", "f_pull",
                        "penalty_flags", "close", "buyable_at_close", "earnings_next", "earnings_ok", "in_order_list"])
        order = {p["code"] for p in log["pending"]}
        for i, x in enumerate(rows, 1):
            if x["score"] < cfg["threshold"]:
                break
            ok, _ = earnings_block(x["code"], log["date"], earn, cfg) if earn is not None else (True, "")
            w.writerow([log["date"], i, x["code"], x["name"], x["sector"], round(x["score"], 2),
                        *[round(x[f"f{j}"], 3) for j in range(6)], int(x["fl"]), x["close"],
                        int(x["close"] <= limit), (earn or {}).get(x["code"]), int(ok), int(x["code"] in order)])
    os.makedirs(os.path.join(REC, "scores"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(REC, "scores", f"{log['date']}.csv"), index=False)
    # その日に分かっていた決算日（report で公平に再計算するため）
    fp = os.path.join(REC, "earnings_log.json")
    hist = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {}
    hist[log["date"]] = earn
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False)

    L = [f"# {log['date']} 紙上売買レポート", ""]
    L.append(f"- 今日の確定損益: {yen(log['day_pnl'])}　累計確定: {yen(log['realized'])}　含み: {yen(log['unreal'])}　今月: {yen(log['month_pnl'])}")
    if log["paused"]:
        L.append(f"- 状態: {log['paused']}")
    L += [f"- 注記: {n}" for n in log["notes"]]
    known = sum(1 for v in earn.values() if v and v != "SAFE") if earn else 0
    safe = sum(1 for v in earn.values() if v == "SAFE") if earn else 0
    L.append(f"- 決算日: 次回が分かる {known}銘柄／直近に決算済みで当分なし {safe}銘柄／不明（買わない） {len(earn) - known - safe if earn else 0}銘柄")
    L += ["", "## 今日の決済"]
    L += [f"- {c['code']} {c['name']}：{c['why']}　{c['entry']:.1f}→{c['exit']:.1f}　{yen(c['pnl'])}" for c in log["closed"]] or ["- なし"]
    L += ["", "## 今日の新規購入（始値）"]
    L += [f"- {p['code']} {p['name']}：{p['entry']:.1f}円×{p['shares']}株　損切り{p['stop']:.1f}／利確{p['take']:.1f}　想定最大損失 約{p['risk']:,}円" for p in log["opened"]] or ["- なし"]
    L += [f"- 見送り：{s['code']} {s['name']}（100株 {s['price100']:,}円・{s['why']}）" for s in log["skipped"]]
    L += ["", "## 引け後の保有"]
    L += [f"- {h['code']} {h['name']}：買値{h['entry']:.1f}　終値{h['close']:.1f}　{yen(h['unreal'])}" for h in log["holding"]] or ["- なし"]
    L += ["", f"## 翌営業日の注文予定（{cfg['threshold']}点以上・決算回避後。買えない銘柄は飛ばして次の順位を採用）"]
    L += [(f"- {p['code']} {p['name']}（{p['sector']}）{p['score']}点：約{p['est_shares']}株 買える見込み" if p["est_shares"] > 0
           else f"- （飛ばす）{p['code']} {p['name']} {p['score']}点：100株 {p['price100']:,}円・{p['why']}")
          for p in log.get("pending_view", [])] or ["- なし"]
    if log["earn_skipped"]:
        L += [f"- 決算回避で除外：" + "、".join(f"{x['code']} {x['name']}（{x['why']}）" for x in log["earn_skipped"][:10])]
    L += ["", "## 点数上位10", "", "| 順位 | 銘柄 | 業種 | 点数 | 5日騰落 | 対TOPIX20日 |", "|---|---|---|---|---|---|"]
    L += [f"| {i} | {x['code']} {x['name']} | {x['sector']} | {x['score']} | {x['mom']*100:+.1f}% | {x['rel']*100:+.1f}% |"
          for i, x in enumerate(log["top10"], 1)]
    L += ["", f"※実際の発注は行っていません。配点 {cfg['weights']}（{'/'.join(FN)}）、固定ルール。"]
    with open(os.path.join(REC, "daily", f"{log['date']}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return "\n".join(L)


def load_state(cfg):
    fp = os.path.join(REC, "paper_state.json")
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    return empty_state(cfg)


def save_state(state):
    os.makedirs(REC, exist_ok=True)
    with open(os.path.join(REC, "paper_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1, default=float)


# ---------------- 成績レポート（3つの基準と比較） ----------------
def simulate_ret(data, R, idx, cfg):
    r = cfg["rules"]
    tp, sl, H, cost = r["take_profit"], r["stop_loss"], r["max_hold_days"], r["cost"]
    T = len(data["dates"])
    out = np.full(len(idx), np.nan)
    for j, (si, t) in enumerate(zip(R.si.values[idx], R.t.values[idx])):
        e = t + 1
        if e + H - 1 >= T:
            continue
        s = data["stocks"][si]
        o, h, l, c = s["o"], s["h"], s["l"], s["c"]
        if o[e] is None:
            continue
        en = o[e]; stop = en * (1 - sl); take = en * (1 + tp); ex = None
        for d in range(e, e + H):
            if None in (o[d], h[d], l[d], c[d]):
                ex = None; break
            if d > e:
                if o[d] <= stop: ex = o[d]; break
                if o[d] >= take: ex = o[d]; break
            if l[d] <= stop: ex = stop; break
            if h[d] >= take: ex = take; break
            if d == e + H - 1: ex = c[d]
        if ex is not None:
            out[j] = ex / en - 1 - cost
    return out


def run_portfolio(data, R, t0, cfg, earn_hist, picker=None, pool="threshold"):
    """紙上売買と同じ売買エンジンで t0 から最後まで運用し、(最終損益, 決済一覧) を返す"""
    state = empty_state(cfg); total_closed = []
    last = None
    for t in range(t0, len(data["dates"])):
        earn = earn_hist.get(data["dates"][t]) if earn_hist else None
        last = step(data, R, t, state, cfg, earn=earn, picker=picker, pool=pool)
        total_closed += last["closed"]
    unreal = last["unreal"] if last else 0.0
    return state["realized"] + unreal, total_closed


def weekly_bootstrap_ci(sel, base, dates, n=5000, seed=0, key="t"):
    """sel の平均 − base の平均 の95%信頼区間（週単位のブロックで再標本化した2.5〜97.5%点）"""
    wk = lambda v: dt.date.fromisoformat(dates[v] if key == "t" else v).isocalendar()[:2]
    s = sel.assign(w=[wk(v) for v in sel[key]]).groupby("w").ret.agg(["sum", "count"])
    if isinstance(base, (int, float)):
        W = s.index; b = None
    else:
        b = base.assign(w=[wk(v) for v in base[key]]).groupby("w").ret.agg(["sum", "count"])
        W = b.index.union(s.index); b = b.reindex(W, fill_value=0)
    s = s.reindex(W, fill_value=0)
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        i = rng.integers(0, len(W), len(W))
        ss = s.iloc[i].sum()
        if ss["count"] == 0:
            continue
        bm = base if b is None else (b.iloc[i].sum()["sum"] / max(b.iloc[i].sum()["count"], 1))
        diffs.append(ss["sum"] / ss["count"] - bm)
    return np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)


def load_recorded_scores(dates, t0):
    """records/scores/*.csv（当日の点数の記録）を読み込む。無ければ None"""
    d = os.path.join(REC, "scores")
    if not os.path.isdir(d):
        return None
    frames = []
    for fn in sorted(os.listdir(d)):
        day = fn[:-4]
        if day in dates and dates.index(day) >= t0:
            f = pd.read_csv(os.path.join(d, fn), dtype={"code": str})
            f["t"] = dates.index(day)
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else None


def report(data, cfg, n_random=200):
    R = features(data)
    dates = data["dates"]
    start = cfg["start_date"]
    t0 = next((i for i, d in enumerate(dates) if d >= start), None)
    if t0 is None:
        print("検証開始日以降のデータがまだありません。"); return
    fp = os.path.join(REC, "earnings_log.json")
    earn_hist = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {}
    days = len(dates) - t0
    rr = cfg["rules"]; thr = cfg["threshold"]
    tr_fp = os.path.join(REC, "paper_trades.csv")
    trades = pd.read_csv(tr_fp, dtype={"code": str}) if os.path.exists(tr_fp) else pd.DataFrame()

    # ---- 研究評価：当日に記録した点数（無ければ再計算）で、全96銘柄の銘柄選択力を見る ----
    rec = load_recorded_scores(dates, t0)
    src = "当日の記録"
    if rec is None:
        rec = R[R.t >= t0].copy(); rec["score"] = score(rec, cfg["weights"], cfg["penalties"]); src = "再計算（当日の記録なし）"
    code2si = {s["code"]: i for i, s in enumerate(data["stocks"])}
    rec = rec[rec.code.isin(code2si)].copy()
    rec["si"] = rec.code.map(code2si)
    key = pd.MultiIndex.from_arrays([R.si.values, R.t.values])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([rec.si.values, rec.t.values]))
    rec = rec[pos >= 0]; pos = pos[pos >= 0]
    rec["ret"] = simulate_ret(data, R, pos, cfg)
    done = rec[rec.ret.notna()]
    sig_all = rec[rec.score >= thr]
    sig_done = done[done.score >= thr]
    top = lambda d: d.sort_values(["t", "score"], ascending=[True, False]).groupby("t").head(rr["max_positions"])
    months = days / 21
    ev = cfg.get("evaluation", {"interim": {"months": 3, "signals": 300}, "full": {"months": 6, "signals": 500, "closed_trades": 100}})
    if months >= ev["full"]["months"] and len(sig_all) >= ev["full"]["signals"] and len(trades) >= ev["full"]["closed_trades"]:
        stage = "本格評価"
    elif months >= ev["interim"]["months"] and len(sig_all) >= ev["interim"]["signals"]:
        stage = "暫定評価（本格評価の条件: 6か月・全シグナル500件・決済済み100回）"
    else:
        stage = "異常がないか確認する段階（暫定評価の条件: 3か月・全シグナル300件）"
    print(f"■ 検証開始 {start}（{days}営業日 ≒ {months:.1f}か月経過）　全シグナル {len(sig_all)}件・決済済み {len(trades)}回 → 【{stage}】")
    print(f"【研究評価】全96銘柄での銘柄選択力（点数の出どころ: {src}。差の95%信頼区間は週単位ブートストラップ5,000回の2.5〜97.5%点）")
    for label, sel in [("全シグナル（1日の数制限なし）", sig_done), (f"上位{rr['max_positions']}", top(sig_done))]:
        if len(sel) == 0:
            print(f"  {label}: 結果が確定したシグナルはまだありません"); continue
        lo, hi = weekly_bootstrap_ci(sel, done, dates)
        print(f"  {label}: {len(sel)}件 SSS {sel.ret.mean()*100:+.3f}% vs 全銘柄 {done.ret.mean()*100:+.3f}%"
              f"　差 {(sel.ret.mean()-done.ret.mean())*100:+.3f}%　95%信頼区間 {lo*100:+.2f}〜{hi*100:+.2f}%")

    # ---- 実運用評価：紙上売買と同じ売買エンジン（翌朝始値・許容損失・資金・決算回避・週末手仕舞い）----
    print("【実運用評価・必須】紙上売買と同じ売買エンジンで比較")
    sss_pnl, sss_closed = run_portfolio(data, R, t0, cfg, earn_hist)
    # B1: 75点以上の候補をランダムな順で買う（ランキングに意味があるか）
    b1 = []
    for k in range(n_random):
        rng = np.random.default_rng(1000 + k)
        b1.append(run_portfolio(data, R, t0, cfg, earn_hist, picker=lambda df, rng=rng: df.iloc[rng.permutation(len(df))])[0])
    b1 = np.array(b1)
    top_pct = (b1 >= sss_pnl).mean() * 100
    print(f"  ① 75点以上からランダム（{n_random}回）: SSS {yen(sss_pnl)}　ランダム中央値 {yen(np.median(b1))}"
          f"（5〜95%: {yen(np.percentile(b1,5))}〜{yen(np.percentile(b1,95))}）　SSSは{'上位1%未満' if top_pct < 1 else f'上位 {top_pct:.0f}%'}")
    # B2: 点数を使わず全銘柄からランダム（銘柄選択そのものに意味があるか）
    b2_trades = []
    for k in range(max(n_random // 2, 1)):
        rng = np.random.default_rng(5000 + k)
        _, cl = run_portfolio(data, R, t0, cfg, earn_hist, picker=lambda df, rng=rng: df.iloc[rng.permutation(len(df))], pool="all")
        b2_trades += [c["ret"] for c in cl]
    if sss_closed and b2_trades:
        st = pd.DataFrame({"t": [dates.index(c["entry_date"]) for c in sss_closed], "ret": [c["ret"] for c in sss_closed]})
        b2m = float(np.mean(b2_trades))
        lo, hi = weekly_bootstrap_ci(st, b2m, dates)
        print(f"  ② 点数を使わず全銘柄からランダム: 1取引の期待値 SSS {st.ret.mean()*100:+.3f}%（{len(st)}回） vs ランダム {b2m*100:+.3f}%"
              f"　差の95%信頼区間 {lo*100:+.2f}〜{hi*100:+.2f}%")
    else:
        print("  ② 決済済みの取引がまだありません")
    # 月間損益（−5万円基準）
    if len(trades):
        mon = trades.groupby(trades.exit_date.str[:7]).pnl.sum()
        worst = mon.min()
        verdict = "PASS" if worst > -rr["monthly_loss_limit"] else "FAIL"
        print(f"  ③ 月間損益（確定ベース）: 最低 {yen(worst)}（{mon.idxmin()}）　月−{rr['monthly_loss_limit']:,}円基準: {verdict}")
        print("     " + "　".join(f"{m}: {yen(v)}" for m, v in mon.items()))
        print(f"  記録済みの紙上売買: {len(trades)}回　勝率 {(trades.pnl>0).mean()*100:.1f}%　損益合計 {yen(trades.pnl.sum())}　1回の損失の最大値 {yen(trades.pnl.min())}")
    else:
        print("  ③ 月間損益: 決済済みの取引がまだありません")
    # C: TOPIX（最初に売買できる日の始値から）
    B = data["benchmark"]; cap = rr["capital"]
    b_open = B.get("o"); t_first = t0 + 1
    if t_first < len(dates):
        base_px = b_open[t_first] if b_open and b_open[t_first] else B["c"][t0]
        topix = cap * (B['c'][-1] / base_px - 1)
        print(f"  ④ TOPIXの買い持ち（{cap:,}円を{dates[t_first]}の{'始値' if b_open else '前日終値'}で買って持ち続ける）: {yen(topix)}"
              f"　SSS {yen(sss_pnl)} → {'PASS（手間とリスクに見合う）' if sss_pnl > topix else 'FAIL（持ち続ける方が良かった）'}")
    print("※配当は0円として計算（権利落ちの下落のみ損失に含む保守的な評価）")
    print("判定: 研究評価は信頼区間の下限がプラスで「優位性あり」の候補。実資金へ進むには実運用評価①上位5%以内・②下限がプラス・③PASS・④TOPIXの買い持ちに勝つ がすべて必要")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check-earnings", action="store_true")
    ap.add_argument("--force", action="store_true", help="時刻チェックを無視")
    a = ap.parse_args()
    cfg = load_config()
    data_fp = os.path.join(HERE, "sss_data.json")
    cache_fp = os.path.join(REC, "earnings_cache.json")
    today_jst = dt.datetime.now(JST).date().isoformat()

    if a.check_earnings:
        from sss_collect import DEFAULT_STOCKS
        earn = fetch_next_earnings(list(DEFAULT_STOCKS), today_jst)
        ok = {c: d for c, d in earn.items() if d}
        print(f"決算日を取得できた銘柄: {len(ok)} / {len(earn)}")
        for c, d in sorted(ok.items(), key=lambda x: x[1])[:15]:
            print(f"  {c} {DEFAULT_STOCKS[c][0]}: {d}")
        miss = [c for c, d in earn.items() if not d]
        if miss:
            print("取得できなかった銘柄:", " ".join(miss))
        return

    if not a.no_fetch and not a.report:
        now = dt.datetime.now(JST)
        if not a.force and now.hour * 60 + now.minute < 15 * 60 + 45:
            sys.exit(f"まだ取引時間中の可能性があります（{now:%H:%M} JST）。15:45以降に実行してください。")
        subprocess.run([sys.executable, os.path.join(HERE, "sss_collect.py"), "--years", "1", "--out", data_fp], check=True)
    with open(data_fp, encoding="utf-8") as f:
        data = json.load(f)
    if a.report:
        report(data, cfg); return
    state = load_state(cfg)
    dates = data["dates"]
    t = len(dates) - 1
    if dates[t] != today_jst and not a.no_fetch:
        print(f"今日（{today_jst}）のデータがありません。休場日の可能性があります。最新は {dates[t]}。何もしません。"); return
    if state["last_date"] and dates[t] <= state["last_date"]:
        print(f"{dates[t]} は処理済みです。"); return
    if dates[t] < cfg["start_date"]:
        print(f"検証開始日 {cfg['start_date']} より前です。"); return
    codes = [s["code"] for s in data["stocks"]]
    cache = json.load(open(cache_fp, encoding="utf-8")) if os.path.exists(cache_fp) else {}
    if a.no_fetch:
        fetched = {c: (cache.get(c) or {}).get("next") for c in codes}
    else:
        fetched = fetch_next_earnings(codes, dates[t])
    earn, cache = resolve_earnings(fetched, cache, dates[t])
    os.makedirs(REC, exist_ok=True)
    json.dump(cache, open(cache_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    R = features(data)
    log = step(data, R, t, state, cfg, earn=earn)
    save_state(state)
    print(write_records(log, cfg, earn))


if __name__ == "__main__":
    main()
