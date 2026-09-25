"""
SSS 紙上売買（実際には発注しない想定売買の記録）

毎営業日の引け後（15:45 JST 以降）に実行する想定:
    python sss_paper.py              # データ取得 → 当日の約定・決済 → 翌日の注文候補を記録
    python sss_paper.py --no-fetch   # 既存の sss_data.json を使う（テスト用）
    python sss_paper.py --report     # 紙上売買の成績を「全銘柄を毎日買った場合」と比較

ルール・配点は paper_config.json で固定する（検証中は変更しないこと）。
記録:
    records/paper_state.json  保有中・翌日の注文予定・損失上限の状態
    records/paper_trades.csv  決済済みの想定売買
    records/signals.csv       毎日の点数上位（シグナル単位の検証用）
    records/daily/YYYY-MM-DD.md  その日の報告
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


# ---------------- 紙上売買の1日分 ----------------
def empty_state(cfg):
    return {"last_date": None, "positions": [], "pending": [], "realized": 0.0,
            "month": None, "month_pnl": 0.0, "pause_day": False, "pause_month": None,
            "start_date": cfg["start_date"]}


def is_week_end(dates, t):
    d = dt.date.fromisoformat(dates[t])
    if d.weekday() == 4:
        return True
    if t + 1 < len(dates):
        return dt.date.fromisoformat(dates[t + 1]).weekday() <= d.weekday()
    # 最新日: 翌営業日が不明なので金曜のみ週末扱い
    return False


def step(data, R, t, state, cfg):
    """日付インデックス t の1日分を処理し、報告用の dict を返す"""
    dates = data["dates"]
    date = dates[t]
    S = data["stocks"]
    code2si = {s["code"]: i for i, s in enumerate(S)}
    r = cfg["rules"]
    tp, sl, H, cost = r["take_profit"], r["stop_loss"], r["max_hold_days"], r["cost"]
    log = {"date": date, "opened": [], "closed": [], "skipped": [], "holding": [], "paused": "", "day_pnl": 0.0}

    m = date[:7]
    if state["month"] != m:
        state["month"], state["month_pnl"] = m, 0.0

    # 1) 前日に決めた注文を今日の始値で約定
    if state["pause_day"]:
        log["paused"] = "前日に1日の損失上限に達したため、今日は新規購入なし"
    elif state["pause_month"] == m:
        log["paused"] = "今月の損失上限に達したため、今月は新規購入なし"
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
            shares = int(r["budget_per_stock"] // (op * 100)) * 100
            if shares <= 0:
                log["skipped"].append({"code": od["code"], "name": od["name"], "price100": round(op * 100)})
                continue
            p = {"code": od["code"], "name": od["name"], "sector": od["sector"], "score": od["score"],
                 "entry_date": date, "entry_t": t, "entry": op, "shares": shares,
                 "stop": op * (1 - sl), "take": op * (1 + tp)}
            state["positions"].append(p)
            log["opened"].append(p)

    # 2) 保有銘柄の決済判定（ツールと同じ順序）
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
        if ex is None and cfg["rules"]["exit_on_week_end"] and is_week_end(dates, t):
            ex, why = c, "週末の手仕舞い"
        if ex is not None:
            pnl = (ex - p["entry"]) * p["shares"] - cost * p["entry"] * p["shares"]
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
            log["holding"].append({**p, "close": c, "unreal": (c - p["entry"]) * p["shares"]})

    # 3) 損失上限
    state["pause_day"] = r["daily_loss_limit"] > 0 and log["day_pnl"] <= -r["daily_loss_limit"]
    if r["monthly_loss_limit"] > 0 and state["month_pnl"] <= -r["monthly_loss_limit"]:
        state["pause_month"] = m

    # 4) 今日の終値で点数を計算し、翌日の注文予定を作る
    today = R[R.t == t].copy()
    today["score"] = score(today, cfg["weights"], cfg["penalties"])
    today = today.sort_values("score", ascending=False)
    held = {p["code"] for p in state["positions"]}
    cands = today[(today.score >= cfg["threshold"]) & (~today.code.isin(held))]
    state["pending"] = [{"code": x.code, "name": x.name, "sector": x.sector, "score": round(float(x.score), 1)}
                        for x in cands.itertuples()]
    log["top10"] = [{"code": x.code, "name": x.name, "sector": x.sector, "score": round(float(x.score), 1),
                     "mom": x.mom, "rel": x.rel, "close": x.close} for x in today.head(10).itertuples()]
    log["pending"] = state["pending"][: r["max_positions"]]
    log["n_scored"] = len(today)
    state["last_date"] = date
    log["realized"] = state["realized"]
    log["month_pnl"] = state["month_pnl"]
    log["unreal"] = sum(h["unreal"] for h in log["holding"])
    return log


# ---------------- 記録 ----------------
def yen(x):
    return ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(round(x)):,}円"


def write_records(log, cfg):
    os.makedirs(os.path.join(REC, "daily"), exist_ok=True)
    # 決済済み
    fp = os.path.join(REC, "paper_trades.csv")
    new = not os.path.exists(fp)
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["entry_date", "exit_date", "code", "name", "sector", "score", "entry", "exit", "shares", "why", "pnl", "ret"])
        for c in log["closed"]:
            w.writerow([c["entry_date"], c["exit_date"], c["code"], c["name"], c["sector"], c["score"],
                        round(c["entry"], 2), round(c["exit"], 2), c["shares"], c["why"], round(c["pnl"]), round(c["ret"], 5)])
    # シグナル（点数上位）
    fp = os.path.join(REC, "signals.csv")
    new = not os.path.exists(fp)
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "rank", "code", "name", "sector", "score", "above_threshold"])
        for i, x in enumerate(log["top10"], 1):
            w.writerow([log["date"], i, x["code"], x["name"], x["sector"], x["score"], int(x["score"] >= cfg["threshold"])])
    # 日次レポート
    L = [f"# {log['date']} 紙上売買レポート", ""]
    L.append(f"- 今日の確定損益: {yen(log['day_pnl'])}　累計確定: {yen(log['realized'])}　含み: {yen(log['unreal'])}　今月: {yen(log['month_pnl'])}")
    if log["paused"]:
        L.append(f"- 状態: {log['paused']}")
    L += ["", "## 今日の決済"]
    L += [f"- {c['code']} {c['name']}：{c['why']}　{c['entry']:.1f}→{c['exit']:.1f}　{yen(c['pnl'])}" for c in log["closed"]] or ["- なし"]
    L += ["", "## 今日の新規購入（始値）"]
    L += [f"- {p['code']} {p['name']}：{p['entry']:.1f}円×{p['shares']}株　損切り{p['stop']:.1f}／利確{p['take']:.1f}" for p in log["opened"]] or ["- なし"]
    L += [f"- 予算オーバーで見送り：{s['code']} {s['name']}（100株 {s['price100']:,}円）" for s in log["skipped"]]
    L += ["", "## 引け後の保有"]
    L += [f"- {h['code']} {h['name']}：買値{h['entry']:.1f}　終値{h['close']:.1f}　{yen(h['unreal'])}" for h in log["holding"]] or ["- なし"]
    L += ["", f"## 翌営業日の注文予定（{cfg['threshold']}点以上・空き枠の分だけ）"]
    L += [f"- {p['code']} {p['name']}（{p['sector']}）{p['score']}点" for p in log["pending"]] or ["- なし"]
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


# ---------------- 成績レポート（基準との比較） ----------------
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


def report(data, cfg):
    R = features(data)
    start = cfg["start_date"]
    t0 = next((i for i, d in enumerate(data["dates"]) if d >= start), None)
    if t0 is None:
        print("検証開始日以降のデータがまだありません。"); return
    idx = np.where(R.t.values >= t0)[0]
    Rf = R.iloc[idx].copy()
    Rf["ret"] = simulate_ret(data, R, idx, cfg)
    Rf["score"] = score(Rf, cfg["weights"], cfg["penalties"])
    done = Rf[Rf.ret.notna()]
    base = done.ret.mean()
    sel = done[done.score >= cfg["threshold"]].sort_values(["t", "score"], ascending=[True, False]).groupby("t").head(cfg["rules"]["max_positions"])
    print(f"検証開始 {start} 〜 結果確定済み {data['dates'][int(done.t.max())] if len(done) else '—'}")
    print(f"全銘柄を毎日買った場合: 期待値 {base*100:+.3f}%（{len(done)}件）")
    if len(sel):
        se = sel.ret.std() / np.sqrt(len(sel)) * 100 if len(sel) > 1 else float('nan')
        print(f"SSSシグナル（上位{cfg['rules']['max_positions']}・{cfg['threshold']}点以上）: 期待値 {sel.ret.mean()*100:+.3f}%（{len(sel)}件、誤差の目安 ±{se:.2f}%）　勝率 {(sel.ret>0).mean()*100:.1f}%")
        print(f"  基準との差 {(sel.ret.mean()-base)*100:+.3f}%  ※件数が100件を超えるまでは判断しないこと")
    fp = os.path.join(REC, "paper_trades.csv")
    if os.path.exists(fp):
        tr = pd.read_csv(fp)
        if len(tr):
            print(f"紙上売買（資金ルール込み）: {len(tr)}回　勝率 {(tr.pnl>0).mean()*100:.1f}%　損益合計 {yen(tr.pnl.sum())}　1回平均 {yen(tr.pnl.mean())}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--force", action="store_true", help="時刻チェックを無視")
    a = ap.parse_args()
    cfg = load_config()
    data_fp = os.path.join(HERE, "sss_data.json")
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
    today_jst = dt.datetime.now(JST).date().isoformat()
    if dates[t] != today_jst and not a.no_fetch:
        print(f"今日（{today_jst}）のデータがありません。休場日の可能性があります。最新は {dates[t]}。何もしません。"); return
    if state["last_date"] and dates[t] <= state["last_date"]:
        print(f"{dates[t]} は処理済みです。"); return
    if dates[t] < cfg["start_date"]:
        print(f"検証開始日 {cfg['start_date']} より前です。"); return
    R = features(data)
    log = step(data, R, t, state, cfg)
    save_state(state)
    print(write_records(log, cfg))


if __name__ == "__main__":
    main()
