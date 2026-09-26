"""
学習前の期間（押し目型を作るときに一度も見ていない期間）での検証

押し目型は 2021-09-27〜2026-09-25 のデータを見て作ったため、その期間の成績は後出しになる。
このスクリプトはそれより前（既定 2011-01-01〜2021-09-24）で、固定した設定（paper_config.json）を
一切変えずに試す。紙上売買（フォワードテスト）の記録には触れない。

使い方（Claude Code の stock-data 環境で）:
    pip install -r requirements.txt --break-system-packages -q
    python sss_backtest_pre.py                     # データ取得から検証まで（10〜20分程度）
    python sss_backtest_pre.py --random 20         # ランダム比較の回数を減らして速く
    python sss_backtest_pre.py --data sss_data_pre.json   # 取得済みデータを使う

出力: 画面表示と pre_period_report.md（records/ には書き込まない）

注意:
  - 決算日の履歴がないため、この検証では決算回避を行わない（紙上売買より少し不利にも有利にもなり得る）
  - 対象は「今の」大型株96銘柄（当時上場していない銘柄は上場後から）。今も残っている会社だけなので、成績は良く出やすい（生存者の偏り）。SSSと比較対象の両方に同じ偏りがかかる
  - 株数・金額は分割履歴から当時の実際の株価に戻して計算する
"""
import argparse, datetime as dt, json, os, subprocess, sys
import numpy as np
import pandas as pd
import sss_paper as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def add_raw_factor(data):
    """分割の全履歴から、各日の「当時の株価 ÷ 調整後株価」を計算して rawF に入れる"""
    dates = data["dates"]; T = len(dates)
    for s in data["stocks"]:
        f = np.ones(T)
        for d, r in s.get("splits", []):
            i = int(np.searchsorted(np.array(dates), d))   # 分割日以降は分割後の株価
            f[:min(i, T)] *= r
        s["rawF"] = f.tolist() if (f != 1).any() else None
    return data


def run_engine(data, groups, empty, t0, cfg, picker=None, pool="threshold"):
    state = sp.empty_state(cfg); closed = []; eq = []; last = None
    for t in range(t0, len(data["dates"])):
        day = groups.get(t, empty)
        last = sp.step(data, day, t, state, cfg, earn=None, picker=picker, pool=pool)
        closed += last["closed"]
        eq.append(state["realized"] + last["unreal"])
    return eq, closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "sss_data_pre.json"))
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2021-09-25", help="この日は含まない（押し目型の作成に使った期間の開始日の前まで）")
    ap.add_argument("--random", type=int, default=50, help="75点以上からランダムに選ぶ運用の回数")
    a = ap.parse_args()
    cfg = sp.load_config()
    rr = cfg["rules"]; thr = cfg["threshold"]

    if not os.path.exists(a.data):
        subprocess.run([sys.executable, os.path.join(HERE, "sss_collect.py"), "--start", a.start, "--end", a.end,
                        "--full-splits", "--out", a.data], check=True)
    data = add_raw_factor(json.load(open(a.data, encoding="utf-8")))
    dates = data["dates"]
    if dates[-1] >= "2021-09-27":
        sys.exit("このデータは押し目型の作成に使った期間（2021-09-27以降）を含んでいます。--end を 2021-09-25 以前にしてください。")
    print(f"期間 {dates[0]}〜{dates[-1]}（{len(dates)}営業日）・{len(data['stocks'])}銘柄。設定は paper_config.json のまま（押し目型・75点・+5%/−2.5%・5日）")

    R = sp.features(data)
    R["score"] = sp.score(R, cfg["weights"], cfg["penalties"])
    t0 = 60
    R = R[R.t >= t0].reset_index(drop=True)
    R["ret"] = sp.simulate_ret(data, R, np.arange(len(R)), cfg)
    R["year"] = [dates[t][:4] for t in R.t]
    done = R[R.ret.notna()]
    sig = done[done.score >= thr]
    top2 = sig.sort_values(["t", "score"], ascending=[True, False]).groupby("t").head(rr["max_positions"])
    L = [f"# 学習前期間での検証（{dates[0]}〜{dates[-1]}）", "",
         "設定は paper_config.json のまま。決算回避なし・配当0円・生存者の偏りあり。", ""]

    # ---- 研究評価（全96銘柄の銘柄選択力）----
    L += ["## 研究評価（シグナル単位・全銘柄平均との比較）", "",
          "| 対象 | 件数 | SSS | 全銘柄 | 差 | 差の95%信頼区間 |", "|---|---|---|---|---|---|"]
    for label, sel in [("全シグナル（75点以上・数制限なし）", sig), ("上位2", top2)]:
        lo, hi = sp.weekly_bootstrap_ci(sel, done, dates)
        L.append(f"| {label} | {len(sel)} | {sel.ret.mean()*100:+.3f}% | {done.ret.mean()*100:+.3f}% | "
                 f"{(sel.ret.mean()-done.ret.mean())*100:+.3f}% | {lo*100:+.2f}〜{hi*100:+.2f}% |")
    L += ["", "### 年ごと（上位2）", "", "| 年 | 件数 | SSS | 全銘柄 | 差 |", "|---|---|---|---|---|"]
    beat = 0; yrs = sorted(done.year.unique())
    for y in yrs:
        s_ = top2[top2.year == y]; b_ = done[done.year == y]
        if len(s_) == 0:
            L.append(f"| {y} | 0 | — | {b_.ret.mean()*100:+.3f}% | — |"); continue
        d = s_.ret.mean() - b_.ret.mean(); beat += d > 0
        L.append(f"| {y} | {len(s_)} | {s_.ret.mean()*100:+.3f}% | {b_.ret.mean()*100:+.3f}% | {d*100:+.3f}% |")
    L.append(f"\n全銘柄平均を上回った年: {beat} / {len(yrs)}")

    # ---- 実運用評価（紙上売買と同じ売買エンジン。当時の実際の株価で100株判定）----
    print("売買エンジンで運用中…（ランダム比較を含むため時間がかかります）")
    groups = {t: g for t, g in R.groupby("t")}
    empty = R.iloc[0:0]
    eq, closed = run_engine(data, groups, empty, t0, cfg)
    final = eq[-1]
    rnd = []
    for k in range(a.random):
        rng = np.random.default_rng(1000 + k)
        e, _ = run_engine(data, groups, empty, t0, cfg, picker=lambda df, rng=rng: df.iloc[rng.permutation(len(df))])
        rnd.append(e[-1])
        print(f"  ランダム {k+1}/{a.random}", end="\r")
    rnd = np.array(rnd)
    b2 = []
    for k in range(max(a.random // 2, 1)):
        rng = np.random.default_rng(5000 + k)
        _, cl = run_engine(data, groups, empty, t0, cfg, picker=lambda df, rng=rng: df.iloc[rng.permutation(len(df))], pool="all")
        b2 += [c["ret"] for c in cl]
    tr = pd.DataFrame(closed)
    L += ["", "## 実運用評価（紙上売買と同じ売買エンジン・資金60万円・許容損失5,000円・当時の株価で100株判定）", ""]
    if len(tr):
        tr["t"] = [dates.index(d) for d in tr.entry_date]
        top_pct = (rnd >= final).mean() * 100
        L.append(f"- 最終損益: {sp.yen(final)}（{len(tr)}回・勝率 {(tr.pnl>0).mean()*100:.1f}%・1回の損失の最大値 {sp.yen(tr.pnl.min())}）")
        L.append(f"- ① 75点以上からランダム（{a.random}回）: 中央値 {sp.yen(np.median(rnd))}、5〜95% {sp.yen(np.percentile(rnd,5))}〜{sp.yen(np.percentile(rnd,95))} → SSSは{'上位1%未満' if top_pct < 1 else f'上位 {top_pct:.0f}%'}")
        if b2:
            m = float(np.mean(b2)); lo, hi = sp.weekly_bootstrap_ci(tr[["t", "ret"]], m, dates)
            L.append(f"- ② 全銘柄からランダム: 1取引の期待値 SSS {tr.ret.mean()*100:+.3f}% vs ランダム {m*100:+.3f}% → 差の95%信頼区間 {lo*100:+.2f}〜{hi*100:+.2f}%")
        mon = tr.groupby(tr.exit_date.str[:7]).pnl.sum()
        L.append(f"- ③ 月間損益の最低: {sp.yen(mon.min())}（{mon.idxmin()}）→ 月−{rr['monthly_loss_limit']:,}円基準: {'PASS' if mon.min() > -rr['monthly_loss_limit'] else 'FAIL'}"
                 f"（マイナスの月 {int((mon<0).sum())} / {len(mon)}）")
        peak = np.maximum.accumulate(np.array(eq)); L.append(f"- 最大の落ち込み: {sp.yen((np.array(eq)-peak).min())}")
        B = data["benchmark"]; bo = B.get("o")
        base_px = bo[t0 + 1] if bo and bo[t0 + 1] else B["c"][t0]
        L.append(f"- ④ TOPIX連動ETFに{rr['capital']:,}円（{dates[t0+1]}から）: {sp.yen(rr['capital']*(B['c'][-1]/base_px-1))}")
        yr = tr.groupby(tr.exit_date.str[:4]).pnl.agg(["sum", "count"])
        L += ["", "### 年ごとの損益（確定ベース）", "", "| 年 | 取引数 | 損益 |", "|---|---|---|"]
        L += [f"| {y} | {int(r['count'])} | {sp.yen(r['sum'])} |" for y, r in yr.iterrows()]
    else:
        L.append("- 取引がありませんでした（100株買える銘柄がなかった可能性）")
    L += ["", "※この結果で配点や75点などを変えないこと。変える場合は次世代の検証として別に行う。"]
    txt = "\n".join(L)
    open(os.path.join(HERE, "pre_period_report.md"), "w", encoding="utf-8").write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
