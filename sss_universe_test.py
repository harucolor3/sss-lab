"""
対象銘柄を増やすと結果が変わるかの検証（研究用。紙上売買の設定・記録には触れない）

比較するデータ:
  既定   : sss_data.json      … 96銘柄（押し目型を作るときに使った銘柄）
  拡大版 : sss_data_ext.json  … 96銘柄＋追加候補（売買が活発なプライム株。直近60日の平均売買代金10億円以上）

検証内容（両方のデータで同じことをする）:
  1. 直近60営業日のテスト（紙上売買と同じ売買エンジン・決算回避なし）
     均等／専門家案／押し目型、直近60日より前で最適化した配点10通り、75点以上からランダム、TOPIX
     資金・同時保有の組み合わせ: 60万×2, 60万×3, 100万×3, 100万×5（許容損失5,000円は据え置き）
  2. 追加銘柄だけでの押し目型の研究評価（押し目型を作るときに一度も見ていない銘柄での検証）

使い方（Claude Code の stock-data 環境、15:45以降に）:
    pip install -r requirements.txt --break-system-packages -q
    python sss_universe_test.py            # データ取得から（合計20〜40分程度）
    python sss_universe_test.py --trials 5 # 最適化の回数を減らして速く
出力: 画面と universe_report.md
"""
import argparse, copy, datetime as dt, json, os, subprocess, sys, time
import numpy as np
import pandas as pd
import sss_paper as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REF = [("均等", [20, 20, 20, 20, 20, 0], 75), ("専門家案", [25, 25, 20, 15, 15, 0], 75), ("押し目型", [0, 30, 0, -15, 15, 40], 75)]
SETTINGS = [(600000, 2), (600000, 3), (1000000, 3), (1000000, 5)]


def load(path, universe, refresh):
    if refresh or not os.path.exists(path):
        cmd = [sys.executable, os.path.join(HERE, "sss_collect.py"), "--years", "5", "--out", path]
        if universe == "extended":
            cmd += ["--universe", "extended"]
        subprocess.run(cmd, check=True)
    data = json.load(open(path, encoding="utf-8"))
    # 取引時間中に取得したデータなら最終日（途中の値）を除く
    gen = data.get("generated", "")
    if gen[:10] == data["dates"][-1] and gen[11:16] < "15:45":
        n = len(data["dates"]) - 1
        data["dates"] = data["dates"][:n]
        data["benchmark"] = {k: (v[:n] if isinstance(v, list) else v) for k, v in data["benchmark"].items()}
        for s in data["stocks"]:
            for k in "ohlcv":
                s[k] = s[k][:n]
        print(f"  {os.path.basename(path)}: 最終日は取引時間中のデータのため除外")
    return data


class Lab:
    def __init__(self, data, cfg):
        self.data, self.cfg = data, cfg
        self.dates = data["dates"]; self.T = len(self.dates)
        H = cfg["rules"]["max_hold_days"]
        R = sp.features(data); R = R[R.t >= 60].reset_index(drop=True)
        R["ret"] = sp.simulate_ret(data, R, np.arange(len(R)), cfg)
        self.R = R
        self.F = R[[f"f{i}" for i in range(6)]].values
        pen = np.array(cfg["penalties"]); fl = R.fl.values
        self.pen = sum(pen[j] * ((fl >> j) & 1) for j in range(4))
        self.t = R.t.values; self.ret = R.ret.values
        self.test_start = self.T - 60
        self.train = (self.t < self.test_start - H - 1) & ~np.isnan(self.ret)
        self.groups = {t: g for t, g in R[R.t >= self.test_start - 1].groupby("t")}
        added = {s["code"] for s in data["stocks"] if s.get("added")}
        self.added = R.code.isin(added).values

    def scores(self, w):
        w = np.array(w, float); lo, hi = np.minimum(w, 0).sum(), np.maximum(w, 0).sum()
        return (self.F @ w - lo) / ((hi - lo) or 1) * 100 - self.pen

    def top2(self, s, mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return idx
        o = idx[np.lexsort((-s[idx], self.t[idx]))]
        tt = self.t[o]
        first = np.r_[True, tt[1:] != tt[:-1]]
        rank = np.arange(len(o)) - np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
        return o[rank < self.cfg["rules"]["max_positions"]]

    def objective(self, w, thr, minn=400):
        s = self.scores(w); sel = self.top2(s, self.train & (s >= thr))
        return (self.ret[sel].mean() if len(sel) >= minn else -9.0), len(sel)

    def climb(self, w, thr, maxstep=20):
        obj, _ = self.objective(w, thr)
        for _ in range(maxstep):
            best = (obj, None)
            for i in range(6):
                for d in (-20, -10, -5, 5, 10, 20):
                    w2 = list(w); w2[i] = max(-50, min(50, w2[i] + d))
                    if w2 == w or not any(w2):
                        continue
                    o, _ = self.objective(w2, thr)
                    if o > best[0]:
                        best = (o, (w2, thr))
            for d in (-10, -5, 5, 10):
                t2 = min(90, max(55, thr + d)); o, _ = self.objective(w, t2)
                if o > best[0]:
                    best = (o, (list(w), t2))
            if best[1] is None:
                break
            obj = best[0]; w, thr = best[1]
        return w, thr, obj

    def engine(self, w, thr, capital, npos, picker=None):
        c = copy.deepcopy(self.cfg); c["weights"] = list(map(float, w)); c["threshold"] = thr
        c["rules"].update(capital=capital, max_positions=npos)
        st = sp.empty_state(c); last = None; n = 0
        for t in range(self.test_start, self.T):
            last = sp.step(self.data, self.groups.get(t, self.R.iloc[0:0]), t, st, c, earn=None, picker=picker)
            n += len(last["closed"])
        return st["realized"] + last["unreal"], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--random", type=int, default=15)
    ap.add_argument("--refresh", action="store_true", help="データを取り直す")
    a = ap.parse_args()
    cfg = sp.load_config()
    L = ["# 対象銘柄を増やした場合の検証", "",
         "紙上売買の設定・記録には触れない研究用の検証。直近60営業日は紙上売買と同じ売買エンジン（許容損失5,000円・決算回避なし・配当0円）。", ""]
    labs = {}
    for label, path, uni in [("既定96銘柄", os.path.join(HERE, "sss_data.json"), "default"),
                             ("拡大版", os.path.join(HERE, "sss_data_ext.json"), "extended")]:
        print(f"■ {label} を準備中…")
        data = load(path, uni, a.refresh)
        labs[label] = Lab(data, cfg)
        lab = labs[label]
        limit = cfg["rules"]["risk_per_trade"] / (cfg["rules"]["stop_loss"] + cfg["rules"]["cost"]) / 100
        last_close = np.array([s["c"][lab.test_start] or np.nan for s in data["stocks"]], float)
        L.append(f"- {label}: {len(data['stocks'])}銘柄（うち追加 {sum(1 for s in data['stocks'] if s.get('added'))}）、"
                 f"検証開始日（{lab.dates[lab.test_start]}）に100株買えた銘柄（株価{limit:,.0f}円以下） {int((last_close <= limit).sum())}")
    L.append("")

    # 1. 直近60日
    rng0 = np.random.default_rng(7)
    starts = [([int(x) for x in rng0.choice(np.arange(-50, 55, 5), 6)], int(rng0.choice([65, 70, 75, 80]))) for _ in range(a.trials)]
    for label, lab in labs.items():
        print(f"■ {label}: 最適化 {a.trials}回（直近60日より前のデータのみ使用）")
        t1 = time.time(); opt = []
        for k, (w0, th0) in enumerate(starts, 1):
            w, th, o = lab.climb(w0, th0); opt.append((w, th, o))
            print(f"  {k}/{a.trials} {w} {th}点 学習期間 {o*100:+.3f}%（{time.time()-t1:.0f}秒）")
        L += [f"## {label}：直近60営業日（{lab.dates[lab.test_start]}〜{lab.dates[-1]}）の損益", "",
              "| 資金×同時保有 | 最適化平均（最低〜最高） | 均等 | 専門家案 | 押し目型 | 75点以上ランダム中央値 | TOPIX同額 |",
              "|---|---|---|---|---|---|---|"]
        B = lab.data["benchmark"]
        for cap, npos in SETTINGS:
            ov = [lab.engine(w, th, cap, npos)[0] for w, th, _ in opt]
            ref = {n: lab.engine(w, th, cap, npos) for n, w, th in REF}
            rnd = [lab.engine(REF[2][1], 75, cap, npos, picker=lambda d, r=np.random.default_rng(300 + i): d.iloc[r.permutation(len(d))])[0]
                   for i in range(a.random)]
            topix = cap * (B["c"][-1] / B["c"][lab.test_start - 1] - 1)
            L.append(f"| {cap//10000}万×{npos} | {sp.yen(np.mean(ov))}（{sp.yen(min(ov))}〜{sp.yen(max(ov))}） | {sp.yen(ref['均等'][0])} | "
                     f"{sp.yen(ref['専門家案'][0])} | {sp.yen(ref['押し目型'][0])}（{ref['押し目型'][1]}回） | {sp.yen(np.median(rnd))} | {sp.yen(topix)} |")
            print(f"  {cap//10000}万×{npos}: 押し目型 {sp.yen(ref['押し目型'][0])} 最適化平均 {sp.yen(np.mean(ov))}")
        L.append("")

    # 2. 追加銘柄だけでの研究評価（押し目型を作るときに見ていない銘柄）
    lab = labs["拡大版"]
    if lab.added.any():
        s = lab.scores(REF[2][1]); ok = ~np.isnan(lab.ret)
        dates = lab.dates
        base = pd.DataFrame({"t": lab.t[ok & lab.added], "ret": lab.ret[ok & lab.added]})
        L += ["## 追加銘柄だけでの押し目型の成績（研究評価・全期間）", "",
              "押し目型を作るときに一度も見ていない銘柄なので、後出しになりにくい検証。", "",
              "| 対象 | 件数 | 押し目型 | 追加銘柄の平均 | 差 | 差の95%信頼区間 |", "|---|---|---|---|---|---|"]
        allsig = ok & lab.added & (s >= 75)
        for name, idx in [("75点以上の全シグナル", np.where(allsig)[0]), ("上位2", lab.top2(s, allsig))]:
            sel = pd.DataFrame({"t": lab.t[idx], "ret": lab.ret[idx]})
            if len(sel) == 0:
                L.append(f"| {name} | 0 | — | — | — | — |"); continue
            lo, hi = sp.weekly_bootstrap_ci(sel, base, dates)
            L.append(f"| {name} | {len(sel)} | {sel.ret.mean()*100:+.3f}% | {base.ret.mean()*100:+.3f}% | "
                     f"{(sel.ret.mean()-base.ret.mean())*100:+.3f}% | {lo*100:+.2f}〜{hi*100:+.2f}% |")
        L.append("")
    L.append("※押し目型は既定96銘柄の全期間を見て作った設定のため、既定96銘柄の直近60日の成績は有利に出る。紙上売買の設定はこの結果で変えない。")
    txt = "\n".join(L)
    open(os.path.join(HERE, "universe_report.md"), "w", encoding="utf-8").write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
