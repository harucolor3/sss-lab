"""
紙上売買の開始前（2026-09-28 より前）に1回だけ実行する準備スクリプト

やること:
  1. 対象銘柄を増やしたデータ（既定96銘柄＋追加候補。直近60日の平均売買代金10億円以上）を5年分取得 → sss_data_ext.json
  2. その時点で残った銘柄を universe.json に固定（以後、毎日の取得はこの一覧だけ。日々入れ替えない）
  3. トラックB（全体で学習・マイナスの配点も可）の配点を、拡大後の全銘柄・全期間（最後の保有期間分を除く）で
     リッジ回帰により学習し、paper_config_b.json に固定
  4. 確認用に、直近60営業日を両トラックの設定で運用した結果を表示（動作確認のみ。成績の証拠ではない）

使い方（Claude Code の stock-data 環境で。土日でも可。平日なら15:45以降）:
    pip install -r requirements.txt --break-system-packages -q
    python sss_setup_tracks.py
    python sss_setup_tracks.py --no-fetch     # 取得済みの sss_data_ext.json を使う
その後、universe.json・paper_config.json・paper_config_b.json を main にコミットする。

一度固定したら、検証中は再実行しないこと（再実行すると対象銘柄とトラックBの配点が変わる）。
"""
import argparse, copy, datetime as dt, json, os, subprocess, sys
import numpy as np
import pandas as pd
import sss_paper as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "sss_data_ext.json")
FACTORS = ["トレンド", "対市場", "出来高", "5日騰落", "安定性", "押し目"]


def drop_intraday_last_day(data):
    gen = data.get("generated", "")
    if gen[:10] == data["dates"][-1] and gen[11:16] < "15:45":
        n = len(data["dates"]) - 1
        data["dates"] = data["dates"][:n]
        data["benchmark"] = {k: (v[:n] if isinstance(v, list) else v) for k, v in data["benchmark"].items()}
        for s in data["stocks"]:
            for k in "ohlcv":
                s[k] = s[k][:n]
        print("最終日は取引時間中のデータのため除外しました")
    return data


def ridge_signed(R, ret, t_end):
    """全体で1つの配点（マイナス可）。ツール sss-lab.html の学習と同じ方法：y=損益（±15%で上限）、リッジ 1e-3×件数、
    係数を絶対値の合計100に換算"""
    m = (R.t.values < t_end) & ~np.isnan(ret)
    X = np.c_[np.ones(m.sum()), R.loc[m, [f"f{i}" for i in range(6)]].values]
    y = np.clip(ret[m], -0.15, 0.15)
    N = len(y)
    A = X.T @ X + np.diag([0] + [1e-3 * N] * 6)
    b = np.linalg.solve(A, X.T @ y)
    w = b[1:] / np.abs(b[1:]).sum() * 100
    return [round(float(x), 1) for x in w], [float(x) for x in b], N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch or not os.path.exists(DATA):
        subprocess.run([sys.executable, os.path.join(HERE, "sss_collect.py"), "--years", "5", "--universe", "extended",
                        "--out", DATA], check=True)
    data = drop_intraday_last_day(json.load(open(DATA, encoding="utf-8")))
    dates = data["dates"]; T = len(dates)
    codes = [s["code"] for s in data["stocks"]]
    added = [s for s in data["stocks"] if s.get("added")]

    # 2. 対象銘柄を固定
    uni = {"frozen_at": dt.date.today().isoformat(), "data_until": dates[-1], "count": len(codes),
           "rule": "既定96銘柄＋追加候補のうち、固定日時点の直近60営業日の平均売買代金10億円以上",
           "codes": codes}
    json.dump(uni, open(os.path.join(HERE, "universe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    limit = 5000 / 0.026 / 100
    affordable = sum(1 for s in data["stocks"] if s["c"][-1] and s["c"][-1] <= limit)
    print(f"対象銘柄: {len(codes)}（既定 {len(codes)-len(added)}＋追加 {len(added)}）。うち100株買える（{limit:,.0f}円以下）: {affordable}")

    # 3. トラックBの配点を学習して固定
    cfgA = sp.load_config("paper_config.json")
    H = cfgA["rules"]["max_hold_days"]
    R = sp.features(data)
    R = R[R.t >= 60].reset_index(drop=True)
    ret = sp.simulate_ret(data, R, np.arange(len(R)), cfgA)
    t_end = T - H - 1
    w, b, N = ridge_signed(R, ret, t_end)
    print(f"トラックBの配点（{dates[60]}〜{dates[t_end-1]} の {N:,}件で学習）: " + " / ".join(f"{n} {x:+.1f}" for n, x in zip(FACTORS, w)))

    cfgA["universe_file"] = "universe.json"
    cfgA["track_name"] = "押し目型（トラックA）"; cfgA["records_dir"] = "records"
    json.dump(cfgA, open(os.path.join(HERE, "paper_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    cfgB = copy.deepcopy(cfgA)
    cfgB.update({
        "_note": (f"トラックB。全体で1つの配点（マイナス可）。拡大後の{len(codes)}銘柄の {dates[60]}〜{dates[t_end-1]} のシグナル"
                  f" {N:,}件でリッジ回帰により学習し、{dt.date.today().isoformat()} に固定。検証中は変更しないこと"),
        "track_name": "全体学習（トラックB）", "records_dir": "records_b",
        "weights": w, "ridge_coef": b, "sector_weights": None})
    cfgB.pop("sector_weights", None)
    json.dump(cfgB, open(os.path.join(HERE, "paper_config_b.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    sp.load_config("paper_config.json")  # 記録先を元に戻す

    # 4. 動作確認：直近60営業日（決算回避なし）
    groups = {t: g for t, g in R.groupby("t")}
    B = data["benchmark"]["c"]
    print("\n動作確認（直近60営業日・決算回避なし。トラックBの配点はこの期間も含めて学習しているので成績の証拠ではない）")
    for name, c in [("トラックA 押し目型", cfgA), ("トラックB 全体学習", cfgB)]:
        st = sp.empty_state(c); last = None; n = 0
        for t in range(T - 60, T):
            last = sp.step(data, groups[t], t, st, c, earn=None); n += len(last["closed"])
        print(f"  {name}: {sp.yen(st['realized'] + last['unreal'])}（{n}回）")
    print(f"  TOPIX買い持ち60万円: {sp.yen(600000 * (B[-1] / B[T - 61] - 1))}")
    print("\n次に universe.json・paper_config.json・paper_config_b.json を main にコミットしてください。")


if __name__ == "__main__":
    main()
