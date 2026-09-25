"""
SSS（Short Swing Score）用データ取得スクリプト

使い方（自分のPCで実行）:
    pip install yfinance
    python sss_collect.py            # 既定の約100銘柄（11セクター）・5年分
    python sss_collect.py --years 3  # 期間を変える
    python sss_collect.py --codes 7203 6758 8306   # 銘柄を指定
    python sss_collect.py --earnings # 過去の決算発表日も取得（遅い。ツールの決算回避に使用）

出力: sss_data.json → ツールの「データを読み込む」から選択してください。

注意:
- yfinance は Yahoo Finance の非公式ライブラリです。個人の研究用途にとどめてください。
- 株価は分割調整済み・配当は未調整（権利落ちの下落はそのまま残ります）。
- 分割の履歴を "splits" に保存します。ツールはこれを使って「当時の実際の株価」で100株の購入金額を計算します。
- 本格運用するときは JPX 公式の J-Quants API への置き換えを推奨します。
"""
import argparse
import datetime as dt
import json
import math
import sys

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance が必要です:  pip install yfinance")

# 売買が活発な東証プライムの大型株（監視10銘柄を含む）。値は (銘柄名, セクター)
# セクターは学習用に独自にまとめた11分類（東証33業種を大まかに束ねたもの）
SECTOR_MAP = {
    "自動車": {"7203": "トヨタ自動車", "7267": "ホンダ", "7201": "日産自動車", "7269": "スズキ",
             "7270": "SUBARU", "7272": "ヤマハ発動機", "6902": "デンソー", "7261": "マツダ"},
    "銀行": {"8306": "三菱UFJ", "8316": "三井住友FG", "8411": "みずほFG", "8308": "りそなHD", "7182": "ゆうちょ銀行"},
    "証券・保険": {"8604": "野村HD", "8601": "大和証券G", "8766": "東京海上HD", "8725": "MS&AD",
               "8630": "SOMPO", "8591": "オリックス", "8750": "第一生命HD"},
    "半導体": {"8035": "東京エレクトロン", "6857": "アドバンテスト", "6526": "ソシオネクスト", "6723": "ルネサス",
            "3436": "SUMCO", "6146": "ディスコ", "285A": "キオクシアHD", "7735": "SCREEN",
            "6920": "レーザーテック", "4062": "イビデン", "6963": "ローム"},
    "電機・電子部品": {"6758": "ソニーG", "6501": "日立", "6752": "パナソニックHD", "6981": "村田製作所",
                "6762": "TDK", "6971": "京セラ", "6503": "三菱電機", "6701": "NEC", "6702": "富士通",
                "7751": "キヤノン", "5803": "フジクラ", "5802": "住友電工"},
    "機械・重工": {"7011": "三菱重工", "7012": "川崎重工", "7013": "IHI", "6954": "ファナック",
               "6301": "コマツ", "6367": "ダイキン", "6273": "SMC", "6326": "クボタ"},
    "商社": {"8058": "三菱商事", "8031": "三井物産", "8001": "伊藤忠", "8053": "住友商事", "8002": "丸紅"},
    "資源・素材": {"5020": "ENEOS", "1605": "INPEX", "5401": "日本製鉄", "5411": "JFE HD", "4063": "信越化学",
               "4005": "住友化学", "5713": "住友金属鉱山", "3407": "旭化成"},
    "通信・IT": {"9432": "NTT", "9433": "KDDI", "9434": "ソフトバンク", "9984": "ソフトバンクG",
              "4755": "楽天G", "6098": "リクルートHD", "4689": "LINEヤフー", "4307": "野村総研"},
    "医薬・消費": {"4502": "武田薬品", "4568": "第一三共", "4519": "中外製薬", "4503": "アステラス",
               "2914": "JT", "2502": "アサヒGHD", "4452": "花王", "3382": "セブン&アイ",
               "9983": "ファーストリテイリング", "7974": "任天堂", "8267": "イオン", "4661": "オリエンタルランド"},
    "運輸・不動産": {"9101": "日本郵船", "9104": "商船三井", "9107": "川崎汽船", "9020": "JR東日本",
                "9022": "JR東海", "9201": "JAL", "9202": "ANA HD", "8801": "三井不動産",
                "8802": "三菱地所", "1925": "大和ハウス", "1812": "鹿島建設", "1801": "大成建設"},
}
DEFAULT_STOCKS = {c: (n, sec) for sec, d in SECTOR_MAP.items() for c, n in d.items()}
BENCH = ("1306", "TOPIX連動ETF（TOPIXの代わり）")


def clean(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--codes", nargs="*", help="銘柄コード（省略時は既定リスト）")
    ap.add_argument("--out", default="sss_data.json")
    ap.add_argument("--earnings", action="store_true", help="過去の決算発表日も取得する")
    args = ap.parse_args()

    stocks = {c: DEFAULT_STOCKS.get(c, (c, "その他")) for c in args.codes} if args.codes else DEFAULT_STOCKS
    tickers = [f"{c}.T" for c in stocks] + [f"{BENCH[0]}.T"]
    print(f"{len(tickers)} 銘柄を取得中…（{args.years}年分）")
    df = yf.download(tickers, period=f"{args.years}y", interval="1d",
                     auto_adjust=False, actions=True, group_by="ticker", progress=True, threads=True)

    bench = df[f"{BENCH[0]}.T"].dropna(subset=["Close"])
    dates = list(bench.index)
    out = {
        "generated": dt.datetime.now().isoformat(timespec="minutes"),
        "source": "yfinance",
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "benchmark": {"code": BENCH[0], "name": BENCH[1],
                      "o": [clean(x, 2) for x in bench["Open"]],
                      "c": [clean(x, 2) for x in bench["Close"]]},
        "stocks": [],
    }
    for code, (name, sector) in stocks.items():
        t = f"{code}.T"
        if t not in df.columns.get_level_values(0):
            print(f"  取得失敗: {code} {name}")
            continue
        s = df[t].reindex(dates)
        if s["Close"].notna().sum() < 120:
            print(f"  データ不足のため除外: {code} {name}")
            continue
        splits = []
        if "Stock Splits" in s.columns:
            for d, r in s["Stock Splits"].items():
                if r is not None and not math.isnan(r) and r not in (0, 1):
                    splits.append([d.strftime("%Y-%m-%d"), float(r)])
        earnings = []
        if args.earnings:
            try:
                ed = yf.Ticker(t).get_earnings_dates(limit=40)
                if ed is not None:
                    earnings = sorted({d.strftime("%Y-%m-%d") for d in ed.index})
            except Exception as e:  # 取得できない銘柄もある
                print(f"  決算日の取得失敗: {code} {name}（{e.__class__.__name__}）")
        out["stocks"].append({
            "code": code, "name": name, "sector": sector, "splits": splits,
            **({"earnings": earnings} if args.earnings else {}),
            "o": [clean(x) for x in s["Open"]], "h": [clean(x) for x in s["High"]],
            "l": [clean(x) for x in s["Low"]], "c": [clean(x) for x in s["Close"]],
            "v": [None if (v is None or math.isnan(v)) else int(v) for v in s["Volume"]],
        })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"完了: {args.out}（{len(out['stocks'])}銘柄 × {len(dates)}営業日）")


if __name__ == "__main__":
    main()
