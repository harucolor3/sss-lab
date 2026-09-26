"""
SSS（Short Swing Score）用データ取得スクリプト

使い方（自分のPCで実行）:
    pip install yfinance
    python sss_collect.py            # 既定の約100銘柄（11セクター）・5年分
    python sss_collect.py --years 3  # 期間を変える
    python sss_collect.py --codes 7203 6758 8306   # 銘柄を指定
    python sss_collect.py --earnings # 過去の決算発表日も取得（遅い。ツールの決算回避に使用）
    python sss_collect.py --start 2011-01-01 --end 2021-09-25 --full-splits --out sss_data_pre.json
                                     # 期間を日付で指定（学習前の期間の検証用）。--full-splits は期間後の分割も含めた全履歴

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

# 拡大用の追加候補（売買が活発なプライムの大型・中型株）。株価の安さでは選ばない（後知恵の偏りを避けるため）。
# 取得後に「直近60日の平均売買代金10億円以上」で絞り込む。コードの誤り・上場廃止は取得失敗として自動で除外される。
EXTRA_SECTOR_MAP = {
    "銀行": {"8309": "三井住友トラスト", "8410": "セブン銀行", "7186": "コンコルディアFG", "8331": "千葉銀行",
            "7167": "めぶきFG", "8304": "あおぞら銀行", "5831": "しずおかFG", "8359": "八十二銀行", "8334": "群馬銀行"},
    "証券・保険": {"8473": "SBI HD", "8628": "松井証券", "8697": "日本取引所G", "8698": "マネックスG",
               "8795": "T&D HD", "7181": "かんぽ生命", "8593": "三菱HCキャピタル", "8570": "イオンFS", "6178": "日本郵政"},
    "自動車": {"7211": "三菱自動車", "7202": "いすゞ", "7259": "アイシン", "7282": "豊田合成", "5108": "ブリヂストン"},
    "運輸・不動産": {"9064": "ヤマトHD", "9005": "東急", "9007": "小田急", "9008": "京王", "9009": "京成",
                "9001": "東武", "9021": "JR西日本", "9142": "JR九州", "9024": "西武HD", "9147": "NIPPON EXPRESS",
                "8830": "住友不動産", "8804": "東京建物", "3289": "東急不動産HD", "3003": "ヒューリック",
                "1928": "積水ハウス", "1878": "大東建託", "1802": "大林組", "1803": "清水建設", "1963": "日揮HD"},
    "資源・素材": {"5019": "出光興産", "5021": "コスモエネルギー", "3402": "東レ", "3401": "帝人", "4183": "三井化学",
               "4188": "三菱ケミカルG", "4208": "UBE", "5711": "三菱マテリアル", "5706": "三井金属", "5714": "DOWA",
               "3861": "王子HD", "5201": "AGC", "5233": "太平洋セメント", "5332": "TOTO", "5801": "古河電工"},
    "電力・ガス": {"9501": "東京電力HD", "9502": "中部電力", "9503": "関西電力", "9504": "中国電力", "9506": "東北電力",
               "9508": "九州電力", "9513": "電源開発", "9531": "東京ガス", "9532": "大阪ガス"},
    "電機・電子部品": {"6724": "セイコーエプソン", "6753": "シャープ", "6770": "アルプスアルパイン", "6479": "ミネベアミツミ",
                "6448": "ブラザー", "7731": "ニコン", "7733": "オリンパス", "7752": "リコー", "6594": "ニデック",
                "6645": "オムロン", "5334": "日本特殊陶業"},
    "機械・重工": {"6472": "NTN", "6471": "日本精工", "6473": "ジェイテクト", "6305": "日立建機", "7003": "三井E&S",
               "6113": "アマダ", "6302": "住友重機", "6268": "ナブテスコ"},
    "商社": {"2768": "双日", "8015": "豊田通商", "8020": "兼松"},
    "通信・IT": {"3659": "ネクソン", "2432": "DeNA", "4385": "メルカリ", "3923": "ラクス", "4751": "サイバーエージェント",
              "9468": "KADOKAWA", "9449": "GMOインターネットG", "4324": "電通G"},
    "医薬・消費": {"2503": "キリンHD", "2269": "明治HD", "2282": "日本ハム", "2801": "キッコーマン", "4911": "資生堂",
               "4901": "富士フイルム", "7832": "バンダイナムコ", "7911": "TOPPAN", "7912": "大日本印刷",
               "3099": "三越伊勢丹", "8233": "高島屋", "8252": "丸井G", "3086": "Jフロント", "9602": "東宝",
               "4506": "住友ファーマ", "4507": "塩野義製薬", "4151": "協和キリン", "4523": "エーザイ",
               "4528": "小野薬品", "4543": "テルモ"},
}
EXTRA_STOCKS = {c: (n, sec) for sec, d in EXTRA_SECTOR_MAP.items() for c, n in d.items() if c not in DEFAULT_STOCKS}
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
    ap.add_argument("--start", help="開始日 YYYY-MM-DD（指定すると --years より優先）")
    ap.add_argument("--end", help="終了日 YYYY-MM-DD（この日は含まない）")
    ap.add_argument("--full-splits", action="store_true", help="期間外も含めた分割の全履歴を保存（過去期間の株数計算に必要）")
    ap.add_argument("--universe", choices=["default", "extended"], default="default",
                    help="extended = 既定96銘柄＋追加候補（直近60日の平均売買代金10億円以上だけ残す）")
    args = ap.parse_args()

    stocks = {c: DEFAULT_STOCKS.get(c, (c, "その他")) for c in args.codes} if args.codes else DEFAULT_STOCKS
    if args.universe == "extended" and not args.codes:
        stocks = {**DEFAULT_STOCKS, **EXTRA_STOCKS}
    tickers = [f"{c}.T" for c in stocks] + [f"{BENCH[0]}.T"]
    span = dict(start=args.start, end=args.end) if args.start else dict(period=f"{args.years}y")
    print(f"{len(tickers)} 銘柄を取得中…（{args.start + '〜' + (args.end or '') if args.start else str(args.years) + '年分'}）")
    df = yf.download(tickers, interval="1d", auto_adjust=False, actions=True,
                     group_by="ticker", progress=True, threads=True, **span)

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
        if code in EXTRA_STOCKS and args.universe == "extended":
            tv = (s["Close"] * s["Volume"]).tail(60).mean()
            if not (tv >= 1e9):
                print(f"  売買代金が少ないため除外: {code} {name}（直近60日平均 {tv/1e8:.1f}億円）")
                continue
        splits = []
        if "Stock Splits" in s.columns:
            for d, r in s["Stock Splits"].items():
                if r is not None and not math.isnan(r) and r not in (0, 1):
                    splits.append([d.strftime("%Y-%m-%d"), float(r)])
        if args.full_splits:
            try:
                sp = yf.Ticker(t).splits
                splits = [[d.strftime("%Y-%m-%d"), float(r)] for d, r in sp.items() if r not in (0, 1)]
            except Exception as e:
                print(f"  分割履歴の取得失敗: {code} {name}（{e.__class__.__name__}）")
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
            **({"added": True} if code in EXTRA_STOCKS and args.universe == "extended" else {}),
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
