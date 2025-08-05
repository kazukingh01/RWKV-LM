import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt
import argparse

# local file
import env


def skewt_logpdf(x, xi, omega, lam, nu):
    z  = (x - xi) / omega
    t1 = stats.t.pdf(z, df=nu)
    t2 = stats.t.cdf(lam * z * np.sqrt((nu + 1)/(nu + z**2)), df=nu + 1)
    return np.log(2/omega) + np.log(t1) + np.log(t2)

def nll(params, data):
    xi, log_omega, lam, log_nu = params
    omega = np.exp(log_omega)
    nu    = np.exp(log_nu)          # ν>0
    ll = skewt_logpdf(data, xi, omega, lam, nu)
    return -np.sum(ll)

def fit_skewt(data, x0=None):
    if x0 is None:                # ざっくり初期値
        x0 = [np.mean(data), np.log(np.std(data)), 0.0, np.log(10.0)]
    res = optimize.minimize(
        nll, x0, args=(data,),
        method="L-BFGS-B"
    )
    if not res.success:
        raise RuntimeError(res.message)
    xi, log_omega, lam, log_nu = res.x
    omega = np.exp(log_omega)
    nu    = np.exp(log_nu)
    return {
        "xi": xi,
        "omega": omega,
        "lambda": lam,
        "nu": nu,
        "fun": res.fun,
        "hess_inv": res.hess_inv.todense()   # 逆ヘッセ近似
    }

def skewt_pdf(x, xi, omega, lam, nu):
    z  = (x - xi) / omega
    t1 = stats.t.pdf(z, df=nu)
    t2 = stats.t.cdf(lam * z * np.sqrt((nu + 1)/(nu + z**2)), df=nu + 1)
    return (2/omega) * t1 * t2

def plot_hist_with_fit(data, params, symbol: str, bins="auto"):
    xi, omega, lam, nu = params["xi"], params["omega"], params["lambda"], params["nu"]
    # ヒストグラム（密度=1 に正規化）
    plt.figure()
    plt.hist(data, bins=bins, density=True, alpha=0.4, edgecolor="black")
    x_min, x_max = np.min(data), np.max(data)
    x = np.linspace(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min), 800)
    pdf = skewt_pdf(x, xi, omega, lam, nu)
    plt.plot(x, pdf, linewidth=2, label="Skew-t PDF (fitted)")
    plt.title("Histogram (density) with fitted Skewed Student-t PDF")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"hist_with_fit_{symbol}.png")

def connect_to_mongodb():
    """Connect to MongoDB using credentials from env.py"""
    if hasattr(env, 'MONGO_USERNAME') and hasattr(env, 'MONGO_PASSWORD'):
        connection_string = f"mongodb://{env.MONGO_USERNAME}:{env.MONGO_PASSWORD}@{env.MONGO_HOST}:{env.MONGO_PORT}/"
    else:
        connection_string = f"mongodb://{env.MONGO_HOST}:{env.MONGO_PORT}/"
    
    client = MongoClient(connection_string)
    db = client[env.MONGO_DATABASE]
    
    return db

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='MongoDBからデータを取得してDataFrameに変換')
    parser.add_argument('--fr', type=str, required=True, help='開始日 (YYYYMMDD形式)')
    parser.add_argument('--to', type=str, required=True, help='終了日 (YYYYMMDD形式)')
    args = parser.parse_args()
    
    # YYYYMMDD形式の文字列をdatetimeオブジェクトに変換
    start_date = datetime.datetime.strptime(args.fr, "%Y%m%d").replace(tzinfo=datetime.timezone.utc)
    end_date   = datetime.datetime.strptime(args.to, "%Y%m%d").replace(tzinfo=datetime.timezone.utc)
    list_symbols = [119,122,123,124,125,126,127,128,129,130,131,132,133,134,135]
    
    db = connect_to_mongodb()
    print(db)
    
    # Get candles
    collection = db.get_collection("candles_10s")
    print(collection)
    query = {
        "unixtime": {
            "$gte": start_date,
            "$lte": end_date
        },
        "metadata.ym": {
            "$gte": int(start_date.strftime("%Y%m")),
            "$lte": int(end_date.strftime("%Y%m"))
        },
        "metadata.symbol": {
            "$in": list_symbols
        }
    }
    # 必要なフィールドのみを取得
    projection = {
        "unixtime": 1,
        "metadata.symbol": 1,
        "ask_price": 1,
        "bid_price": 1,
        "ask_volume": 1,
        "bid_volume": 1
    }
    cursor    = collection.find(query, projection)
    documents = list(tqdm(cursor, desc="Loading candles data"))
    df_candle = pd.DataFrame(documents)
    df_candle["symbol_id"] = df_candle["metadata"].str["symbol"]
    df_candle["price"]     = df_candle[["ask_price", "bid_price"]].mean(axis=1) 
    df_candle["timestamp"] = df_candle["unixtime"].astype(int) // 1000000000
    # datetime.datetime.fromtimestamp(df["timestamp"].min(), tz=datetime.UTC)

    # Get correlations
    collection = db.get_collection("corr_10s_w1h")
    print(collection)
    query = {
        "unixtime": {
            "$gte": start_date,
            "$lte": end_date
        },
        "ym": {
            "$gte": int(start_date.strftime("%Y%m")),
            "$lte": int(end_date.strftime("%Y%m"))
        },
    }
    cols_corr  = []
    cols_corr += [f"p_{min(x,y)}_{max(x,y)}" for i, x in enumerate(list_symbols) for y in list_symbols[i+1:]]
    cols_corr += [f"s_{min(x,y)}_{max(x,y)}" for i, x in enumerate(list_symbols) for y in list_symbols[i+1:]]
    projection = {col: 1 for col in cols_corr} | {"unixtime": 1}
    cursor     = collection.find(query, projection)
    documents  = list(tqdm(cursor, desc="Loading correlations data"))
    df_corr    = pd.DataFrame(documents)
    df_corr["timestamp"] = df_corr["unixtime"].astype(int) // 1000000000
    df_corr    = df_corr.loc[:, ["timestamp"] + cols_corr]
    df_corr.columns = df_corr.columns.str.replace("^p_", "cp_", regex=True).str.replace("^s_", "cs_", regex=True)
    cols_corr  = df_corr.columns.tolist()[1:]

    # create competed timeaxis dataframe
    ## base dataframe
    base_axis = []
    current_timestamp = start_date
    while current_timestamp <= end_date:
        base_axis.append(current_timestamp)
        current_timestamp += datetime.timedelta(seconds=10)
    df_base = pd.DataFrame(base_axis, columns=["timestamp"])
    df_base["timestamp"] = df_base["timestamp"].astype(int) // 1000000000
    ## join candles
    list_symbol = df_candle["symbol_id"].unique().tolist()
    for sid in list_symbol:
        dfwk = df_candle.loc[df_candle["symbol_id"] == sid, ["timestamp", "price", "ask_volume", "bid_volume"]].copy()
        dfwk.columns = ["timestamp", f"p_{sid}", f"vask_{sid}", f"vbid_{sid}"]
        df_base = pd.merge(df_base, dfwk, on="timestamp", how="left")
    cols = [f"p_{x}" for x in list_symbol]
    df_base[cols] = df_base[cols].ffill()
    df_base[cols] = df_base[cols].bfill()
    cols = [f"vask_{x}" for x in list_symbol] + [f"vbid_{x}" for x in list_symbol]
    df_base[cols] = df_base[cols].fillna(0)
    ## join correlations
    df_base = pd.merge(df_base, df_corr, on="timestamp", how="left")
    df_base[cols_corr] = df_base[cols_corr].ffill()
    df_base[cols_corr] = df_base[cols_corr].bfill()

    # GT
    for symbol in list_symbol:
        dfwk = np.log((df_base[f"p_{symbol}"].shift([-1,-2,-3,-4,-5,-6]) / df_base[f"p_{symbol}"].to_numpy().reshape(-1, 1)))
        df_base = pd.concat([df_base, dfwk.max(axis=1).rename(f"gt_max_{symbol}")], axis=1)
        df_base = pd.concat([df_base, dfwk.min(axis=1).rename(f"gt_min_{symbol}")], axis=1)
        df_base = pd.concat([df_base, dfwk[f"p_{symbol}_-6"].copy().rename(f"gt_1m_{symbol}")], axis=1)

    # GT Analysis
    for symbol in list_symbol:
        ndf  = df_base[[f"gt_min_{symbol}", f"gt_max_{symbol}"]].to_numpy().reshape(-1)
        ndf  = ndf[~np.isnan(ndf)]
        print(f"##### symbol: {symbol} #####")
        print(np.quantile(ndf, np.arange(0, 1, 0.01)))
        fit = fit_skewt(ndf)
        plot_hist_with_fit(ndf, fit, symbol, bins=500)
        bins = np.quantile(ndf, [0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
        bins[0]  = float("-inf")
        bins[-1] = float("inf")
        df_base = pd.concat([df_base, pd.cut(df_base[f"gt_min_{symbol}"], bins=bins, labels=[0,1,2,3,4], right=False).rename(f"gt_min_label_{symbol}")], axis=1)
        df_base = pd.concat([df_base, pd.cut(df_base[f"gt_max_{symbol}"], bins=bins, labels=[0,1,2,3,4], right=False).rename(f"gt_max_label_{symbol}")], axis=1)

    # to csv
    boolwk = ~(np.max(np.isnan(df_base[[f"gt_{y}_{x}" for x in list_symbol for y in ["min", "max"]]].to_numpy()), axis=-1))
    df_base.loc[boolwk, :].to_csv("data.csv", index=False)
