from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: List[str] = [
    "ret_1",
    "ret_5",
    "ret_20",
    "close_sma5",
    "close_sma20",
    "close_sma60",
    "sma20_sma60",
    "rsi14",
    "macd_pct",
    "atr14_pct",
    "volatility20",
    "volume_ratio20",
    "dist_high20",
    "dist_low20",
    "bb_position",
    "range20_pct",
]


@dataclass
class ForecastResult:
    horizon: int
    probability_up: float
    cv_accuracy: float
    cv_balanced_accuracy: float
    cv_auc: float | None
    similar_up_rate: float
    expected_return_median: float
    return_low: float
    return_high: float
    price_median: float
    price_low: float
    price_high: float
    analog_count: int


def _period_start_date(period: str):
    from datetime import datetime, timedelta, timezone

    years = {"2y": 2, "5y": 5, "10y": 10}.get(period, 5)
    # Add a small cushion for weekends/holidays and indicator warm-up.
    return (datetime.now(timezone.utc) - timedelta(days=int(years * 365.25) + 14)).date()


def _normalize_ohlcv(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("빈 가격 데이터")

    data = data.copy()
    data = data.rename(columns={c: str(c).title() for c in data.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"가격 데이터 열이 부족합니다: {missing}")

    data = data[required].copy()
    for c in required:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~data.index.isna()]
    # Drop timezone so Plotly / sklearn handling is consistent on Windows.
    try:
        data.index = data.index.tz_localize(None)
    except TypeError:
        pass
    data = data[~data.index.duplicated(keep="last")].sort_index()

    if len(data) < 320:
        raise ValueError(
            f"학습에 사용할 데이터가 {len(data)}거래일뿐입니다. "
            "가능하면 최소 320거래일 이상 데이터가 있는 종목을 사용하세요."
        )
    return data


def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        multi_level_index=False,
        threads=False,
        timeout=30,
    )
    data = _normalize_ohlcv(data, ticker)
    data.attrs["source"] = "Yahoo Finance (yfinance)"
    return data


def _fetch_yahoo_chart_api(ticker: str, period: str) -> pd.DataFrame:
    """Direct Yahoo chart endpoint fallback that does not need yfinance cookies."""
    import json
    import urllib.parse
    import urllib.request
    from datetime import datetime, timezone

    start = _period_start_date(period)
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.now(timezone.utc).timestamp()) + 86400
    quoted = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quoted}"
        f"?period1={period1}&period2={period2}&interval=1d"
        "&events=div%2Csplits&includeAdjustedClose=true"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise ValueError(f"Yahoo 직접 연결 결과 없음: {error}")

    r = result[0]
    ts = r.get("timestamp") or []
    quote = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    adj_block = ((r.get("indicators") or {}).get("adjclose") or [{}])[0]
    if not ts or not quote:
        raise ValueError("Yahoo 직접 연결에 일봉 데이터가 없습니다.")

    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None)
    frame = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=idx,
    )

    # Make OHLC approximately split/dividend adjusted when adjusted close exists.
    adj = pd.Series(adj_block.get("adjclose"), index=idx, dtype="float64")
    raw_close = pd.to_numeric(frame["Close"], errors="coerce")
    ratio = adj / raw_close.replace(0, np.nan)
    if ratio.notna().any():
        for c in ["Open", "High", "Low", "Close"]:
            frame[c] = pd.to_numeric(frame[c], errors="coerce") * ratio

    frame = _normalize_ohlcv(frame, ticker)
    frame.attrs["source"] = "Yahoo Finance (직접 연결)"
    return frame


def _fetch_stooq_us(ticker: str, period: str) -> pd.DataFrame:
    """Fallback for US tickers using Stooq daily CSV."""
    import io
    import urllib.parse
    import urllib.request
    from datetime import datetime, timezone

    # Stooq's common US symbol convention is SYMBOL.US.
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        raise ValueError("Stooq 미국주식 대체 소스는 한국 종목에 사용하지 않습니다.")

    symbol = ticker.lower()
    if symbol.endswith(".us"):
        symbol = symbol[:-3]
    stooq_symbol = f"{symbol}.us"
    start = _period_start_date(period).strftime("%Y%m%d")
    end = datetime.now(timezone.utc).date().strftime("%Y%m%d")
    q = urllib.parse.urlencode({"s": stooq_symbol, "d1": start, "d2": end, "i": "d"})
    url = f"https://stooq.com/q/d/l/?{q}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    if not text.strip() or "No data" in text:
        raise ValueError("Stooq에서 해당 종목 데이터를 찾지 못했습니다.")

    frame = pd.read_csv(io.StringIO(text))
    if "Date" not in frame.columns:
        raise ValueError("Stooq 응답 형식이 예상과 다릅니다.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.set_index("Date")
    frame = _normalize_ohlcv(frame, ticker)
    frame.attrs["source"] = "Stooq (미국주식 대체 소스)"
    return frame


def fetch_price_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download daily OHLCV with automatic fallbacks.

    Order:
      1) yfinance
      2) Yahoo chart API directly
      3) Stooq CSV for US stocks
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("종목코드를 입력하세요.")

    errors = []
    for name, func in [
        ("Yahoo/yfinance", _fetch_yfinance),
        ("Yahoo 직접 연결", _fetch_yahoo_chart_api),
        ("Stooq", _fetch_stooq_us),
    ]:
        try:
            return func(ticker, period)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__} - {exc}")

    detail = " | ".join(errors)
    raise ValueError(
        f"{ticker} 가격 데이터를 3개 경로에서 모두 불러오지 못했습니다. "
        "종목코드를 확인하고, VPN/프록시를 사용 중이면 잠시 끈 뒤 다시 시도하거나 "
        "다른 인터넷(예: 휴대폰 핫스팟)에서 실행해 보세요. "
        f"연결 상세: {detail}"
    )


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    close = df["Close"]

    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)

    for window in (5, 20, 60):
        df[f"sma{window}"] = close.rolling(window).mean()
        df[f"close_sma{window}"] = close / df[f"sma{window}"] - 1

    df["sma20_sma60"] = df["sma20"] / df["sma60"] - 1
    df["rsi14"] = _rsi(close, 14) / 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_pct"] = macd / close

    atr14 = _atr(df, 14)
    df["atr14"] = atr14
    df["atr14_pct"] = atr14 / close

    df["volatility20"] = df["ret_1"].rolling(20).std()
    vol_mean20 = df["Volume"].rolling(20).mean()
    df["volume_ratio20"] = df["Volume"] / vol_mean20.replace(0, np.nan)

    high20 = df["High"].rolling(20).max()
    low20 = df["Low"].rolling(20).min()
    df["dist_high20"] = close / high20 - 1
    df["dist_low20"] = close / low20 - 1
    df["range20_pct"] = high20 / low20 - 1

    std20 = close.rolling(20).std()
    upper = df["sma20"] + 2 * std20
    lower = df["sma20"] - 2 * std20
    band_width = (upper - lower).replace(0, np.nan)
    df["bb_position"] = (close - lower) / band_width

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _safe_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return None


def _time_series_cv_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    random_state: int = 42,
) -> Tuple[float, float, float | None]:
    # The gap separates training observations from each test block so that
    # overlapping forward-return labels are less likely to leak information.
    n_splits = 5 if len(X) >= 700 else 4
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

    accuracies: List[float] = []
    balanced: List[float] = []
    aucs: List[float] = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        accuracies.append(float(accuracy_score(y_test, pred)))
        balanced.append(float(balanced_accuracy_score(y_test, pred)))
        auc = _safe_auc(y_test, prob)
        if auc is not None:
            aucs.append(auc)

    if not accuracies:
        return float("nan"), float("nan"), None

    return (
        float(np.mean(accuracies)),
        float(np.mean(balanced)),
        float(np.mean(aucs)) if aucs else None,
    )


def _similar_pattern_stats(
    X_train: pd.DataFrame,
    forward_returns: pd.Series,
    latest_features: pd.DataFrame,
    k: int = 35,
) -> Tuple[float, float, float, float, int]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    latest_scaled = scaler.transform(latest_features)[0]

    distances = np.sqrt(((train_scaled - latest_scaled) ** 2).mean(axis=1))
    k = int(min(max(10, k), len(X_train)))
    nearest_positions = np.argpartition(distances, k - 1)[:k]

    analog_returns = forward_returns.iloc[nearest_positions].dropna()
    if analog_returns.empty:
        return 0.5, 0.0, 0.0, 0.0, 0

    return (
        float((analog_returns > 0).mean()),
        float(analog_returns.median()),
        float(analog_returns.quantile(0.20)),
        float(analog_returns.quantile(0.80)),
        int(len(analog_returns)),
    )


def forecast_horizon(
    feature_df: pd.DataFrame,
    horizon: int,
    random_state: int = 42,
) -> ForecastResult:
    if horizon < 1:
        raise ValueError("예측 기간은 1거래일 이상이어야 합니다.")

    work = feature_df.copy()
    work["forward_return"] = work["Close"].shift(-horizon) / work["Close"] - 1
    work["target"] = (work["forward_return"] > 0).astype(int)

    latest = work.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    train = work.dropna(subset=FEATURE_COLUMNS + ["forward_return"]).copy()

    if len(train) < 250:
        raise ValueError(f"{horizon}일 모델 학습 데이터가 부족합니다.")

    X = train[FEATURE_COLUMNS]
    y = train["target"].astype(int)
    latest_X = latest[FEATURE_COLUMNS]

    cv_acc, cv_bal_acc, cv_auc = _time_series_cv_metrics(X, y, horizon, random_state)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    model_prob = float(model.predict_proba(latest_X)[0, 1])

    analog_up, ret_median, ret_low, ret_high, analog_count = _similar_pattern_stats(
        X,
        train["forward_return"],
        latest_X,
        k=35,
    )

    # Blend the supervised model and nearest historical analogs. This is a
    # heuristic ensemble, not a guaranteed probability calibration.
    blended_prob = 0.70 * model_prob + 0.30 * analog_up

    last_price = float(latest["Close"].iloc[0])
    return ForecastResult(
        horizon=horizon,
        probability_up=blended_prob,
        cv_accuracy=cv_acc,
        cv_balanced_accuracy=cv_bal_acc,
        cv_auc=cv_auc,
        similar_up_rate=analog_up,
        expected_return_median=ret_median,
        return_low=ret_low,
        return_high=ret_high,
        price_median=last_price * (1 + ret_median),
        price_low=last_price * (1 + ret_low),
        price_high=last_price * (1 + ret_high),
        analog_count=analog_count,
    )


def analyze_ticker(data: pd.DataFrame, horizons: Iterable[int] = (1, 5, 20)) -> Dict:
    features = build_features(data)
    clean = features.dropna(subset=FEATURE_COLUMNS)
    if clean.empty:
        raise ValueError("기술지표를 계산할 수 있는 데이터가 부족합니다.")

    latest = clean.iloc[-1]
    close = float(latest["Close"])
    sma20 = float(latest["sma20"])
    sma60 = float(latest["sma60"])
    rsi = float(latest["rsi14"] * 100)

    if close > sma20 > sma60:
        trend = "상승 추세"
    elif close < sma20 < sma60:
        trend = "하락 추세"
    else:
        trend = "혼조 / 박스권 가능"

    recent60 = features.tail(60)
    support = float(recent60["Low"].quantile(0.10))
    resistance = float(recent60["High"].quantile(0.90))
    range60 = float(recent60["High"].max() / recent60["Low"].min() - 1)
    ma_gap = abs(sma20 / sma60 - 1)
    box_candidate = bool(range60 <= 0.22 and ma_gap <= 0.06)

    # Detect a rough "drop then sideways" setup: a meaningful decline in the
    # preceding 60 trading days followed by a relatively compressed last 60 days.
    drop_then_box = False
    if len(features) >= 125:
        prev = features.iloc[-120:-60]
        if len(prev) >= 40:
            prev_return = float(prev["Close"].iloc[-1] / prev["Close"].iloc[0] - 1)
            drop_then_box = bool(prev_return <= -0.10 and box_candidate)

    forecasts = [forecast_horizon(features, int(h)) for h in horizons]

    return {
        "features": features,
        "latest_date": clean.index[-1],
        "latest_price": close,
        "trend": trend,
        "rsi14": rsi,
        "support": support,
        "resistance": resistance,
        "range60": range60,
        "box_candidate": box_candidate,
        "drop_then_box": drop_then_box,
        "forecasts": forecasts,
    }
