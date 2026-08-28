from __future__ import annotations

import html
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_engine import analyze_ticker, fetch_price_data

st.set_page_config(
    page_title="주식 차트 확률 예측기 Mobile",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Mobile-first styling. It also looks clean on a desktop browser.
st.markdown(
    """
<style>
    .block-container {max-width: 900px; padding-top: 1.0rem; padding-bottom: 4rem;}
    [data-testid="stSidebar"] {display: none;}
    header[data-testid="stHeader"] {background: transparent;}
    h1 {font-size: clamp(1.7rem, 6vw, 2.35rem) !important; margin-bottom: .2rem;}
    h2, h3 {letter-spacing: -0.02em;}
    .small-muted {color: #6b7280; font-size: .88rem; line-height: 1.55;}
    .hero-note {padding: .8rem 1rem; border-radius: 14px; background: rgba(49,130,206,.08); margin: .65rem 0 1rem 0;}
    .status-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.65rem; margin:.6rem 0 1rem 0;}
    .status-card {border:1px solid rgba(128,128,128,.22); border-radius:16px; padding:.9rem; min-height:92px;}
    .status-label {font-size:.78rem; color:#6b7280; margin-bottom:.30rem;}
    .status-value {font-size:1.35rem; font-weight:750; line-height:1.15; word-break:keep-all;}
    .level-card {border:1px solid rgba(128,128,128,.22); border-radius:16px; padding:.85rem; text-align:center;}
    .level-name {font-size:.76rem; color:#6b7280;}
    .level-price {font-size:1.15rem; font-weight:750; margin-top:.2rem;}
    .forecast-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; margin:.5rem 0 .8rem 0;}
    .forecast-card {border:1px solid rgba(128,128,128,.22); border-radius:16px; padding:.85rem .55rem; text-align:center;}
    .forecast-days {font-size:.78rem; color:#6b7280;}
    .forecast-prob {font-size:1.42rem; font-weight:800; margin:.15rem 0;}
    .forecast-label {font-size:.76rem; font-weight:650;}
    .confidence {font-size:.70rem; color:#6b7280; margin-top:.28rem;}
    div[data-testid="stTextInput"] input {font-size:1.08rem; min-height:48px; border-radius:12px;}
    div[data-testid="stSelectbox"] > div > div {min-height:48px; border-radius:12px;}
    div.stButton > button {min-height:50px; border-radius:13px; font-size:1.05rem; font-weight:750; width:100%;}
    .stDataFrame {border-radius:14px; overflow:hidden;}
    @media (max-width: 520px) {
        .block-container {padding-left:.85rem; padding-right:.85rem; padding-top:.65rem;}
        .forecast-grid {grid-template-columns:repeat(3,minmax(0,1fr)); gap:.35rem;}
        .forecast-card {padding:.72rem .3rem;}
        .forecast-prob {font-size:1.20rem;}
        .status-value {font-size:1.18rem;}
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=1800, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    return fetch_price_data(ticker, period)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def is_korean_ticker(ticker: str) -> bool:
    t = ticker.strip().upper()
    return t.endswith(".KS") or t.endswith(".KQ")


def price_text(value: float, ticker: str) -> str:
    if is_korean_ticker(ticker):
        return f"₩{value:,.0f}"
    return f"${value:,.2f}"


def probability_label(p: float) -> str:
    if p >= 0.62:
        return "상승 우세"
    if p <= 0.38:
        return "하락 우세"
    return "방향성 약함"


def auc_label(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "검증 부족"
    if v >= 0.70:
        return "검증 강함"
    if v >= 0.60:
        return "검증 보통"
    if v >= 0.55:
        return "검증 약함"
    return "검증 매우 약함"


def card(label: str, value: str) -> str:
    return (
        '<div class="status-card">'
        f'<div class="status-label">{html.escape(label)}</div>'
        f'<div class="status-value">{html.escape(value)}</div>'
        '</div>'
    )


def make_chart(df: pd.DataFrame, support: float, resistance: float) -> go.Figure:
    view = df.tail(180).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], name="가격"
    ))
    fig.add_trace(go.Scatter(x=view.index, y=view["sma20"], mode="lines", name="SMA20", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=view.index, y=view["sma60"], mode="lines", name="SMA60", line=dict(width=1.7)))
    fig.add_hline(y=support, line_dash="dot", annotation_text="지지")
    fig.add_hline(y=resistance, line_dash="dot", annotation_text="저항")
    fig.update_layout(
        height=430,
        margin=dict(l=3, r=3, t=20, b=4),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


st.title("📱 주식 차트 확률 예측기")
st.markdown('<div class="small-muted">V5 Mobile · 휴대폰 화면 최적화</div>', unsafe_allow_html=True)

with st.form("analysis_form", clear_on_submit=False):
    ticker = st.text_input(
        "종목코드",
        value="PYPL",
        placeholder="예: PYPL, STZ, BX, 005930.KS",
        help="미국주식은 티커, 코스피는 .KS, 코스닥은 .KQ를 붙입니다.",
    ).strip().upper()
    period = st.selectbox("학습 데이터 기간", ["2y", "5y", "10y"], index=1)
    run = st.form_submit_button("🔎 분석 시작", type="primary", use_container_width=True)

st.markdown(
    '<div class="hero-note small-muted">차트·거래량·기술지표·과거 유사패턴·Random Forest를 이용한 참고용 확률입니다. 실적·뉴스·금리·돌발 이벤트를 완전히 반영하지 못하며 수익을 보장하지 않습니다.</div>',
    unsafe_allow_html=True,
)

if not run:
    st.markdown("### 사용 방법")
    st.markdown(
        """
1. 종목코드를 입력합니다. 미국주식은 `PYPL`, `STZ`, `BX`처럼 입력합니다.
2. 기본값 `5y`로 두고 **분석 시작**을 누릅니다.
3. **현재 추세 → 지지/저항 → 1·5·20일 상승확률 → AUC** 순서로 봅니다.
4. 확률 하나만 보지 말고 예상가격 범위와 과거 유사패턴도 같이 확인합니다.
        """
    )
    st.markdown("**한국주식 예시** · 삼성전자 `005930.KS` · 코스닥 `247540.KQ`")
    st.stop()

try:
    with st.spinner("가격 데이터 수집 → 기술지표 계산 → 과거 검증 → 예측 중..."):
        raw = load_data(ticker, period)
        source = raw.attrs.get("source", "알 수 없음")
        result = analyze_ticker(raw, horizons=(1, 5, 20))
except Exception as exc:
    st.error(f"분석 중 오류: {exc}")
    st.info("데이터 연결 문제면 VPN/프록시를 끄거나 다른 인터넷으로 다시 시도해 주세요. 미국주식은 Yahoo 실패 시 Stooq로 자동 전환합니다.")
    st.stop()

features = result["features"]
recent60 = features.tail(60)
range_low = float(recent60["Low"].min())
range_high = float(recent60["High"].max())
pattern = "하락 후 박스권 후보" if result["drop_then_box"] else ("박스권 후보" if result["box_candidate"] else "뚜렷한 박스권 아님")

st.caption(f"데이터: {source} · 기준일 {result['latest_date'].strftime('%Y-%m-%d')}")
st.subheader(f"{ticker} 현재 상태")
status_html = '<div class="status-grid">' + ''.join([
    card("최근 종가", price_text(result["latest_price"], ticker)),
    card("추세", result["trend"]),
    card("RSI(14)", f"{result['rsi14']:.1f}"),
    card("60일 가격 범위", pct(result["range60"])),
]) + '</div>'
st.markdown(status_html, unsafe_allow_html=True)

st.caption(
    f"최근 60거래일 최저~최고: {price_text(range_low, ticker)} ~ {price_text(range_high, ticker)} · 패턴: {pattern}"
)

level_cols = st.columns(3)
with level_cols[0]:
    st.markdown(f'<div class="level-card"><div class="level-name">지지 후보</div><div class="level-price">{price_text(result["support"], ticker)}</div></div>', unsafe_allow_html=True)
with level_cols[1]:
    st.markdown(f'<div class="level-card"><div class="level-name">현재가</div><div class="level-price">{price_text(result["latest_price"], ticker)}</div></div>', unsafe_allow_html=True)
with level_cols[2]:
    st.markdown(f'<div class="level-card"><div class="level-name">저항 후보</div><div class="level-price">{price_text(result["resistance"], ticker)}</div></div>', unsafe_allow_html=True)

st.markdown("### 상승확률 요약")
forecast_cards = []
for f in result["forecasts"]:
    auc_txt = "N/A" if f.cv_auc is None else f"{f.cv_auc:.3f}"
    forecast_cards.append(
        '<div class="forecast-card">'
        f'<div class="forecast-days">{f.horizon}거래일</div>'
        f'<div class="forecast-prob">{f.probability_up*100:.1f}%</div>'
        f'<div class="forecast-label">{probability_label(f.probability_up)}</div>'
        f'<div class="confidence">AUC {auc_txt} · {auc_label(f.cv_auc)}</div>'
        '</div>'
    )
st.markdown('<div class="forecast-grid">' + ''.join(forecast_cards) + '</div>', unsafe_allow_html=True)

st.markdown("### 차트")
st.plotly_chart(make_chart(features, result["support"], result["resistance"]), use_container_width=True, config={"displayModeBar": False})

st.markdown("### 상세 예측")
rows = []
for f in result["forecasts"]:
    rows.append({
        "기간": f"{f.horizon}일",
        "상승확률": f"{f.probability_up*100:.1f}%",
        "유사패턴": f"{f.similar_up_rate*100:.1f}%",
        "예상 중앙": price_text(f.price_median, ticker),
        "예상 범위": f"{price_text(f.price_low, ticker)} ~ {price_text(f.price_high, ticker)}",
        "검증 정확도": f"{f.cv_accuracy*100:.1f}%" if pd.notna(f.cv_accuracy) else "N/A",
        "균형 정확도": f"{f.cv_balanced_accuracy*100:.1f}%" if pd.notna(f.cv_balanced_accuracy) else "N/A",
        "AUC": f"{f.cv_auc:.3f}" if f.cv_auc is not None else "N/A",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("📖 결과를 아주 쉽게 읽는 법", expanded=True):
    st.markdown(
        """
- **추세**: 현재 가격과 20·60일 이동평균선의 위치로 본 흐름입니다.
- **RSI**: 대략 30 이하면 약한 상태, 50 전후 중립, 70 이상이면 과열 가능성을 봅니다.
- **지지/저항 후보**: 최근 60거래일 저가·고가 분포에서 계산한 가격대입니다.
- **상승확률**: 머신러닝 70% + 현재와 비슷한 과거 35개 구간 30%를 합친 값입니다.
- **유사패턴**: 과거에 지금과 비슷한 상황 이후 실제로 상승했던 비율입니다.
- **예상 범위**: 비슷한 과거 상황의 미래 수익률 중 20~80% 구간입니다.
- **AUC**: 0.50이면 거의 무작위입니다. 대략 0.60 이상부터 참고 가치가 조금 커지고, 0.70 이상이면 상대적으로 강한 편으로 봅니다.
        """
    )

with st.expander("⚙️ 프로그램이 내부에서 하는 일"):
    st.markdown(
        """
1. Yahoo Finance → Yahoo 직접 연결 → Stooq 순으로 일봉 OHLCV를 가져옵니다.
2. 수익률, 이동평균선, RSI, MACD, ATR, 변동성, 거래량, 볼린저 위치 등을 계산합니다.
3. 과거 각 날짜의 1·5·20거래일 뒤 상승/하락을 정답으로 만듭니다.
4. `TimeSeriesSplit`으로 과거→미래 순서 검증을 합니다.
5. Random Forest가 최신 날짜의 상승확률을 계산합니다.
6. 현재와 가장 비슷한 과거 35개 구간을 찾아 실제 이후 수익률을 비교합니다.
7. 두 결과를 결합해 최종 확률과 예상가격 범위를 표시합니다.
        """
    )

with st.expander("⚠️ 주의"):
    st.markdown(
        """
- 실적 발표, 가이던스, 소송, 인수합병, 금리·CPI 같은 이벤트는 차트만으로 예측하기 어렵습니다.
- AUC가 0.5대 초반이면 상승확률 숫자가 높거나 낮아도 신뢰를 크게 두지 않는 편이 좋습니다.
- 이 도구는 투자 판단 보조용이며 매수·매도 신호를 보장하지 않습니다.
        """
    )
