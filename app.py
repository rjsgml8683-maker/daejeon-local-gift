import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="우리 상권 리스크 진단", page_icon="📊", layout="wide")

URL = "https://apis.data.go.kr/B190001/localGiftsPaymentV3/paymentsV3"
REGIONS = {"30110": "동구", "30140": "중구", "30170": "서구",
           "30200": "유성구", "30230": "대덕구"}
GREEN = "#157f5a"


# ──────────────────────────────── 데이터
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load(start_ym: str, end_ym: str) -> pd.DataFrame:
    key = st.secrets["SERVICE_KEY"]
    rows, bar = [], st.progress(0.0, text="공공데이터포털에서 결제 데이터를 받는 중…")
    for n, (code, name) in enumerate(REGIONS.items(), start=1):
        page = 1
        while True:
            res = requests.get(URL, params={
                "serviceKey": key, "page": page, "perPage": 1000,
                "cond[usage_rgn_cd::EQ]": code,
                "cond[crtr_ym::GTE]": start_ym, "cond[crtr_ym::LTE]": end_ym,
                "returnType": "JSON"}, timeout=60)
            res.raise_for_status()
            batch = res.json().get("data", [])
            rows += batch
            if len(batch) < 1000:
                break
            page += 1
            time.sleep(0.2)
        bar.progress(n / len(REGIONS), text=f"{name} 수집 완료 ({len(rows):,}건)")
    bar.empty()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["지역"] = df["usage_rgn_cd"].map(REGIONS)
    df["동"] = df["emd_nm"].fillna("기타")
    for c in ["stlm_amt", "stlm_nocs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def build(df: pd.DataFrame, min_share: float):
    m = (df[df["동"] != "기타"]
         .groupby(["crtr_ym", "지역", "동"], as_index=False)
         .agg(결제금액=("stlm_amt", "sum"), 결제건수=("stlm_nocs", "sum")))
    m["점유율"] = m["결제금액"] / m.groupby("crtr_ym")["결제금액"].transform("sum") * 100
    n_month = m["crtr_ym"].nunique()
    m = m[m.groupby(["지역", "동"])["crtr_ym"].transform("count") == n_month]

    stat = (m.groupby(["지역", "동"])["점유율"]
            .agg(평균점유율="mean", 표준편차="std").reset_index())
    stat["변동계수"] = stat["표준편차"] / stat["평균점유율"]
    stat = stat[stat["평균점유율"] >= min_share].sort_values("변동계수", ascending=False)
    return m, stat


# ──────────────────────────────── 예측
def forecast(series: np.ndarray, months: int = 3):
    """최근 12개월 선형 추세를 연장하고, 추세 잔차로 예상 구간을 만든다."""
    y = series[-12:] if len(series) >= 12 else series
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sigma = resid.std(ddof=1) if len(y) > 2 else 0.0

    fx = np.arange(len(y), len(y) + months)
    center = np.clip(slope * fx + intercept, 0.01, None)
    band = 1.96 * sigma
    return center, np.clip(center - band, 0.01, None), center + band, slope


def advise(grade: str, slope: float, recent: float, avg: float):
    """등급과 추세를 조합해 행동 제안을 만든다."""
    if grade == "변동 큼":
        head = "지역화폐 매출 비중이 달마다 크게 흔들리는 상권입니다."
        acts = ["지역화폐 매출에만 기대지 말고 결제 수단별 매출 비중을 따로 관리하세요.",
                "성수기 매출을 평달 운영비로 환산해 최소 2~3개월치 고정비를 확보해 두세요.",
                "할인율과 발행 한도에 관한 지자체 공지를 미리 확인하고 재고·인력 계획에 반영하세요."]
    elif grade == "보통":
        head = "평균적인 수준으로 움직이는 상권입니다."
        acts = ["분기마다 비중 변화를 확인하고, 두 분기 연속 하락하면 원인을 점검하세요.",
                "같은 자치구의 다른 상권과 비교해 우리 동네만의 흐름인지 확인하세요."]
    else:
        head = "지역화폐 매출 비중이 꾸준히 유지되는 상권입니다."
        acts = ["수요가 안정적이므로 재고와 인력 계획을 평달 기준으로 세워도 무리가 적습니다.",
                "비중이 일정한 만큼 지역화폐 할인 행사의 판촉 효과를 가늠하기 쉽습니다."]

    if slope < -0.005:
        acts.insert(0, "최근 비중이 내려가는 추세입니다. 인근 상권으로 수요가 옮겨가고 있는지 확인하세요.")
    elif slope > 0.005:
        acts.insert(0, "최근 비중이 올라가는 추세입니다. 늘어난 방문을 단골로 묶을 방법을 준비하세요.")

    if recent < avg * 0.8:
        acts.append("최근 비중이 24개월 평균보다 20% 이상 낮습니다. 자금 계획을 보수적으로 잡으세요.")
    return head, acts


# ──────────────────────────────── 화면
st.title("우리 상권 리스크 진단")
st.caption("한국조폐공사 지역사랑상품권 결제정보 · 대전광역시")

with st.sidebar:
    st.header("분석 조건")
    years = st.slider("분석 기간 (연월)", 2023, 2026, (2024, 2025))
    start_ym, end_ym = f"{years[0]}01", f"{years[1]}12"
    min_share = st.number_input("최소 평균 점유율 (%)", 0.0, 5.0, 0.3, 0.1)
    st.divider()
    st.caption("등급은 대전 안에서의 상대 위치입니다. 지역화폐 결제 비중의 "
               "변동계수를 삼등분해 안정 · 보통 · 변동 큼으로 나눕니다.")
    st.caption("폐업률과의 상관은 확인되지 않아(r=0.06, p=0.65) 폐업 예측이 아니라 "
               "지역화폐 의존도의 안정성 지표로 해석합니다.")

if "SERVICE_KEY" not in st.secrets:
    st.error("인증키가 설정되지 않았습니다. 앱 설정의 Secrets에 SERVICE_KEY를 등록하세요.")
    st.stop()

df = load(start_ym, end_ym)
if df.empty:
    st.warning("해당 기간에 데이터가 없습니다. 기간을 바꿔보세요.")
    st.stop()

m, stat = build(df, min_share)
lo, hi = stat["변동계수"].quantile([1 / 3, 2 / 3])


def grade_of(cv):
    return "안정" if cv < lo else ("보통" if cv < hi else "변동 큼")


c1, c2 = st.columns(2)
gu = c1.selectbox("자치구", list(REGIONS.values()), index=3)
pool = stat[stat["지역"] == gu].sort_values("평균점유율", ascending=False)
dong = c2.selectbox("행정동", pool["동"].tolist())

row = stat[(stat["지역"] == gu) & (stat["동"] == dong)].iloc[0]
cv, avg = row["변동계수"], row["평균점유율"]
g = grade_of(cv)

d = m[(m["지역"] == gu) & (m["동"] == dong)].sort_values("crtr_ym")
share = d["점유율"].to_numpy()
months = d["crtr_ym"].tolist()
recent, prev = share[-3:].mean(), share[-6:-3].mean()
chg = (recent - prev) / prev * 100
center, lowb, highb, slope = forecast(share, 3)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"### :{'green' if g=='안정' else 'orange' if g=='보통' else 'red'}[{g}]")
k1.caption("변동성 등급")
k2.metric("변동계수", f"{cv:.3f}")
k3.metric("최근 3개월 비중", f"{recent:.2f}%", f"{chg:+.1f}%")
k4.metric("3개월 뒤 예상", f"{center[-1]:.2f}%",
          f"{(center[-1] - recent) / recent * 100:+.1f}%")

head, acts = advise(g, slope, recent, avg)
st.info(f"**{gu} {dong}** — {head} 최근 3개월 지역화폐 결제 비중은 **{recent:.2f}%**로 "
        f"직전 3개월 대비 **{chg:+.1f}%** 변화했고, 24개월 평균은 {avg:.2f}%입니다.")

st.subheader("무엇을 하면 좋을까")
for a in acts:
    st.markdown(f"- {a}")

t1, t2, t3, t4 = st.tabs(["비중 추이와 예측", "월별 결제금액", "대전 전체 비교", "지표 설명"])

with t1:
    st.caption("대전 전체 지역화폐 결제액 중 이 동네의 비중. 마지막 세 구간은 추세를 연장한 예상치입니다.")
    fut = [f"+{i}개월" for i in range(1, 4)]
    chart = pd.DataFrame(index=months + fut)
    chart["실적"] = list(share) + [np.nan] * 3
    chart["예상"] = [np.nan] * len(share) + list(center)
    chart["예상 하한"] = [np.nan] * len(share) + list(lowb)
    chart["예상 상한"] = [np.nan] * len(share) + list(highb)
    st.line_chart(chart, height=340, color=[GREEN, "#9a9a9a", "#dfdfdf", "#dfdfdf"])
    st.caption("최근 12개월 선형 추세를 연장한 값입니다. 과거 자료로 검증한 3개월 예측 "
               "평균 오차율은 약 20%(중앙값 11%)이며, 정책이 바뀌면 크게 벗어날 수 있습니다.")

with t2:
    st.caption("금액 자체는 발행 예산과 할인 정책에 따라 출렁입니다. "
               "2025년 상반기 급감은 상권이 아니라 예산 축소의 결과입니다.")
    st.bar_chart((d.set_index("crtr_ym")["결제금액"] / 1e8).rename("억 원"), height=320)

with t3:
    pick = st.multiselect("비교할 상권",
                          [f"{r.지역} {r.동}" for r in stat.itertuples()],
                          default=[f"{gu} {dong}"])
    if pick:
        wide = m.assign(상권=m["지역"] + " " + m["동"])
        wide = wide[wide["상권"].isin(pick)].pivot(
            index="crtr_ym", columns="상권", values="점유율")
        st.line_chart(wide, height=320)
    view = stat.copy()
    view["등급"] = view["변동계수"].apply(grade_of)
    view["평균점유율"] = view["평균점유율"].round(2)
    view["변동계수"] = view["변동계수"].round(3)
    st.dataframe(view[["지역", "동", "평균점유율", "변동계수", "등급"]],
                 hide_index=True, use_container_width=True)

with t4:
    st.markdown("""
**이 지표가 보여주는 것**

대전 전체 지역사랑상품권 결제액 가운데 해당 행정동이 차지하는 비중입니다.
금액 자체는 발행 예산과 할인 정책에 따라 모든 지역이 함께 움직이므로,
비중으로 보아야 상권별 차이가 드러납니다.

**이 지표가 보여주지 못하는 것**

- 상권 전체 매출이 아닙니다. 카드 · 현금 · 간편결제는 포함되지 않습니다.
- 개별 점포의 매출이 아닙니다. 행정동 단위 집계입니다.
- 폐업 위험을 예측하지 않습니다.

**검증 내역**

전국일반음식점표준데이터 중 대전 63,889건을 행정동별로 집계해 2024~2025년
폐업률을 계산하고 변동계수와 비교했습니다. 결과는 r=0.06, p=0.65로 유의미한
상관이 없었고, 시차를 둔 비교와 개업 대비 폐업 배율, 추세 지표로도 마찬가지였습니다.
따라서 본 지표는 폐업 예측이 아니라 **지역화폐 의존도의 안정성**을 보는 용도로 한정합니다.
""")

st.divider()
st.caption("출처 : 공공데이터포털 한국조폐공사 지역사랑상품권 결제정보, "
           "행정안전부 전국일반음식점표준데이터")
