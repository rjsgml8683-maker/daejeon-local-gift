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
GRADE_COLOR = {"안정": "#3ecf8e", "주의": "#ffd166", "위험": "#f4978e"}


# ──────────────────────────────── 데이터 수집
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
                "cond[crtr_ym::GTE]": start_ym,
                "cond[crtr_ym::LTE]": end_ym,
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
    df["stlm_amt"] = pd.to_numeric(df["stlm_amt"], errors="coerce").fillna(0)
    df["stlm_nocs"] = pd.to_numeric(df["stlm_nocs"], errors="coerce").fillna(0)
    return df


# ──────────────────────────────── 지표 계산
def build(df: pd.DataFrame, min_share: float):
    m = (df[df["동"] != "기타"]
         .groupby(["crtr_ym", "지역", "동"], as_index=False)
         .agg(결제금액=("stlm_amt", "sum"), 결제건수=("stlm_nocs", "sum")))

    m["점유율"] = m["결제금액"] / m.groupby("crtr_ym")["결제금액"].transform("sum") * 100

    n_month = m["crtr_ym"].nunique()
    full = m.groupby(["지역", "동"])["crtr_ym"].transform("count") == n_month
    m = m[full]

    stat = (m.groupby(["지역", "동"])["점유율"]
            .agg(평균점유율="mean", 표준편차="std").reset_index())
    stat["변동계수"] = stat["표준편차"] / stat["평균점유율"]
    stat = stat[stat["평균점유율"] >= min_share].sort_values("변동계수", ascending=False)
    return m, stat


def grade_of(cv, lo, hi):
    return "안정" if cv < lo else ("주의" if cv < hi else "위험")


# ──────────────────────────────── 화면
st.title("우리 상권 리스크 진단")
st.caption("한국조폐공사 지역사랑상품권 결제정보 · 대전광역시")

with st.sidebar:
    st.header("분석 조건")
    years = st.slider("분석 기간 (연월)", 2023, 2026, (2024, 2025))
    start_ym, end_ym = f"{years[0]}01", f"{years[1]}12"
    min_share = st.number_input("최소 평균 점유율 (%)", 0.0, 5.0, 0.3, 0.1,
                                help="결제가 거의 없는 외곽 지역을 제외합니다")
    st.divider()
    st.caption("등급은 대전 안에서의 상대 위치입니다. "
               "변동계수 분포를 삼등분해 안정·주의·위험으로 나눕니다.")

if "SERVICE_KEY" not in st.secrets:
    st.error("인증키가 설정되지 않았습니다. 앱 설정의 Secrets에 SERVICE_KEY를 등록하세요.")
    st.stop()

df = load(start_ym, end_ym)
if df.empty:
    st.warning("해당 기간에 데이터가 없습니다. 기간을 바꿔보세요.")
    st.stop()

m, stat = build(df, min_share)
lo, hi = stat["변동계수"].quantile([1 / 3, 2 / 3])

c1, c2 = st.columns(2)
gu = c1.selectbox("자치구", list(REGIONS.values()), index=3)
pool = stat[stat["지역"] == gu].sort_values("평균점유율", ascending=False)
dong = c2.selectbox("행정동", pool["동"].tolist())

row = stat[(stat["지역"] == gu) & (stat["동"] == dong)].iloc[0]
cv = row["변동계수"]
g = grade_of(cv, lo, hi)

d = m[(m["지역"] == gu) & (m["동"] == dong)].sort_values("crtr_ym")
share = d["점유율"].to_numpy()
recent, prev = share[-3:].mean(), share[-6:-3].mean()
chg = (recent - prev) / prev * 100

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"### :{'green' if g=='안정' else 'orange' if g=='주의' else 'red'}[{g}]")
k1.caption("리스크 등급")
k2.metric("변동계수", f"{cv:.3f}")
k3.metric("최근 3개월 점유율", f"{recent:.2f}%", f"{chg:+.1f}%")
k4.metric("24개월 평균 점유율", f"{row['평균점유율']:.2f}%")

st.info(
    f"**{gu} {dong}**의 최근 3개월 상권 점유율은 **{recent:.2f}%**로 "
    f"직전 3개월 대비 **{chg:+.1f}%** 변화했습니다. "
    f"분석 기간 동안 {share.min():.2f}% ~ {share.max():.2f}% 사이를 오갔습니다."
)

t1, t2, t3 = st.tabs(["상권 점유율 추이", "월별 결제금액", "대전 전체 비교"])

with t1:
    st.caption("대전 전체 결제액 중 이 동네의 비중. 정책 영향을 걷어낸 상권의 체력입니다.")
    st.line_chart(d.set_index("crtr_ym")["점유율"], color=GREEN, height=320)

with t2:
    st.caption("금액 자체는 발행 예산과 할인 정책에 따라 크게 출렁입니다.")
    st.bar_chart((d.set_index("crtr_ym")["결제금액"] / 1e8).rename("억 원"), height=320)

with t3:
    pick = st.multiselect(
        "비교할 상권 (여러 개 선택 가능)",
        [f"{r.지역} {r.동}" for r in stat.itertuples()],
        default=[f"{gu} {dong}"])
    if pick:
        wide = m.assign(상권=m["지역"] + " " + m["동"])
        wide = wide[wide["상권"].isin(pick)].pivot(
            index="crtr_ym", columns="상권", values="점유율")
        st.line_chart(wide, height=320)

    view = stat.copy()
    view["등급"] = view["변동계수"].apply(lambda v: grade_of(v, lo, hi))
    view["평균점유율"] = view["평균점유율"].round(2)
    view["변동계수"] = view["변동계수"].round(3)
    st.dataframe(view[["지역", "동", "평균점유율", "변동계수", "등급"]],
                 hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "분석 단위는 행정동이며 개별 점포의 매출이 아닙니다. "
    "24개월 전체 자료가 있는 상권만 포함합니다. "
    "출처 : 공공데이터포털 한국조폐공사 지역사랑상품권 결제정보")
