import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="우리 상권 지역화폐 리스크 진단",
                   page_icon="📊", layout="wide")

URL = "https://apis.data.go.kr/B190001/localGiftsPaymentV3/paymentsV3"
REGIONS = {"30110": "동구", "30140": "중구", "30170": "서구",
           "30200": "유성구", "30230": "대덕구"}
AGES = {"01": "19세 이하", "02": "20대", "03": "30대",
        "04": "40대", "05": "50대", "06": "60대 이상", "99": "미상"}
GREEN = "#157f5a"
W_POLICY, W_MIX, W_TREND = 0.4, 0.3, 0.3   # 가중치


# ──────────────────────────────── 데이터
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load(start_ym: str, end_ym: str) -> pd.DataFrame:
    key = st.secrets["SERVICE_KEY"]
    rows, bar = [], st.progress(
        0.0, text="공공데이터포털에서 결제 데이터를 받는 중입니다. 처음 접속 시 1분 정도 걸립니다…")
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
    df["crtr_ym"] = df["crtr_ym"].astype(str)
    df["지역"] = df["usage_rgn_cd"].map(REGIONS)
    df["동"] = df["emd_nm"].fillna("기타")
    df["연령대"] = df["par_ag"].map(AGES).fillna("미상")
    df["성별"] = df["par_gend"].map({"M": "남성", "F": "여성"}).fillna("미상")
    for c in ["stlm_amt", "stlm_nocs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def build(df: pd.DataFrame, min_share: float):
    """월 × 행정동 집계와 세 가지 리스크 지표를 계산한다."""
    base = df[df["동"] != "기타"]
    m = (base.groupby(["crtr_ym", "지역", "동"], as_index=False)
         .agg(결제금액=("stlm_amt", "sum"), 결제건수=("stlm_nocs", "sum")))
    m["점유율"] = m["결제금액"] / m.groupby("crtr_ym")["결제금액"].transform("sum") * 100
    n_month = m["crtr_ym"].nunique()
    m = m[m.groupby(["지역", "동"])["crtr_ym"].transform("count") == n_month]

    amt = m.pivot_table(index=["지역", "동"], columns="crtr_ym", values="결제금액")
    shr = m.pivot_table(index=["지역", "동"], columns="crtr_ym", values="점유율")
    keep = shr.mean(axis=1) >= min_share
    amt, shr = amt[keep], shr[keep]
    cols = list(shr.columns)

    stat = pd.DataFrame(index=shr.index)
    stat["평균점유율"] = shr.mean(axis=1)
    stat["변동계수"] = shr.std(axis=1) / shr.mean(axis=1)

    # ① 정책 의존 — 대전 전체 결제가 가장 적었던 달들과 가장 많았던 달들을 비교
    total = m.groupby("crtr_ym")["결제금액"].sum().sort_values()
    k = max(2, len(cols) // 4)
    low_m, high_m = list(total.index[:k]), list(total.index[-k:])
    stat["잔존율"] = amt[low_m].mean(axis=1) / amt[high_m].mean(axis=1) * 100

    # ② 고객층 집중 — 성별 × 연령대 결제액의 허핀달 지수(0~1, 클수록 쏠림)
    recent = base[base["crtr_ym"].isin(cols[-12:])]
    mix = (recent.groupby(["지역", "동", "성별", "연령대"])["stlm_amt"].sum()
           .groupby(level=[0, 1]).apply(lambda x: ((x / x.sum()) ** 2).sum()))
    stat["집중도"] = mix.reindex(stat.index)
    top = (recent.groupby(["지역", "동", "성별", "연령대"])["stlm_amt"].sum()
           .groupby(level=[0, 1]).idxmax().apply(lambda t: f"{t[2]} {t[3]}"))
    stat["최대고객층"] = top.reindex(stat.index)
    share_top = (recent.groupby(["지역", "동", "성별", "연령대"])["stlm_amt"].sum()
                 .groupby(level=[0, 1]).apply(lambda x: x.max() / x.sum() * 100))
    stat["최대고객층비중"] = share_top.reindex(stat.index)

    # ③ 하락 지속 — 최근 6개월 중 전월보다 떨어진 달 수와 3개월 추세
    s6 = shr[cols[-6:]]
    stat["하락개월"] = (s6.diff(axis=1).iloc[:, 1:] < 0).sum(axis=1)
    stat["최근추세"] = (shr[cols[-3:]].mean(axis=1) /
                    shr[cols[-6:-3]].mean(axis=1) - 1) * 100

    # 0~100 점수화 (클수록 위험)
    def rank01(sr, reverse=False):
        r = sr.rank(pct=True)
        return (1 - r if reverse else r) * 100

    stat["정책점수"] = rank01(stat["잔존율"], reverse=True)
    stat["집중점수"] = rank01(stat["집중도"])
    stat["추세점수"] = rank01(stat["하락개월"] * 10 - stat["최근추세"])
    stat["리스크점수"] = (stat["정책점수"] * W_POLICY +
                     stat["집중점수"] * W_MIX +
                     stat["추세점수"] * W_TREND)
    return m, stat.reset_index().sort_values("리스크점수", ascending=False), low_m, high_m


# ──────────────────────────────── 예측
def forecast(series: np.ndarray, months: int = 3):
    y = series[-12:] if len(series) >= 12 else series
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    sigma = (y - (slope * x + intercept)).std(ddof=1) if len(y) > 2 else 0.0
    fx = np.arange(len(y), len(y) + months)
    center = np.clip(slope * fx + intercept, 0.01, None)
    return center, np.clip(center - 1.96 * sigma, 0.01, None), center + 1.96 * sigma, slope


def advise(row, slope):
    acts = []
    if row["정책점수"] >= 60:
        acts.append(f"예산이 줄었던 달 매출이 많던 달의 **{row['잔존율']:.1f}%**까지 내려갔습니다. "
                    "지역화폐 비수기를 가정해 2~3개월치 고정비를 따로 확보해 두세요.")
    if row["집중점수"] >= 60:
        acts.append(f"결제의 **{row['최대고객층비중']:.0f}%**가 {row['최대고객층']}에 몰려 있습니다. "
                    "이 고객층의 소비가 줄면 타격이 큽니다. 다른 연령대를 겨냥한 메뉴나 시간대를 검토하세요.")
    if row["하락개월"] >= 4 or row["최근추세"] < -10:
        acts.append(f"최근 6개월 중 **{int(row['하락개월'])}개월**이 전월보다 낮았습니다. "
                    "인근 상권으로 수요가 옮겨가고 있는지 확인하세요.")
    if slope > 0.005:
        acts.append("비중이 올라가는 추세입니다. 늘어난 방문을 단골로 묶을 방법을 준비하세요.")
    if not acts:
        acts.append("세 지표 모두 평균 범위입니다. 분기마다 비중 변화를 확인하는 정도로 충분합니다.")
    return acts


# ──────────────────────────────── 화면
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="st-"], button, input, select, textarea {
  font-family: 'Noto Sans KR', system-ui, sans-serif;
}
#MainMenu, header [data-testid="stToolbar"], footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

h1 { font-size: 2.1rem !important; font-weight: 700 !important;
     letter-spacing: -0.04em; margin-bottom: .2rem !important; }
h2, h3 { letter-spacing: -0.02em; }

/* 지표 카드 */
div[data-testid="stMetric"],
div[data-testid="column"] > div > div > div:has(> div[data-testid="stMetric"]) {
  background: #fff; border: 1px solid #e6e6e6; border-radius: 14px;
  padding: 16px 18px;
}
div[data-testid="stMetricLabel"] p { font-size: .82rem; color: #6b6b6b; }
div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 700; }
div[data-testid="stMetricDelta"] { font-size: .8rem; }

/* 탭 */
button[data-baseweb="tab"] {
  background: #fafafa; border: 1px solid #ededed; border-radius: 999px;
  padding: 6px 16px; margin-right: 8px; font-size: .9rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
  background: #157F5A; border-color: #157F5A; color: #fff;
}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }

/* 안내 박스 */
div[data-testid="stAlert"] {
  border-radius: 14px; border: 1px solid #e6e6e6; background: #fafafa;
}
div[data-testid="stAlert"] p { color: #2b2b2b; }

/* 사이드바 */
section[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #ededed; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

/* 표 */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("우리 상권 지역화폐 리스크 진단")

with st.sidebar:
    st.header("분석 조건")
    years = st.slider("분석 기간 (연월)", 2023, 2026, (2024, 2025))
    start_ym, end_ym = f"{years[0]}01", f"{years[1]}12"
    min_share = st.number_input("최소 평균 점유율 (%)", 0.0, 5.0, 0.3, 0.1)
    mobile = st.checkbox("모바일 보기 (세로로 쌓기)", value=False,
                         help="휴대폰에서 글자가 겹쳐 보이면 켜세요")
    st.divider()
    st.caption("리스크 점수 = 정책 의존 40% + 고객층 집중 30% + 하락 지속 30%. "
               "대전 안에서의 상대 순위를 0~100으로 환산합니다.")
    st.caption("폐업 위험이 아니라 매출이 흔들릴 가능성을 봅니다. "
               "폐업률과의 상관은 확인되지 않았습니다(r=0.06, p=0.65).")
    go = st.button("가양동 · 가오동 불러오기", use_container_width=True)
    if go:
        st.session_state.update(gu="동구", dong="가양동",
                                pick=["동구 가양동", "동구 가오동"])

if "SERVICE_KEY" not in st.secrets:
    st.error("인증키가 설정되지 않았습니다. 앱 설정의 Secrets에 SERVICE_KEY를 등록하세요.")
    st.stop()

df = load(start_ym, end_ym)
if df.empty:
    st.warning("해당 기간에 데이터가 없습니다. 기간을 바꿔보세요.")
    st.stop()

m, stat, low_m, high_m = build(df, min_share)


def fmt_ym(ym):
    return f"{ym[:4]}년 {int(ym[4:])}월"


span = sorted(m["crtr_ym"].unique())
st.caption(f"한국조폐공사 지역사랑상품권 결제정보 · 대전광역시 · "
           f"{fmt_ym(span[0])} ~ {fmt_ym(span[-1])} 기준 ({len(stat)}개 행정동)")

c1, c2 = (st.container(), st.container()) if mobile else st.columns(2)
gu_list = list(REGIONS.values())
gu = c1.selectbox("자치구", gu_list,
                  index=gu_list.index(st.session_state.get("gu", "유성구")))
pool = stat[stat["지역"] == gu].sort_values("평균점유율", ascending=False)
dong_list = pool["동"].tolist()
_d = st.session_state.get("dong")
dong = c2.selectbox("행정동", dong_list,
                    index=dong_list.index(_d) if _d in dong_list else 0)

row = stat[(stat["지역"] == gu) & (stat["동"] == dong)].iloc[0]
score = row["리스크점수"]
grade = "높음" if score >= 66 else ("보통" if score >= 33 else "낮음")
color = "red" if grade == "높음" else ("orange" if grade == "보통" else "green")

d = m[(m["지역"] == gu) & (m["동"] == dong)].sort_values("crtr_ym")
share = d["점유율"].to_numpy()
months = d["crtr_ym"].tolist()
recent = share[-3:].mean()
center, lowb, highb, slope = forecast(share, 3)

if mobile:
    (k1, k2), (k3, k4) = st.columns(2), st.columns(2)
else:
    k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"### :{color}[리스크 {grade}]")
k1.caption(f"종합 {score:.0f}점 / 100")
k2.metric("정책 의존", f"{row['정책점수']:.0f}점", f"잔존율 {row['잔존율']:.1f}%",
          delta_color="off")
k3.metric("고객층 집중", f"{row['집중점수']:.0f}점",
          f"{row['최대고객층']} {row['최대고객층비중']:.0f}%", delta_color="off")
k4.metric("하락 지속", f"{row['추세점수']:.0f}점",
          f"6개월 중 {int(row['하락개월'])}개월 하락", delta_color="off")

st.info(f"**{gu} {dong}** — 최근 3개월 지역화폐 결제 비중은 **{recent:.2f}%**, "
        f"3개월 뒤 예상은 **{center[-1]:.2f}%**입니다. "
        f"24개월 평균 비중은 {row['평균점유율']:.2f}%입니다.")

st.subheader("무엇을 하면 좋을까")
for a in advise(row, slope):
    st.markdown(f"- {a}")

t1, t2, t3, t4, t5 = st.tabs(
    ["비중 추이와 예측", "고객층 구성", "월별 결제금액", "대전 전체 비교", "지표 설명"])

with t1:
    st.caption("대전 전체 지역화폐 결제액 중 이 동네의 비중. 마지막 세 구간은 추세를 연장한 예상치입니다.")

    def next_ym(ym, k):
        t = int(ym[:4]) * 12 + int(ym[4:]) - 1 + k
        return f"{t // 12:04d}{t % 12 + 1:02d}"

    fut = [next_ym(months[-1], i) for i in range(1, 4)]
    pad = [np.nan] * (len(share) - 1) + [share[-1]]
    chart = pd.DataFrame(index=months + fut)
    chart["실적"] = list(share) + [np.nan] * 3
    chart["예상"] = pad + list(center)
    chart["예상 하한"] = pad + list(lowb)
    chart["예상 상한"] = pad + list(highb)
    st.line_chart(chart, height=340, color=[GREEN, "#9a9a9a", "#dfdfdf", "#dfdfdf"])
    st.caption("최근 12개월 선형 추세를 연장한 값입니다. 과거 자료로 검증한 3개월 예측 "
               "평균 오차율은 약 20%(중앙값 11%)이며, 정책이 바뀌면 크게 벗어날 수 있습니다.")

with t2:
    st.caption("최근 12개월 결제액을 성별 · 연령대로 나눈 구성입니다. 한 칸에 쏠릴수록 리스크가 큽니다.")
    sub = df[(df["지역"] == gu) & (df["동"] == dong) &
             (df["crtr_ym"].isin(months[-12:]))]
    pv = sub.pivot_table(index="연령대", columns="성별",
                         values="stlm_amt", aggfunc="sum").fillna(0)
    st.bar_chart((pv / pv.values.sum() * 100).round(1), height=320)
    st.caption(f"가장 큰 고객층은 **{row['최대고객층']}**으로 전체의 "
               f"{row['최대고객층비중']:.0f}%를 차지합니다.")

with t3:
    st.caption("금액 자체는 발행 예산과 할인 정책에 따라 출렁입니다. "
               f"결제가 가장 적었던 달은 {', '.join(fmt_ym(x) for x in sorted(low_m)[:3])} 등입니다.")
    st.bar_chart((d.set_index("crtr_ym")["결제금액"] / 1e8).rename("억 원"), height=320)

with t4:
    options = [f"{r.지역} {r.동}" for r in stat.itertuples()]
    default = [o for o in st.session_state.get("pick", [f"{gu} {dong}"]) if o in options]
    pick = st.multiselect("비교할 상권", options, default=default or [f"{gu} {dong}"])
    if pick:
        wide = m.assign(상권=m["지역"] + " " + m["동"])
        wide = wide[wide["상권"].isin(pick)].pivot(
            index="crtr_ym", columns="상권", values="점유율")
        st.line_chart(wide, height=320)
    view = stat.copy()
    view["등급"] = pd.cut(view["리스크점수"], [-1, 33, 66, 101],
                        labels=["낮음", "보통", "높음"])
    for c in ["리스크점수", "정책점수", "집중점수", "추세점수"]:
        view[c] = view[c].round(0)
    view["잔존율"] = view["잔존율"].round(1)
    st.dataframe(view[["지역", "동", "리스크점수", "등급", "정책점수",
                       "집중점수", "추세점수", "잔존율", "최대고객층"]],
                 hide_index=True, use_container_width=True)

with t5:
    st.markdown(f"""
**리스크 점수 구성**

| 지표 | 가중치 | 무엇을 재나 |
| --- | --- | --- |
| 정책 의존 | 40% | 대전 전체 결제가 가장 적었던 달의 매출이, 가장 많았던 달의 몇 %인지(잔존율). 낮을수록 예산에 따라 크게 흔들린다 |
| 고객층 집중 | 30% | 성별 × 연령대 결제액의 허핀달 지수. 한 집단에 쏠릴수록 그 집단의 소비가 줄 때 타격이 크다 |
| 하락 지속 | 30% | 최근 6개월 중 전월보다 낮았던 달 수와 3개월 추세 |

세 지표를 대전 77개 동 안에서 순위로 환산해 0~100점으로 만들고 가중 합산합니다.

**무엇을 보여주지 못하나**

- 상권 전체 매출이 아닙니다. 카드 · 현금 · 간편결제는 포함되지 않습니다.
- 개별 점포가 아니라 행정동 단위입니다.
- 폐업 위험을 예측하지 않습니다. 대전 일반음식점 63,889건의 폐업률과 대조한 결과
  유의미한 상관이 없었습니다(r=0.06, p=0.65). 시차를 둔 비교와 개업 대비 폐업 배율로도
  마찬가지였습니다. 따라서 **매출이 흔들릴 가능성**을 보는 지표로 한정합니다.

**참고 — 2025년에 실제로 있었던 일**

대전 전체 지역화폐 결제액은 2025년 6월 약 14억 원에서 11월 약 889억 원으로 60배 넘게 늘었습니다.
상권이 좋아진 것이 아니라 발행 예산이 복원된 결과입니다. 금액이 아니라 비중을 보는 이유입니다.
""")

st.divider()
st.caption("출처 : 공공데이터포털 한국조폐공사 지역사랑상품권 결제정보, "
           "행정안전부 전국일반음식점표준데이터")
