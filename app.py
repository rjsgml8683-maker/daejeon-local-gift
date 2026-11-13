import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

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


@st.cache_data(show_spinner=False)
def load_coords() -> pd.DataFrame:
    """행정동 중심 좌표 (전국일반음식점표준데이터의 업소 좌표 중앙값)"""
    try:
        return pd.read_csv("dong_coords.csv")
    except Exception:
        return pd.DataFrame(columns=["지역", "동", "lat", "lon"])


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


def headline(row, grade):
    """한 줄 진단 문장."""
    body = {"안심": "지역화폐 매출이 꾸준한 편입니다",
            "주의": "지역화폐 매출이 때때로 흔들리는 편입니다",
            "조심": "지역화폐 매출이 크게 흔들리는 편입니다"}[grade]
    tail = ""
    top = max([("정책", row["정책점수"]), ("고객층", row["집중점수"]),
               ("하락", row["추세점수"])], key=lambda x: x[1])
    if top[1] >= 60:
        tail = {"정책": " 다만 예산 정책에는 특히 민감한 상권입니다",
                "고객층": " 다만 손님층이 한쪽에 몰려 있는 상권입니다",
                "하락": " 다만 최근 몇 달 계속 줄어드는 중입니다"}[top[0]]
    return body + tail


def advise(row, slope):
    """가장 두드러진 한 가지를 중심으로 제안한다."""
    cards = []
    if row["정책점수"] >= 60:
        cards.append((
            "예산이 줄면 크게 흔들려요",
            f"작년처럼 지역화폐 예산이 줄었을 때, 이 동네 매출은 평소의 "
            f"**{row['잔존율']:.0f}% 수준**까지 떨어졌습니다.",
            "지역화폐 비수기를 대비해 운영비 두세 달치를 따로 두세요. "
            "할인율이나 발행 한도가 바뀐다는 시청 공지는 미리 챙겨 보시고요."))
    if row["집중점수"] >= 60:
        cards.append((
            "손님층이 한쪽에 몰려 있어요",
            f"결제의 **{row['최대고객층비중']:.0f}%**가 {row['최대고객층']} 손님입니다.",
            "이 손님들이 줄면 타격이 큽니다. 다른 연령대가 찾을 만한 메뉴나 "
            "시간대를 하나쯤 만들어 두면 완충이 됩니다."))
    if row["하락개월"] >= 4 or row["최근추세"] < -10:
        cards.append((
            "최근 계속 줄고 있어요",
            f"최근 6개월 중 **{int(row['하락개월'])}개월**이 전달보다 낮았습니다.",
            "가까운 동네로 손님이 옮겨가고 있는지 확인해 보세요. "
            "비교 탭에서 이웃 상권과 나란히 볼 수 있습니다."))
    if not cards:
        cards.append((
            "지금은 별다른 신호가 없어요",
            "세 가지 지표 모두 대전 평균 범위 안에 있습니다.",
            "분기에 한 번 정도 이 화면을 열어 흐름만 확인하시면 충분합니다."))
    if slope > 0.005:
        cards.append((
            "다만 최근에는 올라가는 중",
            "지역화폐 소비 비중이 오르는 추세입니다.",
            "늘어난 방문을 단골로 묶을 방법을 지금 준비해 두면 좋습니다."))
    return cards


# ──────────────────────────────── 화면
st.title("우리 동네 지역화폐 장사 진단")

with st.sidebar:
    st.header("설정")
    mobile = st.checkbox("휴대폰으로 보기", value=False,
                         help="글자가 겹쳐 보이면 켜세요")
    with st.expander("분석 조건 (전문가용)"):
        years = st.slider("분석 기간 (연월)", 2023, 2026, (2024, 2025))
        min_share = st.number_input("최소 평균 비중 (%)", 0.0, 5.0, 0.3, 0.1)
        if st.button("가양동 · 가오동 불러오기", use_container_width=True):
            st.session_state.update(gu="동구", dong="가양동",
                                    pick=["동구 가양동", "동구 가오동"])
    start_ym, end_ym = f"{years[0]}01", f"{years[1]}12"
    st.divider()
    st.caption("한국조폐공사가 공개한 지역사랑상품권 결제 자료로 만들었습니다. "
               "가게 한 곳이 아니라 동네 단위로 보여 드립니다.")

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

view_mode = st.radio("보기", ["사장님용", "분석용"], horizontal=True,
                     label_visibility="collapsed")
st.write("")

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
grade = "조심" if score >= 66 else ("주의" if score >= 33 else "안심")
color = {"조심": "red", "주의": "orange", "안심": "green"}[grade]

d = m[(m["지역"] == gu) & (m["동"] == dong)].sort_values("crtr_ym")
share = d["점유율"].to_numpy()
months = d["crtr_ym"].tolist()
recent = share[-3:].mean()
center, lowb, highb, slope = forecast(share, 3)

st.markdown(f"## :{color}[{grade}] · {dong}은 {headline(row, grade)}")
st.write("")

if view_mode == "분석용":
    st.caption(f"변동계수 {row['변동계수']:.3f} · 24개월 평균 비중 "
               f"{row['평균점유율']:.2f}% · 종합 {score:.0f}점")

for title, fact, action in advise(row, slope):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(fact)
        st.caption(action)

st.write("")
st.markdown("#### 우리 동네 지역화폐 소비가 어떻게 움직였나")
st.caption("대전 전체 지역화폐 결제액 가운데 이 동네가 차지한 몫입니다. "
           "맨 뒤 세 칸은 지금 흐름이 이어진다면 어떻게 될지 내다본 값입니다.")


def next_ym(ym, k):
    t = int(ym[:4]) * 12 + int(ym[4:]) - 1 + k
    return f"{t // 12:04d}{t % 12 + 1:02d}"


fut = [next_ym(months[-1], i) for i in range(1, 4)]
pad = [np.nan] * (len(share) - 1) + [share[-1]]
front = pd.DataFrame(index=months + fut)
front["지금까지"] = list(share) + [np.nan] * 3
front["앞으로 예상"] = pad + list(center)
st.line_chart(front, height=300, color=[GREEN, "#bdbdbd"])
st.caption(f"최근 석 달 평균 {recent:.2f}%, 석 달 뒤 예상 {center[-1]:.2f}%. "
           "예상은 지난 1년 흐름을 이어 본 값으로 평균 20% 정도 어긋납니다.")

st.write("")
st.markdown("##### 더 자세히 보기")

if view_mode == "사장님용":
    names = ["우리 동네 위치", "누가 쓰나", "달마다 얼마나"]
else:
    names = ["우리 동네 위치", "누가 쓰나", "달마다 얼마나",
             "다른 동네와 비교", "분석 방법", "이 진단의 한계"]
tabs = st.tabs(names)
t0, t1, t2 = tabs[0], tabs[1], tabs[2]
t3 = tabs[3] if len(tabs) > 3 else st.empty()
t4 = tabs[4] if len(tabs) > 4 else st.empty()
t5 = tabs[5] if len(tabs) > 5 else st.empty()

with t0:
    coords = load_coords()
    if coords.empty:
        st.info("좌표 파일(dong_coords.csv)이 저장소에 없어 지도를 표시할 수 없습니다.")
    else:
        mp = stat.merge(coords, on=["지역", "동"], how="inner")
        if mp.empty:
            st.info("좌표와 연결된 행정동이 없습니다.")
        else:
            mp["size"] = 120 + mp["평균점유율"] / mp["평균점유율"].max() * 500
            def rgb(v):
                if v >= 66:
                    return "#d1495b"
                return "#f0a202" if v >= 33 else "#157f5a"
            mp["color"] = mp["리스크점수"].apply(rgb)
            sel = (mp["지역"] == gu) & (mp["동"] == dong)
            mp.loc[sel, "color"] = "#1c1c1c"
            mp.loc[sel, "size"] = mp["size"].max() * 1.3
            if HAS_FOLIUM:
                fm = folium.Map(location=[36.345, 127.385], zoom_start=11,
                                tiles=None, control_scale=True)
                folium.TileLayer(
                    tiles="https://xdworld.vworld.kr/2d/Base/service/{z}/{x}/{y}.png",
                    attr="VWorld", name="브이월드").add_to(fm)
                for r in mp.itertuples():
                    picked = (r.지역 == gu and r.동 == dong)
                    folium.CircleMarker(
                        location=[r.lat, r.lon],
                        radius=6 + r.평균점유율 / mp["평균점유율"].max() * 18,
                        color="#1c1c1c" if picked else rgb(r.리스크점수),
                        weight=3 if picked else 1,
                        fill=True, fill_opacity=0.65,
                        fill_color=rgb(r.리스크점수),
                        tooltip=f"{r.지역} {r.동} · 리스크 {r.리스크점수:.0f}점 "
                                f"· 비중 {r.평균점유율:.2f}%").add_to(fm)
                st_folium(fm, height=460, use_container_width=True,
                          returned_objects=[])
            else:
                st.map(mp, latitude="lat", longitude="lon",
                       size="size", color="color", zoom=10.5)
            top = mp.nlargest(5, "리스크점수")[["지역", "동", "리스크점수"]]
            top["리스크점수"] = top["리스크점수"].round(0)
            st.caption("리스크가 높은 상권 : " +
                       ", ".join(f"{r.지역} {r.동}({r.리스크점수:.0f})" for r in top.itertuples()))

with t1:
    st.caption("최근 1년간 결제액을 성별 · 나이대로 나눈 모습입니다. "
               "한쪽으로 몰려 있을수록 그 손님들이 줄었을 때 타격이 큽니다.")
    sub = df[(df["지역"] == gu) & (df["동"] == dong) &
             (df["crtr_ym"].isin(months[-12:]))]
    pv = sub.pivot_table(index="연령대", columns="성별",
                         values="stlm_amt", aggfunc="sum").fillna(0)
    st.bar_chart((pv / pv.values.sum() * 100).round(1), height=320)
    st.caption(f"가장 큰 손님층은 {row['최대고객층']}으로 전체의 "
               f"{row['최대고객층비중']:.0f}%입니다.")

with t2:
    st.caption("달마다 이 동네에서 지역화폐로 결제된 금액입니다. "
               "전체가 함께 오르내리는 것은 상권이 아니라 예산 때문입니다.")
    st.bar_chart((d.set_index("crtr_ym")["결제금액"] / 1e8).rename("억 원"), height=320)

if view_mode == "분석용":
  with t3:
    options = [f"{r.지역} {r.동}" for r in stat.itertuples()]
    default = [o for o in st.session_state.get("pick", [f"{gu} {dong}"]) if o in options]
    pick = st.multiselect("견주어 볼 동네", options, default=default or [f"{gu} {dong}"])
    if pick:
        wide = m.assign(상권=m["지역"] + " " + m["동"])
        wide = wide[wide["상권"].isin(pick)].pivot(
            index="crtr_ym", columns="상권", values="점유율")
        st.line_chart(wide, height=320)
    st.caption("같은 대전 안에서도 동네마다 흐름이 꽤 다릅니다.")

if view_mode == "분석용":
  with t4:
    st.markdown("##### 이 동네의 세 가지 지표")
    g1, g2, g3 = st.columns(3)
    for col, label, val, note in [
            (g1, "정책 의존", row["정책점수"], f"예산 저점기 잔존율 {row['잔존율']:.1f}%"),
            (g2, "손님층 집중", row["집중점수"],
             f"{row['최대고객층']} {row['최대고객층비중']:.0f}%"),
            (g3, "하락 지속", row["추세점수"],
             f"6개월 중 {int(row['하락개월'])}개월 하락")]:
        col.markdown(f"**{label}** {val:.0f}점")
        col.progress(min(int(val), 100))
        col.caption(note)
    st.caption(f"변동계수 {row['변동계수']:.3f} · 24개월 평균 비중 {row['평균점유율']:.2f}% · "
               f"종합 {score:.0f}점 (정책 40% + 집중 30% + 하락 30%)")

    st.markdown("##### 대전 전체 순위")
    view = stat.copy()
    view["등급"] = pd.cut(view["리스크점수"], [-1, 33, 66, 101],
                        labels=["안심", "주의", "조심"])
    for c in ["리스크점수", "정책점수", "집중점수", "추세점수"]:
        view[c] = view[c].round(0)
    view["잔존율"] = view["잔존율"].round(1)
    st.dataframe(view[["지역", "동", "리스크점수", "등급", "정책점수",
                       "집중점수", "추세점수", "잔존율", "최대고객층"]],
                 hide_index=True, use_container_width=True)

if view_mode == "분석용":
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
