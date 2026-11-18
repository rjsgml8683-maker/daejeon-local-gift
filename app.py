import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

st.set_page_config(page_title="우리 동네 지역화폐 리스크 진단",
                   page_icon="📊", layout="wide")

URL = "https://apis.data.go.kr/B190001/localGiftsPaymentV3/paymentsV3"
REGIONS = {"30110": "동구", "30140": "중구", "30170": "서구",
           "30200": "유성구", "30230": "대덕구"}
AGES = {"01": "19세 이하", "02": "20대", "03": "30대",
        "04": "40대", "05": "50대", "06": "60대 이상", "99": "미상"}
GREEN = "#157f5a"
W_POLICY, W_MIX, W_TREND = 0.4, 0.3, 0.3   # 가중치


# ──────────────────────────────── 데이터
CACHE_FILE = "payments_cache.csv.gz"   # 공공데이터포털이 해외 서버(Streamlit)를 막을 때 쓰는 사본


def _fetch_api(key: str, start_ym: str, end_ym: str, bar):
    """공공데이터포털에서 받는다. 막히거나 JSON이 아니면 None."""
    rows = []
    for n, (code, name) in enumerate(REGIONS.items(), start=1):
        page = 1
        while True:
            payload = None
            for attempt in range(2):
                try:
                    res = requests.get(URL, params={
                        "serviceKey": key, "page": page, "perPage": 1000,
                        "cond[usage_rgn_cd::EQ]": code,
                        "cond[crtr_ym::GTE]": start_ym, "cond[crtr_ym::LTE]": end_ym,
                        "returnType": "JSON"}, timeout=30)
                    payload = res.json()
                    break
                except Exception:
                    time.sleep(1.5)
            if payload is None:
                return None
            batch = payload.get("data", [])
            rows += batch
            if len(batch) < 1000:
                break
            page += 1
            time.sleep(0.2)
        bar.progress(n / len(REGIONS), text=f"{name} 수집 완료 ({len(rows):,}건)")
    return rows


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load(start_ym: str, end_ym: str) -> pd.DataFrame:
    key = st.secrets["SERVICE_KEY"]
    bar = st.progress(
        0.0, text="공공데이터포털에서 결제 데이터를 받는 중입니다. 처음 접속 시 1분 정도 걸립니다…")
    rows = _fetch_api(key, start_ym, end_ym, bar)
    bar.empty()
    if rows is None:
        # 해외 IP 차단 등으로 API가 안 될 때: 한국에서 미리 받아 둔 같은 API 자료를 쓴다
        try:
            cache = pd.read_csv(CACHE_FILE, dtype=str)
        except Exception:
            return pd.DataFrame()
        cache = cache[(cache["crtr_ym"] >= start_ym) & (cache["crtr_ym"] <= end_ym)]
        cache = cache.astype(object).where(cache.notna(), None)
        rows = cache.to_dict("records")

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

    # ① 저점기 잔존율 — 대전 전체 결제가 가장 적었던 달들과 가장 많았던 달들을 비교
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

    # ④ 상권 유형 — 네 가지 성격으로 묶기 (k-means)
    mon = pd.Series([int(c[4:]) for c in cols], index=cols)
    seas = shr.T.groupby(mon).mean().T
    stat["계절진폭"] = (seas.max(axis=1) - seas.min(axis=1)) / seas.mean(axis=1)

    feat = stat[["평균점유율", "변동계수", "잔존율", "계절진폭"]]
    sim = pd.DataFrame(index=stat.index, columns=stat.index, dtype=float)
    stat["유형"] = "분류 안 됨"
    if len(feat) >= 8:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        # 극단값이 군집을 독차지하지 않도록 순위로 바꾼 뒤 표준화
        X = StandardScaler().fit_transform(feat.rank(pct=True))
        labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
        stat["군집"] = labels

        # 묶음별 평균 흔들림 순서대로 이름을 붙인다
        shake = stat.groupby("군집")["변동계수"].mean().sort_values()
        tiers = ["가장 꾸준한 상권", "대체로 안정적인 상권",
                 "자주 흔들리는 상권", "크게 출렁이는 상권"]
        names = {c: tiers[i] if i < len(tiers) else f"유형 {i + 1}"
                 for i, c in enumerate(shake.index)}
        stat["유형"] = stat["군집"].map(names)

        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        sim = pd.DataFrame((1 - d / d.max()) * 100,
                           index=stat.index, columns=stat.index)

    out = stat.reset_index().sort_values("리스크점수", ascending=False)
    return m, out, low_m, high_m, sim


# ──────────────────────────────── 예측
def forecast(series: np.ndarray, months: int = 3):
    y = series[-12:] if len(series) >= 12 else series
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    sigma = (y - (slope * x + intercept)).std(ddof=1) if len(y) > 2 else 0.0
    fx = np.arange(len(y), len(y) + months)
    center = np.clip(slope * fx + intercept, 0.01, None)
    return center, np.clip(center - 1.96 * sigma, 0.01, None), center + 1.96 * sigma, slope


def flags(row, slope):
    """한 줄 진단과 제안 카드가 같은 기준을 쓰도록 신호를 한곳에서 판정한다."""
    return {
        "정책": row["정책점수"] >= 60,
        "집중": row["집중점수"] >= 60,
        "하락": row["추세점수"] >= 60 or row["하락개월"] >= 4 or row["최근추세"] < -5,
        "상승": slope > 0.005 and row["최근추세"] > 3,
    }


def headline(row, grade, sig):
    """한 줄 진단 문장."""
    body = {"안심": "지역화폐 매출이 꾸준한 편입니다",
            "주의": "지역화폐 매출이 때때로 흔들리는 편입니다",
            "조심": "지역화폐 매출이 크게 흔들리는 편입니다"}[grade]
    if sig["하락"]:
        return body + " 그리고 최근 몇 달 줄어드는 중입니다"
    if sig["정책"]:
        return body + " 특히 전체 결제가 적은 달에 많이 줄어듭니다"
    if sig["집중"]:
        return body + " 손님층이 한쪽에 몰려 있습니다"
    if sig["상승"]:
        return body + " 최근에는 올라가는 중입니다"
    return body


@st.cache_data(show_spinner=False)
def backtest(shr_values: tuple, horizon: int = 3) -> float:
    """과거 구간에서 3개월 선행 예측을 반복해 평균 절대 오차율(MAPE)을 구한다."""
    errs = []
    for series in shr_values:
        y = np.array(series)
        if len(y) < 12:
            continue
        for cut in range(9, len(y) - horizon + 1):
            train = y[max(0, cut - 12):cut]
            x = np.arange(len(train))
            a, b = np.polyfit(x, train, 1)
            fx = np.arange(len(train), len(train) + horizon)
            pred = np.clip(a * fx + b, 0.01, None)
            actual = y[cut:cut + horizon]
            errs.extend(np.abs(pred - actual) / actual)
    return float(np.median(errs) * 100) if errs else float("nan")


def ai_summary(payload: dict) -> str:
    """Gemini에 수치를 넘겨 사장님께 드리는 말로 바꿔 받는다."""
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        return ""
    lines = "\n".join(f"- {k}: {v}" for k, v in payload.items())
    prompt = (
        "너는 공공데이터로 상권을 분석해 소상공인에게 알려주는 도우미다.\n"
        "아래는 한 동네의 지역사랑상품권 결제 분석 결과다. 이 수치만 가지고 "
        "가게 사장님께 말하듯 세 문단으로 정리해라.\n\n"
        f"{lines}\n\n"
        "지표의 뜻 (반드시 이대로 이해할 것):\n"
        "- 소비 몫: 대전 전체 지역화폐 결제액 중 이 동네가 차지한 비율. "
        "가게 수나 매출액이 아니라 결제 금액의 비율이다.\n"
        "- 저점기 잔존율: 대전 전체 결제가 가장 적었던 달의 이 동네 결제액이, "
        "가장 많았던 달의 몇 퍼센트였는지. 살아남은 가게 비율이 절대 아니다.\n"
        "- 흔들림 순위: 달마다 소비 몫이 얼마나 출렁였는지의 순위. 1위가 가장 심하다.\n"
        "- 상권 유형: 비슷한 성격끼리 묶은 분류 이름일 뿐, 등급이 아니다.\n\n"
        "규칙:\n"
        "- 주어진 수치 밖의 사실을 만들지 마라. 특히 원인을 단정하지 마라.\n"
        "- 지표를 매출액이나 가게 수로 바꿔 말하지 마라.\n"
        "- '예산 때문에', '정책 탓에' 같은 인과는 쓰지 마라. 자료로 확인되지 않았다.\n"
        "- 전문용어(변동계수, 점유율, 허핀달) 대신 쉬운 말로 풀어 써라.\n"
        "- 과장하거나 겁주지 말고, 담담하게 사실과 할 일을 말해라.\n"
        "- 각 문단은 세 문장 이내. 머리말이나 맺음말 없이 본문만.\n")
    # gemini-2.5-flash는 신규 키에서 막혀서 최신 별칭을 쓰고, 혼잡하면 다음 모델로 넘어간다
    last_err = ""
    for model in ("gemini-flash-lite-latest", "gemini-flash-latest"):
        try:
            res = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1500}},
                timeout=40)
            res.raise_for_status()
            parts = res.json()["candidates"][0]["content"]["parts"]
            text = "".join(pt.get("text", "") for pt in parts
                           if not pt.get("thought")).strip()
            if text:
                return text
            last_err = "빈 응답"
        except Exception as e:
            last_err = str(e)
    return f"__ERROR__{last_err}"


def advise(row, sig):
    """판정된 신호에 따라 할 일을 제안한다."""
    cards = []
    if sig["정책"]:
        cards.append((
            "결제가 적은 달에 크게 줄어요",
            f"대전 전체 지역화폐 결제가 가장 적었던 달, 이 동네 결제액은 "
            f"가장 많았던 달의 {row['잔존율']:.0f}% 수준이었습니다.",
            "지역화폐 결제가 뜸한 시기를 대비해 운영비 여유를 두세요. "
            "할인율이나 발행 한도 공지가 뜨면 미리 챙겨 보시고요."))
    if sig["집중"]:
        cards.append((
            "손님층이 한쪽에 몰려 있어요",
            f"결제의 {row['최대고객층비중']:.0f}%가 {row['최대고객층']} 손님입니다.",
            "이 손님들이 줄면 타격이 큽니다. 다른 연령대가 찾을 만한 메뉴나 "
            "시간대를 하나쯤 만들어 두면 완충이 됩니다."))
    if sig["하락"]:
        cards.append((
            "최근 줄어드는 중이에요",
            f"최근 석 달이 직전 석 달보다 {abs(row['최근추세']):.1f}% 낮고, "
            f"6개월 중 {int(row['하락개월'])}개월이 전달보다 낮았습니다.",
            "가까운 동네로 손님이 옮겨가고 있는지 확인해 보세요. "
            "'닮은 동네' 탭에서 비슷한 상권의 흐름과 견줘 볼 수 있습니다."))
    if sig["상승"] and len(cards) <= 1:
        cards.append((
            "최근에는 올라가는 중",
            f"최근 석 달이 직전 석 달보다 {row['최근추세']:.1f}% 높습니다.",
            "늘어난 방문을 단골로 묶을 방법을 지금 준비해 두면 좋습니다."))
    if not cards:
        cards.append((
            "지금은 별다른 신호가 없어요",
            "세 가지 지표 모두 대전 평균 범위 안에 있습니다.",
            "분기에 한 번 정도 이 화면을 열어 흐름만 확인하시면 충분합니다."))
    return cards[:2]


# ──────────────────────────────── 화면
st.title("우리 동네 지역화폐 리스크 진단")

with st.sidebar:
    st.header("설정")
    wide = st.checkbox("넓은 화면으로 보기", value=False,
                       help="PC나 태블릿에서 보실 때 켜면 가로로 넓게 배치됩니다")
    mobile = not wide
    with st.expander("분석 조건 (전문가용)"):
        years = st.slider("분석 기간 (연월)", 2023, 2026, (2024, 2026))
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

m, stat, low_m, high_m, sim = build(df, min_share)


def fmt_ym(ym):
    return f"{ym[:4]}년 {int(ym[4:])}월"


span = sorted(m["crtr_ym"].unique())
all_dong = df[df["동"] != "기타"].groupby(["지역", "동"]).ngroups
st.caption(f"한국조폐공사 지역사랑상품권 결제정보 · 대전광역시 · "
           f"{fmt_ym(span[0])} ~ {fmt_ym(span[-1])} 기준 · "
           f"자료가 있는 {all_dong}개 동네 중 모든 달 기록이 있고 "
           f"평균 비중 {min_share}% 이상인 {len(stat)}개를 분석했습니다")

view_mode = st.radio("보기", ["사장님용", "분석용"], horizontal=True,
                     label_visibility="collapsed")
st.write("")

c1, c2 = (st.container(), st.container()) if mobile else st.columns(2)
if mobile:
    st.caption("자치구와 행정동을 차례로 골라 주세요")
gu_list = list(REGIONS.values())
gu = c1.selectbox("자치구", gu_list,
                  index=gu_list.index(st.session_state.get("gu", "유성구")))
pool = stat[stat["지역"] == gu].sort_values("평균점유율", ascending=False)
dong_list = pool["동"].tolist()
_d = st.session_state.get("dong")
dong = c2.selectbox("행정동", dong_list,
                    index=dong_list.index(_d) if _d in dong_list else 0)

_sel = stat[(stat["지역"] == gu) & (stat["동"] == dong)]
if _sel.empty:
    st.warning("선택한 동네는 분석 조건에 맞는 자료가 없습니다. 다른 동네를 골라 주세요.")
    st.stop()
row = _sel.iloc[0]
score = row["리스크점수"]
grade = "조심" if score >= 66 else ("주의" if score >= 33 else "안심")
color = {"조심": "red", "주의": "orange", "안심": "green"}[grade]

d = m[(m["지역"] == gu) & (m["동"] == dong)].sort_values("crtr_ym")
share = d["점유율"].to_numpy()
months = d["crtr_ym"].tolist()
recent = share[-3:].mean()
center, lowb, highb, slope = forecast(share, 3)

sig = flags(row, slope)

# ── 머리 카드 (직접 만든 HTML)
rank = int(stat["리스크점수"].rank(method="min", ascending=False)
           [(stat["지역"] == gu) & (stat["동"] == dong)].iloc[0])
trend_word = ("올라가는 중" if slope > 0.005
              else ("내려가는 중" if slope < -0.005 else "비슷한 수준"))
accent = {"조심": "#c0392b", "주의": "#c77700", "안심": "#157f5a"}[grade]
soft = {"조심": "#fdeceb", "주의": "#fdf3e3", "안심": "#e9f6f0"}[grade]

card = f"""
<div style="font-family:'Noto Sans KR',system-ui,sans-serif;color:#171717;
            border:1px solid #e6e6e6;border-radius:18px;padding:28px 30px;
            background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.04)">
  <div style="font-size:13px;color:#8a8a8a;letter-spacing:.02em">
    대전광역시 · {gu} · {fmt_ym(span[0])}~{fmt_ym(span[-1])}
  </div>
  <div style="display:flex;align-items:baseline;gap:12px;margin:6px 0 18px">
    <span style="font-size:30px;font-weight:700;letter-spacing:-.03em">{dong}</span>
    <span style="background:{soft};color:{accent};font-size:14px;font-weight:700;
                 padding:5px 14px;border-radius:999px">{grade}</span>
  </div>
  <div style="font-size:19px;line-height:1.55;font-weight:500;margin-bottom:22px">
    {headline(row, grade, sig)}
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
              gap:16px;border-top:1px solid #f0f0f0;padding-top:18px">
    <div>
      <div style="font-size:12px;color:#8a8a8a;margin-bottom:4px">우리 동네 몫</div>
      <div style="font-size:24px;font-weight:700">{recent:.2f}<span
           style="font-size:14px;font-weight:500">%</span></div>
      <div style="font-size:12px;color:#8a8a8a">최근 석 달</div>
    </div>
    <div>
      <div style="font-size:12px;color:#8a8a8a;margin-bottom:4px">흔들림 순위</div>
      <div style="font-size:24px;font-weight:700">{rank}<span
           style="font-size:14px;font-weight:500">위</span></div>
      <div style="font-size:12px;color:#8a8a8a">{len(stat)}개 동네 중</div>
    </div>
    <div>
      <div style="font-size:12px;color:#8a8a8a;margin-bottom:4px">저점기 잔존율</div>
      <div style="font-size:24px;font-weight:700">{row['잔존율']:.0f}<span
           style="font-size:14px;font-weight:500">%</span></div>
      <div style="font-size:12px;color:#8a8a8a">한산한 달에 남는 결제</div>
    </div>
    <div>
      <div style="font-size:12px;color:#8a8a8a;margin-bottom:4px">주요 손님층</div>
      <div style="font-size:24px;font-weight:700">{row['최대고객층']}</div>
      <div style="font-size:12px;color:#8a8a8a">결제의 {row['최대고객층비중']:.0f}%</div>
    </div>
  </div>
  <div style="margin-top:20px;padding:14px 16px;background:#fafafa;
              border-radius:12px;font-size:14px;line-height:1.6;color:#3f3f3f">
    이 기간 평균은 <b>{row['평균점유율']:.2f}%</b>였고 최근 석 달은
    <b>{recent:.2f}%</b>로 {trend_word}입니다.
    흐름이 이어진다면 석 달 뒤에는 <b>{center[-1]:.2f}%</b> 언저리가 됩니다.
  </div>
</div>
"""
components.html(card, height=620 if mobile else 360, scrolling=False)

st.write("")
st.write("")
st.markdown("#### 우리 동네 지역화폐 소비는 이렇게 움직였습니다")


def next_ym(ym, k):
    t = int(ym[:4]) * 12 + int(ym[4:]) - 1 + k
    return f"{t // 12:04d}{t % 12 + 1:02d}"


fut = [next_ym(months[-1], i) for i in range(1, 4)]
pad = [np.nan] * (len(share) - 1) + [share[-1]]
front = pd.DataFrame(index=months + fut)
front["지금까지"] = list(share) + [np.nan] * 3
front["앞으로 예상"] = pad + list(center)
st.line_chart(front, height=300, color=[GREEN, "#bdbdbd"])
mape = backtest(tuple(tuple(v) for v in
                      m.pivot_table(index=["지역", "동"], columns="crtr_ym",
                                    values="점유율").dropna().to_numpy()))
st.caption("금액이 아니라 대전 전체에서 차지하는 몫입니다. "
           "금액은 전 지역이 함께 오르내리기 때문입니다. "
           + (f"예상은 지난 흐름을 그대로 이어 본 값입니다. 같은 방식으로 과거를 "
              f"되짚어 보면 석 달 뒤 값이 실제와 중앙값 기준 {mape:.0f}% 어긋났습니다."
              if mape == mape else
              "예상은 지난 흐름을 그대로 이어 본 단순 추정이라 실제와 다를 수 있습니다."))

st.write("")
st.markdown("#### 이웃과 견주면")
gu_avg = stat[stat["지역"] == gu]["평균점유율"].mean()
cmp = pd.DataFrame({
    "구분": [f"{dong}", f"{gu} 평균", "대전 평균"],
    "지역화폐 소비 몫": [f"{row['평균점유율']:.2f}%", f"{gu_avg:.2f}%",
                  f"{stat['평균점유율'].mean():.2f}%"],
    "저점기 잔존율": [f"{row['잔존율']:.1f}%",
                      f"{stat[stat['지역'] == gu]['잔존율'].mean():.1f}%",
                      f"{stat['잔존율'].mean():.1f}%"],
    "흔들림 점수": [f"{score:.0f}점",
               f"{stat[stat['지역'] == gu]['리스크점수'].mean():.0f}점",
               f"{stat['리스크점수'].mean():.0f}점"]})
st.dataframe(cmp, hide_index=True, use_container_width=True)

st.write("")
st.markdown("#### 지금 하면 좋을 일")
for title, fact, action in advise(row, sig):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(fact)
        st.caption(action)

if st.secrets.get("GEMINI_API_KEY", ""):
    st.write("")
    if st.button("AI가 풀어서 설명해 주기", use_container_width=True):
        near_txt = "없음"
        if not sim.empty and (gu, dong) in sim.index:
            nb = sim.loc[(gu, dong)].drop(index=(gu, dong)).sort_values(
                ascending=False).head(3)
            info2 = stat.set_index(["지역", "동"])
            near_txt = ", ".join(
                f"{a} {b}(최근 {info2.loc[(a, b), '최근추세']:+.0f}%)"
                for a, b in nb.index)
        dd = d.copy()
        dd["월"] = dd["crtr_ym"].str[4:].astype(int)
        sidx = dd.groupby("월")["점유율"].mean()
        sidx = (sidx / sidx.mean() * 100).round(0)
        with st.spinner("문장을 쓰는 중입니다. 10초쯤 걸립니다…"):
            text = ai_summary({
                "동네": f"{gu} {dong}",
                "상권 유형": row["유형"],
                "대전 전체 중 소비 몫": f"{recent:.2f}% (기간 평균 {row['평균점유율']:.2f}%)",
                "흔들림 순위": f"{len(stat)}곳 중 {rank}위",
                "저점기 잔존율": f"{row['잔존율']:.1f}% (대전 평균 {stat['잔존율'].mean():.1f}%)",
                "주요 손님층": f"{row['최대고객층']} {row['최대고객층비중']:.0f}%",
                "최근 3개월 추세": f"{row['최근추세']:+.1f}%",
                "석 달 뒤 예상": f"{center[-1]:.2f}%",
                "가장 활발한 달": f"{sidx.idxmax()}월 (지수 {sidx.max():.0f})",
                "가장 한산한 달": f"{sidx.idxmin()}월 (지수 {sidx.min():.0f})",
                "닮은 동네 최근 흐름": near_txt})
        if not text:
            st.info("AI 설명 기능이 설정되어 있지 않습니다.")
        elif text.startswith("__ERROR__"):
            st.warning("지금은 설명을 받아오지 못했습니다. 잠시 뒤 다시 눌러 보세요.")
        else:
            with st.container(border=True):
                st.markdown(text)
                st.caption("위 수치를 바탕으로 AI가 쓴 문장입니다. "
                           "숫자는 분석 결과이고, 문장 표현만 AI가 맡았습니다.")

st.write("")
st.markdown("##### 더 자세히 보기")

if view_mode == "사장님용":
    names = ["닮은 동네", "우리 동네 위치", "누가 쓰나", "달마다 얼마나"]
else:
    names = ["닮은 동네", "우리 동네 위치", "누가 쓰나", "달마다 얼마나",
             "다른 동네와 비교", "분석 방법", "이 진단의 한계"]
tabs = st.tabs(names)
tw, t0, t1, t2 = tabs[0], tabs[1], tabs[2], tabs[3]
t3 = tabs[4] if len(tabs) > 4 else st.empty()
t4 = tabs[5] if len(tabs) > 5 else st.empty()
t5 = tabs[6] if len(tabs) > 6 else st.empty()

with tw:
    key = (gu, dong)
    if sim.empty or key not in sim.index:
        st.info("닮은 동네를 찾으려면 분석 대상이 여덟 곳 이상이어야 합니다. "
                "왼쪽 설정에서 최소 비중을 낮춰 보세요.")
    else:
        st.caption("소비 규모, 흔들림, 저점기 잔존율, 계절 흐름 네 가지가 닮은 동네를 "
                   "찾았습니다. 수치가 가까울수록 닮음 정도가 높습니다.")
        near = sim.loc[key].drop(index=key).sort_values(ascending=False).head(3)
        info = stat.set_index(["지역", "동"])
        for (sgu, sdong), pct in near.items():
            r2 = info.loc[(sgu, sdong)]
            move = ("올라가는 중" if r2["최근추세"] > 3
                    else ("내려가는 중" if r2["최근추세"] < -3 else "비슷한 수준"))
            with st.container(border=True):
                c_a, c_b = st.columns([3, 1])
                c_a.markdown(f"**{sgu} {sdong}**")
                c_a.caption(f"소비 몫 {r2['평균점유율']:.2f}% · 잔존율 "
                            f"{r2['잔존율']:.1f}% · 최근 {move}")
                c_b.markdown(f"### {pct:.0f}%")
                c_b.caption("닮음")

        ups = sum(1 for k2 in near.index if info.loc[k2, "최근추세"] > 3)
        me_up = row["최근추세"] > 3
        if ups >= 2 and not me_up:
            st.warning(f"닮은 세 곳 중 {ups}곳은 최근 올라가는 중인데 "
                       f"{dong}만 그렇지 않습니다. 동네 사정이 아니라 "
                       "우리 상권만의 변화일 수 있습니다.")
        elif ups == 0 and not me_up:
            st.info("닮은 동네들도 함께 내려가는 중입니다. "
                    "우리 가게만의 문제가 아닐 수 있습니다.")

        st.write("")
        st.markdown("##### 대전 상권을 네 유형으로 묶으면")
        st.caption(f"{dong}은 **{row['유형']}** 유형입니다.")
        tp = (stat.groupby("유형")
              .agg(동네수=("동", "count"), 평균소비몫=("평균점유율", "mean"),
                   평균흔들림=("변동계수", "mean"))
              .sort_values("평균흔들림").round(2).reset_index())
        st.dataframe(tp, hide_index=True, use_container_width=True)
        st.caption("흔들림이 적은 순서입니다. 네 가지 수치(소비 규모 · 흔들림 · "
                   "저점기 잔존율 · 계절 흐름)로 묶은 뒤, 묶음별 평균 흔들림에 따라 "
                   "이름을 붙였습니다.")

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
                fm = folium.Map(location=[36.345, 127.385], zoom_start=12,
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
                fm.fit_bounds([[mp["lat"].min(), mp["lon"].min()],
                               [mp["lat"].max(), mp["lon"].max()]])
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
    st.bar_chart((pv / pv.values.sum() * 100).round(1), height=300)
    st.caption(f"가장 큰 손님층은 {row['최대고객층']}으로 전체의 "
               f"{row['최대고객층비중']:.0f}%입니다.")

    if len(months) >= 24:
        st.markdown("##### 손님층이 어떻게 바뀌었나")
        prev = df[(df["지역"] == gu) & (df["동"] == dong) &
                  (df["crtr_ym"].isin(months[-24:-12]))]
        def mix_of(frame):
            t = frame.groupby("연령대")["stlm_amt"].sum()
            return t / t.sum() * 100
        now_mix, old_mix = mix_of(sub), mix_of(prev)
        chg = pd.DataFrame({"지지난해": old_mix, "지난해": now_mix}).fillna(0)
        chg["변화"] = (chg["지난해"] - chg["지지난해"]).round(1)
        chg = chg.round(1).sort_index()
        st.dataframe(chg, use_container_width=True)
        up = chg["변화"].idxmax()
        down = chg["변화"].idxmin()
        st.caption(f"{up} 손님 비중이 가장 많이 늘었고({chg.loc[up, '변화']:+.1f}%p), "
                   f"{down} 손님이 가장 많이 줄었습니다({chg.loc[down, '변화']:+.1f}%p). "
                   "손님층이 바뀌면 잘 팔리는 메뉴와 시간대도 달라집니다.")

with t2:
    st.caption("달마다 이 동네에서 지역화폐로 결제된 금액입니다. "
               "전체가 함께 오르내리는 것은 상권이 아니라 발행 규모 때문입니다.")
    st.bar_chart((d.set_index("crtr_ym")["결제금액"] / 1e8).rename("억 원"), height=300)

    st.markdown("##### 해마다 되풀이되는 달별 패턴")
    dd = d.copy()
    dd["월"] = dd["crtr_ym"].str[4:].astype(int)
    years_in = dd["crtr_ym"].str[:4].nunique()
    if years_in >= 2:
        season = dd.groupby("월")["점유율"].mean()
        season = (season / season.mean() * 100).round(0)
        season.index = [f"{i}월" for i in season.index]
        st.bar_chart(season.rename("평달을 100으로 봤을 때"), height=280,
                     color=GREEN)
        hi, lo = season.idxmax(), season.idxmin()
        st.caption(f"{hi}이 가장 활발하고({season.max():.0f}) {lo}이 가장 한산합니다"
                   f"({season.min():.0f}). 같은 달끼리 묶어 평균을 낸 값이라 "
                   "해마다 되풀이되는 흐름을 봅니다. 재고와 인력 계획에 참고하세요.")
    else:
        st.caption("달별 패턴은 두 해 이상 자료가 있어야 볼 수 있습니다. "
                   "왼쪽 설정에서 기간을 늘려 보세요.")

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
            (g1, "저점기 잔존율", row["정책점수"], f"결제 저점기 잔존율 {row['잔존율']:.1f}%"),
            (g2, "손님층 집중", row["집중점수"],
             f"{row['최대고객층']} {row['최대고객층비중']:.0f}%"),
            (g3, "하락 지속", row["추세점수"],
             f"6개월 중 {int(row['하락개월'])}개월 하락")]:
        col.markdown(f"**{label}** {val:.0f}점")
        col.progress(min(int(val), 100))
        col.caption(note)
    st.caption(f"변동계수 {row['변동계수']:.3f} · 기간 평균 비중 {row['평균점유율']:.2f}% · "
               f"종합 {score:.0f}점 (잔존율 40% + 집중 30% + 하락 30%, 탐색적 가중치)")

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
| 저점기 잔존율 | 40% | 대전 전체 결제가 가장 적었던 달의 결제액이, 가장 많았던 달의 몇 %인지. 낮을수록 전체 흐름에 크게 휩쓸린다. 저점의 원인이 정책인지는 이 지표만으로 확인되지 않는다 |
| 고객층 집중 | 30% | 성별 × 연령대 결제액의 허핀달 지수. 한 집단에 쏠릴수록 그 집단의 소비가 줄 때 타격이 크다 |
| 하락 지속 | 30% | 최근 6개월 중 전월보다 낮았던 달 수와 3개월 추세 |

**상권 유형 분류**

소비 규모 · 흔들림 · 저점기 잔존율 · 계절 진폭 네 값을 순위로 바꿔 표준화한 뒤
k-means 군집 분석으로 네 유형으로 묶었습니다. 순위로 바꾸는 이유는 한두 곳의
극단값이 군집을 독차지하는 것을 막기 위해서입니다. 유형 이름은 묶음별 평균 흔들림 순서를
보고 붙인 것으로, 알고리즘이 지은 것이 아닙니다. '닮은 동네'의 닮음 정도는 같은
네 값 사이의 거리를 0~100으로 환산한 값입니다.

세 지표를 분석 대상 동네 안에서 순위로 환산해 0~100점으로 만들고 가중 합산합니다. 가중치는 검증된 값이 아니라 탐색적으로 정한 설계값입니다.

**무엇을 보여주지 못하나**

- 상권 전체 매출이 아닙니다. 카드 · 현금 · 간편결제는 포함되지 않습니다.
- 개별 점포가 아니라 행정동 단위입니다.
- 폐업 위험을 예측하지 않습니다. 대전 일반음식점 63,889건(별도 분석)의 폐업률과 대조한 결과
  유의미한 상관이 없었습니다(r=0.06, p=0.65). 시차를 둔 비교와 개업 대비 폐업 배율로도
  마찬가지였습니다. 따라서 **매출이 흔들릴 가능성**을 보는 지표로 한정합니다.

**참고 — 2025년에 실제로 있었던 일**

대전 전체 지역화폐 결제액은 2025년 6월 약 14억 원에서 11월 약 889억 원으로 60배 넘게 늘었습니다.
상권이 좋아진 것이 아니라 발행 예산이 복원된 결과입니다. 금액이 아니라 비중을 보는 이유입니다.
""")

st.divider()
st.caption("출처 : 공공데이터포털 한국조폐공사 지역사랑상품권 결제정보, "
           "행정안전부 전국일반음식점표준데이터")
