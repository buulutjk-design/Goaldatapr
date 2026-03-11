import requests
import time
import math
import random
import os
from datetime import datetime, date, timezone, timedelta
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8680855478:AAGJFbu6B_Q7AZNMl5-3BDO9GWBAPiNYna4")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "8480843841"))
API_KEY = os.environ.get("API_KEY", "0c0c1ad20573b309924dd3d7b1bc3e62")
API_URL = "https://v3.football.api-sports.io"

CONFIDENCE_THRESHOLD = 72
CACHE_TIME = 43200
analysis_cache = {}
shown_banko = set()
bot_active = True

PRIORITY_LEAGUES = {
    304: "Singapore Premier League",
    78: "Bundesliga",
    79: "2. Bundesliga",
    80: "3. Liga",
    103: "Eliteserien Norway",
    104: "1. Division Norway",
    119: "Superliga Denmark",
    120: "1. Division Denmark",
    88: "Eredivisie",
    89: "Eerste Divisie",
    218: "Bundesliga Austria",
    113: "Allsvenskan",
    114: "Superettan",
    244: "Veikkausliiga",
    245: "Ykkonen",
}

ALL_LEAGUES = {
    304: "Singapore Premier League",
    78: "Bundesliga",
    79: "2. Bundesliga",
    80: "3. Liga",
    81: "DFB Pokal",
    103: "Eliteserien Norway",
    104: "1. Division Norway",
    119: "Superliga Denmark",
    120: "1. Division Denmark",
    88: "Eredivisie",
    89: "Eerste Divisie",
    218: "Bundesliga Austria",
    219: "2. Liga Austria",
    113: "Allsvenskan",
    114: "Superettan",
    244: "Veikkausliiga",
    245: "Ykkonen",
    39: "Premier League",
    40: "Championship",
    41: "League One",
    42: "League Two",
    45: "FA Cup",
    140: "La Liga",
    141: "Segunda Division",
    142: "Copa del Rey",
    135: "Serie A",
    136: "Serie B",
    61: "Ligue 1",
    62: "Ligue 2",
    203: "Super Lig",
    204: "1. Lig Turkey",
    205: "2. Lig Turkey",
    94: "Primeira Liga",
    95: "Segunda Liga",
    144: "Jupiler Pro League",
    179: "Scottish Premiership",
    180: "Scottish Championship",
    235: "Premier League Russia",
    236: "FNL Russia",
    207: "Super League Switzerland",
    197: "Super League Greece",
    106: "Ekstraklasa",
    283: "Liga 1 Romania",
    210: "HNL Croatia",
    286: "Super Liga Serbia",
    333: "Premier League Ukraine",
    271: "OTP Bank Liga Hungary",
    345: "Czech Liga",
    172: "First League Bulgaria",
    357: "League of Ireland",
    98: "J1 League",
    99: "J2 League",
    292: "K League 1",
    293: "K League 2",
    169: "Chinese Super League",
    296: "Thai League 1",
    313: "Liga 1 Indonesia",
    188: "A-League Australia",
    233: "Egyptian Premier League",
    307: "Saudi Pro League",
    435: "UAE Pro League",
    253: "MLS",
    262: "Liga MX",
    71: "Serie A Brazil",
    72: "Serie B Brazil",
    128: "Liga Profesional Argentina",
    265: "Primera Division Chile",
    239: "Liga BetPlay Colombia",
    2: "Champions League",
    3: "Europa League",
    848: "Conference League",
    11: "Copa Libertadores",
    1: "World Cup",
    4: "Euro Championship",
    16: "UEFA Nations League",
}


def safe_request(url):
    headers = {"x-apisports-key": API_KEY}
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1)
    return None


def poisson_prob(lam, k):
    if k > 15:
        return 0.0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)


def weighted_avg(values, decay=0.88):
    if not values:
        return 0.0
    total_w = 0.0
    ws = 0.0
    w = 1.0
    for v in reversed(values):
        ws += v * w
        total_w += w
        w *= decay
    return ws / total_w if total_w > 0 else 0.0


def over_prob_poisson(lam_total, threshold):
    k_max = math.floor(threshold)
    p_under = sum(poisson_prob(lam_total, k) for k in range(k_max + 1))
    return max(0.0, min(1.0, 1 - p_under))


def dixon_coles_tau(h, a, lh, la, rho=-0.1):
    if h == 0 and a == 0:
        return 1 - lh * la * rho
    elif h == 0 and a == 1:
        return 1 + lh * rho
    elif h == 1 and a == 0:
        return 1 + la * rho
    elif h == 1 and a == 1:
        return 1 - rho
    return 1.0


def over_prob_dc(lh, la, threshold):
    p_under = 0.0
    k_max = math.floor(threshold)
    for h in range(k_max + 2):
        for a in range(k_max + 2):
            if h + a > k_max:
                continue
            p = poisson_prob(lh, h) * poisson_prob(la, a) * dixon_coles_tau(h, a, lh, la)
            p_under += p
    return max(0.0, min(1.0, 1 - p_under))


def monte_carlo(lh, la, sims=10000):
    def sample(lam):
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1
    o15 = 0
    o25 = 0
    for _ in range(sims):
        t = sample(lh) + sample(la)
        if t > 1:
            o15 += 1
        if t > 2:
            o25 += 1
    return o15 / sims, o25 / sims


def get_last6(team_id):
    r = safe_request(API_URL + "/fixtures?team=" + str(team_id) + "&last=6&status=FT")
    if not r:
        return []
    matches = []
    for m in r.get("response", []):
        gh = m["goals"]["home"]
        ga = m["goals"]["away"]
        if gh is None or ga is None:
            continue
        is_home = m["teams"]["home"]["id"] == team_id
        ht_h = m.get("score", {}).get("halftime", {}).get("home") or 0
        ht_a = m.get("score", {}).get("halftime", {}).get("away") or 0
        matches.append({
            "scored": gh if is_home else ga,
            "conceded": ga if is_home else gh,
            "gh": gh,
            "ga": ga,
            "is_home": is_home,
            "ht_scored": ht_h if is_home else ht_a,
            "ht_conceded": ht_a if is_home else ht_h,
            "sh_scored": (gh - ht_h) if is_home else (ga - ht_a),
            "sh_conceded": (ga - ht_a) if is_home else (gh - ht_h),
        })
    return matches


def get_venue6(team_id, venue="home"):
    r = safe_request(API_URL + "/fixtures?team=" + str(team_id) + "&last=20&status=FT")
    if not r:
        return []
    matches = []
    for m in r.get("response", []):
        is_home = m["teams"]["home"]["id"] == team_id
        if venue == "home" and not is_home:
            continue
        if venue == "away" and is_home:
            continue
        gh = m["goals"]["home"]
        ga = m["goals"]["away"]
        if gh is None or ga is None:
            continue
        ht_h = m.get("score", {}).get("halftime", {}).get("home") or 0
        ht_a = m.get("score", {}).get("halftime", {}).get("away") or 0
        matches.append({
            "scored": gh if is_home else ga,
            "conceded": ga if is_home else gh,
            "ht_scored": ht_h if is_home else ht_a,
            "ht_conceded": ht_a if is_home else ht_h,
            "sh_scored": (gh - ht_h) if is_home else (ga - ht_a),
            "sh_conceded": (ga - ht_a) if is_home else (gh - ht_h),
        })
        if len(matches) >= 6:
            break
    return matches


def get_xg(team_id, last=6):
    r = safe_request(API_URL + "/fixtures?team=" + str(team_id) + "&last=" + str(last) + "&status=FT")
    if not r:
        return 0.0
    fixture_ids = [m["fixture"]["id"] for m in r.get("response", [])]
    xg_vals = []
    for fid in fixture_ids[:4]:
        rs = safe_request(API_URL + "/fixtures/statistics?fixture=" + str(fid) + "&team=" + str(team_id))
        if not rs or not rs.get("response"):
            continue
        for team_stat in rs["response"]:
            if team_stat["team"]["id"] != team_id:
                continue
            st = {s["type"]: s["value"] for s in team_stat.get("statistics", [])}
            xg = st.get("expected_goals") or st.get("xG") or st.get("Expected Goals")
            if xg:
                try:
                    xg_vals.append(float(str(xg)))
                except Exception:
                    pass
        time.sleep(0.1)
    return round(sum(xg_vals) / len(xg_vals), 2) if xg_vals else 0.0


def get_h2h(id1, id2):
    r = safe_request(API_URL + "/fixtures/headtohead?h2h=" + str(id1) + "-" + str(id2) + "&last=8")
    if not r:
        return []
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    matches = []
    for m in r.get("response", []):
        gh = m["goals"]["home"]
        ga = m["goals"]["away"]
        if gh is None or ga is None:
            continue
        try:
            md = datetime.fromisoformat(m["fixture"]["date"].replace("Z", "+00:00"))
            if md >= cutoff:
                matches.append({"total": gh + ga})
        except Exception:
            pass
    return matches


def calc_lambda(gen6, venue6):
    gen_s = weighted_avg([m["scored"] for m in gen6])
    gen_c = weighted_avg([m["conceded"] for m in gen6])
    ven_s = weighted_avg([m["scored"] for m in venue6]) if venue6 else gen_s
    ven_c = weighted_avg([m["conceded"] for m in venue6]) if venue6 else gen_c
    scored = ven_s * 0.65 + gen_s * 0.35
    conceded = ven_c * 0.65 + gen_c * 0.35
    return scored, conceded


def analyze(id1, name1, id2, name2):
    key = str(id1) + "_" + str(id2)
    now = datetime.now().timestamp()
    if key in analysis_cache:
        if now - analysis_cache[key]["time"] < CACHE_TIME:
            return analysis_cache[key]["data"]

    home_gen = get_last6(id1)
    away_gen = get_last6(id2)
    home_venue = get_venue6(id1, "home")
    away_venue = get_venue6(id2, "away")

    if len(home_gen) < 3 or len(away_gen) < 3:
        return None

    home_xg = get_xg(id1)
    away_xg = get_xg(id2)

    h_s, h_c = calc_lambda(home_gen, home_venue)
    a_s, a_c = calc_lambda(away_gen, away_venue)

    lh = ((h_s + a_c) / 2) * 1.06
    la = (a_s + h_c) / 2

    if home_xg > 0:
        lh = lh * 0.75 + home_xg * 0.25
    if away_xg > 0:
        la = la * 0.75 + away_xg * 0.25

    lh = max(0.2, min(lh, 5.0))
    la = max(0.2, min(la, 5.0))
    lam = lh + la

    o15_p = over_prob_poisson(lam, 1.5)
    o15_dc = over_prob_dc(lh, la, 1.5)
    mc15, mc25 = monte_carlo(lh, la, 10000)
    o15 = o15_p * 0.35 + o15_dc * 0.35 + mc15 * 0.30

    o25_p = over_prob_poisson(lam, 2.5)
    o25_dc = over_prob_dc(lh, la, 2.5)
    o25 = o25_p * 0.35 + o25_dc * 0.35 + mc25 * 0.30

    h2h = get_h2h(id1, id2)
    if h2h:
        h2h_o15 = sum(1 for m in h2h if m["total"] > 1) / len(h2h)
        h2h_o25 = sum(1 for m in h2h if m["total"] > 2) / len(h2h)
        o15 = o15 * 0.70 + h2h_o15 * 0.30
        o25 = o25 * 0.70 + h2h_o25 * 0.30

    o15 = int(max(5, min(95, o15 * 100)))
    o25 = int(max(5, min(95, o25 * 100)))

    ho15 = sum(1 for m in home_gen if m["gh"] + m["ga"] > 1)
    ao15 = sum(1 for m in away_gen if m["gh"] + m["ga"] > 1)
    ho25 = sum(1 for m in home_gen if m["gh"] + m["ga"] > 2)
    ao25 = sum(1 for m in away_gen if m["gh"] + m["ga"] > 2)

    warns = []
    if len(home_gen) < 4:
        warns.append("low_home")
    if len(away_gen) < 4:
        warns.append("low_away")
    if not h2h:
        warns.append("no_h2h")

    if len(warns) == 0:
        reliability = "Yuksek"
    elif len(warns) == 1:
        reliability = "Orta"
    else:
        reliability = "Dusuk"

    result = {
        "h": name1,
        "a": name2,
        "lh": round(lh, 2),
        "la": round(la, 2),
        "lam": round(lam, 2),
        "o15": o15,
        "u15": 100 - o15,
        "o25": o25,
        "u25": 100 - o25,
        "ho15": ho15,
        "ao15": ao15,
        "ho25": ho25,
        "ao25": ao25,
        "h2h_count": len(h2h),
        "reliability": reliability,
        "warns": warns,
    }
    analysis_cache[key] = {"data": result, "time": now}
    return result


def confidence_label(prob):
    if prob >= 80:
        return "YUKSEK GUVEN - GUVENLI"
    if prob >= 72:
        return "IYI GUVEN - GUVENLI"
    return "ORTA GUVEN"


def format_result(result, league="", kickoff=""):
    o15 = result["o15"]
    o25 = result["o25"]
    u15 = result["u15"]
    u25 = result["u25"]

    league_line = "Lig: " + league + "\n" if league else ""
    ko_line = "Saat: " + kickoff + "\n" if kickoff else ""

    if o15 >= CONFIDENCE_THRESHOLD:
        v15 = "1.5 UST " + str(o15) + "% [ " + confidence_label(o15) + " ]"
    else:
        v15 = "1.5 ALT " + str(u15) + "% [ DUSUK GOL BEKLENTISI ]"

    if o25 >= CONFIDENCE_THRESHOLD:
        v25 = "2.5 UST " + str(o25) + "% [ " + confidence_label(o25) + " ]"
    else:
        v25 = "2.5 ALT " + str(u25) + "% [ DUSUK GOL BEKLENTISI ]"

    h2h_line = "H2H 2025+: " + str(result["h2h_count"]) + " mac\n" if result["h2h_count"] > 0 else "H2H 2025+: Veri yok\n"

    msg = "━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "  MAC ANALIZI BANKO\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += league_line
    msg += "Ev: " + result["h"] + "\n"
    msg += "Dep: " + result["a"] + "\n"
    msg += ko_line + "\n"
    msg += "xG: Ev L" + str(result["lh"]) + "  Dep L" + str(result["la"]) + "  Toplam " + str(result["lam"]) + "\n\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += v15 + "\n"
    msg += "Son 6: Ev " + str(result["ho15"]) + "/6  Dep " + str(result["ao15"]) + "/6\n\n"
    msg += v25 + "\n"
    msg += "Son 6: Ev " + str(result["ho25"]) + "/6  Dep " + str(result["ao25"]) + "/6\n\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += h2h_line
    msg += "Guvenilirlik: " + result["reliability"] + "\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━"
    return msg


def get_todays_fixtures():
    today = date.today().strftime("%Y-%m-%d")
    year = datetime.now().year
    all_fixtures = []
    seen_ids = set()

    league_order = list(PRIORITY_LEAGUES.keys()) + [lid for lid in ALL_LEAGUES if lid not in PRIORITY_LEAGUES]

    for league_id in league_order:
        league_name = ALL_LEAGUES.get(league_id, str(league_id))
        for season in [year, year - 1]:
            url = API_URL + "/fixtures?date=" + today + "&league=" + str(league_id) + "&season=" + str(season)
            r = safe_request(url)
            if not r:
                continue
            matches = r.get("response", [])
            if not matches:
                continue
            for m in matches:
                fid = m["fixture"]["id"]
                if fid in seen_ids:
                    continue
                if m["fixture"]["status"]["short"] not in ["NS", "TBD"]:
                    continue
                seen_ids.add(fid)
                all_fixtures.append({
                    "league": league_name,
                    "home_id": m["teams"]["home"]["id"],
                    "home_name": m["teams"]["home"]["name"],
                    "away_id": m["teams"]["away"]["id"],
                    "away_name": m["teams"]["away"]["name"],
                    "kickoff": m["fixture"]["date"],
                    "fid": fid,
                })
            break
        time.sleep(0.2)
    return all_fixtures


def find_banko():
    global shown_banko
    fixtures = get_todays_fixtures()
    if not fixtures:
        return None, None, None

    unseen = [f for f in fixtures if f["fid"] not in shown_banko]
    if not unseen:
        shown_banko.clear()
        unseen = fixtures

    for fix in unseen:
        shown_banko.add(fix["fid"])
        result = analyze(fix["home_id"], fix["home_name"], fix["away_id"], fix["away_name"])
        if not result:
            continue
        if result["reliability"] == "Dusuk":
            continue
        if result["o15"] >= CONFIDENCE_THRESHOLD or result["o25"] >= CONFIDENCE_THRESHOLD:
            try:
                ko = datetime.fromisoformat(fix["kickoff"].replace("Z", "+00:00"))
                ko_str = ko.strftime("%H:%M")
            except Exception:
                ko_str = ""
            return fix, result, ko_str
        time.sleep(0.15)
    return None, None, None


def banko_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Tekrar Analiz", callback_data="banko"),
            InlineKeyboardButton("Botu Kapat", callback_data="dur"),
        ]
    ])


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    await update.message.reply_text(
        "Otomatik /banko Mac Analizi\n\n"
        "%72+ guven duvari\n"
        "Tum dunya ligleri\n"
        "Bol gollu ligler oncelikli\n\n"
        "/banko - Analiz baslat\n"
        "/dur   - Botu durdur"
    )


async def banko_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_active
    if update.effective_user.id != ADMIN_ID:
        return

    if not bot_active:
        bot_active = True

    wait = await update.message.reply_text(
        "Bultendeki maclar analiz ediliyor...\n"
        "Bol gollu liglerden basliyorum."
    )

    try:
        fix, result, ko_str = find_banko()
        if not fix or not result:
            await wait.edit_text(
                "Bugun %72+ guvenli mac bulunamadi.\n\n"
                "/banko ile tekrar dene.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("Tekrar Dene", callback_data="banko")
                ]])
            )
            return

        msg = format_result(result, fix["league"], ko_str)
        await wait.edit_text(msg, reply_markup=banko_keyboard())

    except Exception as e:
        await wait.edit_text("Hata: " + str(e))


async def dur_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_active
    if update.effective_user.id != ADMIN_ID:
        return
    bot_active = False
    await update.message.reply_text(
        "Bot durduruldu. API cekimi durduruldu.\n"
        "Tasarruf modu aktif.\n\n"
        "/banko ile yeniden baslat."
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_active
    query = update.callback_query
    if query.from_user.id != ADMIN_ID:
        await query.answer("Yetkisiz.")
        return

    await query.answer()

    if query.data == "dur":
        bot_active = False
        await query.edit_message_text(
            "Bot durduruldu. API cekimi durduruldu.\n"
            "Tasarruf modu aktif.\n\n"
            "/banko ile yeniden baslat."
        )
        return

    if query.data == "banko":
        bot_active = True
        await query.edit_message_text(
            "Bultendeki maclar analiz ediliyor...\n"
            "Bol gollu liglerden basliyorum."
        )
        try:
            fix, result, ko_str = find_banko()
            if not fix or not result:
                await query.edit_message_text(
                    "Bugun %72+ guvenli mac bulunamadi.\n\n"
                    "Tekrar dene.",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("Tekrar Dene", callback_data="banko")
                    ]])
                )
                return
            msg = format_result(result, fix["league"], ko_str)
            await query.edit_message_text(msg, reply_markup=banko_keyboard())
        except Exception as e:
            await query.edit_message_text("Hata: " + str(e))


def run_bot():
    while True:
        try:
            app = Application.builder().token(BOT_TOKEN).build()
            app.add_handler(CommandHandler("start", start_cmd))
            app.add_handler(CommandHandler("banko", banko_cmd))
            app.add_handler(CommandHandler("dur", dur_cmd))
            app.add_handler(CallbackQueryHandler(button_handler))
            print("BANKO BOT RUNNING")
            app.run_polling(drop_pending_updates=True)
        except Exception as e:
            print("BOT RESTARTING: " + str(e))
            time.sleep(5)


run_bot()
