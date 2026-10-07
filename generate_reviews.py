"""
AccessIQ smart keypad doorknob lock: simulated review generator.

Produces review records in an Amazon / PromptCloud-style 18-column layout,
plus hidden ground-truth label columns (prefixed label_) for evaluation.
Seeded so every team member regenerates the identical dataset.

Usage: python generate_reviews.py [N] [SEED]
"""
import csv, random, sys, hashlib
from datetime import date, datetime, timedelta

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 6422
rng = random.Random(SEED)

# ---------------------------------------------------------------------------
# Product (fictional)
# ---------------------------------------------------------------------------
PRODUCT_ID = "B0AIQK1LCK"
PRODUCT_NAME = ("AccessIQ K1 Keyless Entry Smart Doorknob Lock with Backlit Keypad, "
                "Bluetooth App Control, Auto-Lock, 100 User Codes")
BRAND = "AccessIQ"
CATEGORY = "Tools & Home Improvement > Safety & Security > Door Hardware & Locks > Knobs"
DESCRIPTION = ("Keyless entry doorknob with a backlit 10-digit keypad, Bluetooth app control, "
               "auto-lock (30s-5min), up to 100 user codes, one-time guest codes, activity log, "
               "anti-peep code entry, low-battery alerts and a backup key. Runs on 4 AA batteries. "
               "Fits standard doors 1-3/8\" to 1-3/4\" thick with 2-3/8\" or 2-3/4\" backset. "
               "Optional AccessIQ Wi-Fi Bridge (sold separately) enables remote access.")
PRICE = "$79.99"
URL = f"https://www.example.com/dp/{PRODUCT_ID}"
FINISHES = [("Satin Nickel", .5), ("Matte Black", .35), ("Aged Bronze", .15)]

THEMES = ["installation_difficulty", "lock_reliability", "battery_life", "keypad_usability",
          "build_quality", "app_connectivity", "security_concerns"]

# ---------------------------------------------------------------------------
# Phrase banks. Negative phrases are (text, severity). Many wordings per issue
# on purpose: recognising paraphrases as one theme is the core task.
# ---------------------------------------------------------------------------
NEG = {
"installation_difficulty": [
    ("The instructions are basically just pictures and half of them don't match what's in the box.", "low"),
    ("Install took me almost three hours. The latch would not line up with the strike plate no matter what I did.", "medium"),
    ("Our door is 1-3/4\" and the screws they include were too short, had to go to the hardware store.", "medium"),
    ("Flipping the handle for a left-hand door is not explained anywhere. Figured it out from a video someone else posted.", "low"),
    ("Getting the ribbon cable through the door without pinching it was a nightmare.", "medium"),
    ("I pinched the cable during install and the keypad was dead. Took it apart twice before it worked.", "medium"),
    ("Says it fits standard holes but I still had to chisel out the edge of the door for the latch.", "medium"),
    ("Setup is not beginner friendly. The backset adjustment is fiddly and the paper template was useless.", "low"),
    ("Took two of us to hold the inside and outside pieces together while tightening the screws.", "low"),
    ("Installing it was way harder than the 15 minutes the box claims. More like an afternoon.", "low"),
    ("The latch bolt kept sticking until I loosened everything and reinstalled it from scratch.", "medium"),
    ("Struggled with the install, the inside housing would not sit flush against the door.", "low"),
    ("Not a simple swap for an old knob like the listing implies. Expect to drill.", "medium"),
    ("Programming the master code during setup failed four times before it took.", "low"),
],
"lock_reliability": [
    ("The lock works inconsistently. Some days fine, some days I punch the code in three times.", "high"),
    ("It sometimes does not unlock even though the keypad flashes green.", "high"),
    ("The keypad stops responding randomly and I have to wait a minute before it wakes up.", "high"),
    ("Got locked out twice this month because it just wouldn't open. Had to use the backup key both times.", "high"),
    ("You hear the motor grind but the latch doesn't retract.", "high"),
    ("Auto-lock is hit or miss, half the time I end up locking it by hand.", "medium"),
    ("Every few days the handle spins freely and won't open the door at all.", "high"),
    ("Code is accepted, green light, beep, and the knob still won't turn. Every so often.", "high"),
    ("Mine started glitching after about two months. Have to enter my code over and over.", "medium"),
    ("Sometimes the knob just won't engage after a correct code. Very frustrating in the rain with groceries.", "high"),
    ("Keypad goes completely dark at random, then comes back on its own an hour later.", "high"),
    ("It freezes up. Pulling the batteries and putting them back in is the only fix.", "high"),
    ("Worked great for a few weeks and now it fails to unlock maybe 1 out of 5 tries.", "high"),
    ("When it's cold out the lock gets sluggish and sometimes doesn't open on the first try.", "medium"),
    ("Unreliable. My kids have been stuck outside waiting for me because it wouldn't take their code.", "high"),
],
"battery_life": [
    ("Batteries lasted about five weeks. That's with name-brand AAs.", "medium"),
    ("Eats batteries. On my third set in four months.", "medium"),
    ("No low battery warning at all, it just died and I was locked out.", "high"),
    ("Battery indicator in the app said 80% the day before it died.", "high"),
    ("The low-battery beep is so quiet nobody in my house noticed it.", "medium"),
    ("Battery life is nowhere near the 'up to 12 months' claimed. Closer to 6 weeks.", "medium"),
    ("Changing the batteries means taking the inside cover off with a screwdriver every time. Annoying when you do it this often.", "low"),
    ("Rechargeable AAs don't work in it, it reads them as dead.", "low"),
    ("Batteries drain way faster with auto-lock and Bluetooth on, which kind of defeats the point.", "medium"),
    ("Power dies fast. I keep a pack of AAs by the door now.", "medium"),
    ("It went from full to dead in a weekend once. Not sure what happened.", "high"),
],
"keypad_usability": [
    ("Keypad numbers are almost impossible to read in direct sunlight.", "low"),
    ("The buttons are touch-sensitive and register double presses all the time.", "medium"),
    ("You have to press each button really hard or it doesn't register.", "low"),
    ("Two numbers per button (1/2, 3/4...) confuses my parents every single time.", "low"),
    ("Backlight is so dim at night you can barely see what you're pressing.", "low"),
    ("Adding a new user code from the keypad is a 12-step process. Without the app it's hopeless.", "low"),
    ("Can't use the keypad with gloves on, which matters here in winter.", "low"),
    ("The keypad times out too quickly if you're a little slow typing the code.", "low"),
    ("The beep is ear-splitting and there's no way to turn the volume down from the keypad.", "low"),
    ("My 8 year old can't figure out the keypad. Needs a clearer layout.", "low"),
    ("Hard to hit the right number. The keys are tiny and very close together.", "low"),
    ("Wet fingers and the keypad goes haywire.", "medium"),
],
"build_quality": [
    ("Feels cheap. The outside knob is mostly plastic with a metal-look coating.", "low"),
    ("The finish started peeling within three months on a covered porch.", "low"),
    ("The handle got wobbly after a few weeks even after re-tightening.", "medium"),
    ("Rattles every time the door closes.", "low"),
    ("Water got behind the keypad after a storm and now two buttons don't work.", "high"),
    ("The keypad cover cracked when my son bumped it with his bike.", "medium"),
    ("For eighty bucks I expected something sturdier. The inside housing flexes when you push on it.", "low"),
    ("Black finish is fading to gray on the side that gets afternoon sun.", "low"),
    ("Knob has noticeable play in it. Doesn't feel like it would survive a few years.", "medium"),
    ("The battery cover clip snapped the second time I opened it.", "low"),
],
"app_connectivity": [
    ("App won't pair with the lock. Tried on two different phones.", "medium"),
    ("Bluetooth drops constantly once you're more than a few feet from the door.", "medium"),
    ("Remote unlock needs the Wi-Fi bridge which is sold separately. Should be clearer in the listing.", "low"),
    ("Since the last firmware update the app keeps logging me out and losing the lock.", "medium"),
    ("Notifications show up 10-15 minutes late, if they show up at all.", "medium"),
    ("App is clunky and asks you to make an account just to use your own lock.", "low"),
    ("The activity log is missing entries so I can't tell who came in when.", "medium"),
    ("After the update my guest codes disappeared and I had to redo all of them.", "medium"),
    ("Connection to the app is spotty, sometimes it finds the lock, sometimes it doesn't.", "medium"),
    ("Unlocking from the app takes forever to connect. Faster to just type the code.", "low"),
    ("App crashes when I try to schedule a temporary code.", "medium"),
],
"security_concerns": [
    ("I deleted my old dog walker's code and two weeks later the log shows it was used to get in.", "high"),
    ("Came home and the door was unlocked even though the app said locked all day. I'm genuinely scared now.", "high"),
    ("The keypad accepted a code I never set. I watched it happen. That should never be possible.", "high"),
    ("Auto-lock failed overnight and we slept with the front door unlocked. Not acceptable for a lock.", "high"),
    ("After a power cycle it reset to the factory default code without telling me.", "high"),
    ("A locksmith friend opened it in seconds without the code or key. Feels like a false sense of security.", "high"),
    ("The wear on the most-used buttons makes it obvious which numbers are in our code.", "medium"),
    ("Someone jiggled the outside knob hard and it opened. Reported it to support, no reply.", "high"),
    ("The app said 'locked' but the door was actually unlocked when I got home.", "high"),
    ("The lock unlocked by itself in the middle of the night. The log shows 'unlocked by app' and none of us did it.", "high"),
],
}

POS = {
"installation_difficulty": [
    "Installed in about 20 minutes with just a screwdriver.",
    "Fit right into the existing holes from our old knob, no drilling.",
    "Install was easy, the video in the app walks you through every step.",
    "Swapped it in on my own in under half an hour and I am not handy.",
    "Instructions were clear and everything lined up the first time.",
],
"lock_reliability": [
    "Works every single time, no hiccups.",
    "Six months in and it has never failed to open.",
    "Auto-lock is reliable, I never wonder if I left the door unlocked.",
    "Opens quick and quiet, haven't had one issue.",
    "Has been rock solid through a hot summer and a cold winter.",
],
"battery_life": [
    "Still on the original batteries after seven months.",
    "The low-battery warning gives you weeks of notice, plenty of time.",
    "Battery life has been great, way better than I expected.",
    "First set of batteries lasted almost a year.",
],
"keypad_usability": [
    "The backlit keypad is easy to see at night.",
    "My elderly parents can use the keypad with no trouble.",
    "Love that the kids can just punch in their code after school.",
    "Keypad is responsive and the numbers are big and clear.",
    "Setting up codes for each family member took a couple of minutes.",
],
"build_quality": [
    "Feels solid and heavy, not cheap at all.",
    "The finish still looks brand new after a year on a south-facing door.",
    "Matches our other hardware and looks more expensive than it is.",
    "Survived a pretty rough winter with no issues.",
],
"app_connectivity": [
    "App paired on the first try and is easy to use.",
    "Love getting a notification when the kids get home.",
    "Guest codes through the app are perfect for our Airbnb.",
    "Being able to check the activity log from work is great.",
    "Bluetooth connects instantly when I walk up.",
],
"security_concerns": [
    "The anti-peep feature where you can type extra digits around your code is a nice touch.",
    "Feel a lot safer knowing it locks itself behind us.",
    "No more hidden key under the mat, which was always a worry.",
    "Being able to delete a code the moment someone moves out is great peace of mind.",
],
}

NEG_TITLES = {
"installation_difficulty": ["Install was a headache", "Not a 15 minute install", "Instructions are terrible",
                            "Hard to install", "Didn't fit my door", "Prepare to drill", "Frustrating setup"],
"lock_reliability": ["Unreliable", "Locked out AGAIN", "Works... sometimes", "Stopped working right",
                     "Inconsistent", "Can't trust it", "Glitchy", "Not reliable"],
"battery_life": ["Battery hog", "Batteries die way too fast", "No battery warning", "Eats AAs",
                 "Battery life is a joke", "Constantly changing batteries"],
"keypad_usability": ["Keypad is annoying", "Hard to use keypad", "Can't read the numbers",
                     "Confusing keypad", "Buttons are finicky"],
"build_quality": ["Feels cheap", "Finish peeling already", "Flimsy", "Poor build", "Not built to last"],
"app_connectivity": ["App is terrible", "Bluetooth keeps dropping", "Won't connect", "App needs work",
                     "App is frustrating", "Connectivity issues"],
"security_concerns": ["SECURITY ISSUE", "Do not trust this lock", "Not secure",
                      "Serious safety problem", "Scary", "Safety concern"],
}
POS_TITLES = ["Love it!", "Great lock", "Five stars", "Best purchase this year", "So convenient",
              "Never carrying keys again", "Works great", "Highly recommend", "Exactly what we needed",
              "Great value", "Easy and reliable", "Love this lock", "Perfect for our family",
              "Wish I bought it sooner", "Good lock", "Solid", "Very happy", "Game changer"]
MIXED_TITLES = ["Good but not perfect", "Mostly happy", "Pros and cons", "Almost great",
                "Decent with one big flaw", "Love it except for one thing", "3 stars for now", "It's ok"]
NEUTRAL_TITLES = ["It's fine", "Does the job", "Average", "OK for the price", "Nothing special", "Meh"]

OPEN_POS = ["", "", "", "Bought this for our back door.", "Replaced our old keyed knob with this.",
            "Got this for the side door to the garage.", "We have teenagers who lose keys constantly.",
            "Bought two of these for a rental property.", "Second one I've bought.",
            "Got this after locking myself out one too many times.", "Upgraded from a basic keyed knob."]
OPEN_NEG = ["", "", "Wanted to love this.", "Bought this for our front door.",
            "Had high hopes for this one.", "Replaced a perfectly good keyed knob with this, big mistake.",
            "Bought for our rental unit.", "Been using it about three months now."]
CLOSE_POS = ["", "", "", "Highly recommend!", "Would buy again.", "Great value for the money.",
             "No more fumbling for keys.", "Ordering another for the back door.", "10/10.",
             "Very happy with it."]
CLOSE_NEG = ["", "", "Returning it.", "Would not recommend.", "Customer support has not been helpful.",
             "Look elsewhere.", "Very disappointed.", "Sent it back for a refund.",
             "Support told me to reset it, which didn't help."]
CLOSE_MIXED = ["", "", "If they fix that it's a 5 star lock.", "Still keeping it for now.",
               "Overall decent for the price.", "Hoping a firmware update fixes it.",
               "Would buy again, but know what you're getting."]
GENERIC_POS = ["Love this lock.", "Great product.", "So convenient!", "Works exactly as described.",
               "Super easy to use.", "Exactly what I wanted.", "Best upgrade we've made to the house.",
               "Couldn't be happier."]
NEUTRAL_BODY = ["It's a lock. It locks. Nothing special but nothing wrong either.",
                "Does what it says. Not amazing, not terrible.",
                "Fine for the price. Haven't used the app much.",
                "It works ok. Honestly the keypad is the only feature we use.",
                "Average smart lock. No real complaints but no wow factor.",
                "Works as expected so far. Only had it a couple weeks so we'll see.",
                "It's fine. Took a little getting used to.",
                "Gets the job done. Wouldn't call it great."]

FEATURE_REQUESTS = [
    "Wish it had a fingerprint reader.",
    "Would love Alexa or Google Home support.",
    "Please add Apple Home support.",
    "It really needs a matching deadbolt version.",
    "A rechargeable battery pack would be a big improvement.",
    "Would be nice if you could schedule codes for certain days of the week.",
    "Hope they add a way to change the beep volume.",
    "Should come with the Wi-Fi bridge included.",
    "A dark bronze finish option would be nice.",
    "Wish there was a doorbell built into the keypad.",
]

OFF_TOPIC = [
    ("Box was crushed", "Box arrived completely crushed. Lock inside seemed okay but haven't installed it yet.", [3, 4, 2]),
    ("Slow shipping", "Took almost three weeks to arrive when it said 2 days.", [2, 3, 1]),
    ("Left in the rain", "Delivery driver left it in the rain on the porch. Packaging was soaked.", [3, 2]),
    ("Wrong color sent", "Ordered satin nickel, received matte black. Exchange process was a hassle.", [2, 3]),
    ("Too much packaging", "Way too much plastic packaging for one doorknob.", [4, 3]),
    ("Late delivery", "Package was marked delivered but showed up two days later.", [3, 4]),
    ("Gift", "Bought this as a gift, they haven't installed it yet. Shipping was fast though.", [5, 4]),
    ("Great seller", "Seller was great, refund was processed quickly when I ordered the wrong one.", [5, 4]),
    ("Box was opened", "Arrived with the outer box opened and retaped. Made me nervous about whether it was used.", [2, 3]),
    ("Price dropped", "Price dropped $15 a week after I bought it. Annoying.", [3, 4]),
    ("Arrived on time", "Shipping box was way bigger than needed. Arrived on time.", [4, 5]),
    ("Fast delivery", "Haven't installed it yet, just reviewing the fast delivery.", [5]),
    ("Return hassle", "Return label never got emailed to me, had to call twice.", [1, 2]),
    ("Arrived early", "Came a day early, nicely packed.", [5]),
]

NAMES_FIRST = ["Mike", "Jen", "Carlos", "Ashley", "David", "Maria", "Chris", "Linda", "Kevin", "Priya",
               "Tom", "Angela", "Jamal", "Sarah", "Luis", "Karen", "Brian", "Tasha", "Steve", "Nicole",
               "Raj", "Emily", "Greg", "Diana", "Andre", "Megan", "Paul", "Yolanda", "Josh", "Hannah"]
HANDLES = ["Amazon Customer", "Amazon Customer", "Amazon Customer", "HomeownerFL", "DIY Dad",
           "busy mom of 3", "J. Rivera", "Kindle Customer", "TechieGran", "R. Patel", "landlord_bob",
           "Coffee Addict", "Weekend Warrior", "K. Nguyen", "Retired Teacher", "night shift nurse"]

# ---------------------------------------------------------------------------
NEG_WEIGHTS = {"lock_reliability": .27, "battery_life": .22, "app_connectivity": .14,
               "installation_difficulty": .14, "keypad_usability": .12, "build_quality": .11}
UPDATE_DATE = date(2026, 4, 15)       # firmware update referenced by some app complaints
START, END = date(2025, 3, 1), date(2026, 9, 30)

def wpick(d):
    keys = list(d); return rng.choices(keys, weights=[d[k] for k in keys])[0]

def rdate(after=None):
    lo = after or START
    return lo + timedelta(days=rng.randint(0, (END - lo).days))

def name():
    if rng.random() < .45: return rng.choice(HANDLES)
    return f"{rng.choice(NAMES_FIRST)} {rng.choice('ABCDEFGHJKLMNPRSTW')}."

def casualize(t):
    """Light informal noise so text isn't uniformly polished."""
    r = rng.random()
    if r < .08: t = t.lower()
    elif r < .13: t = t.replace(". ", "... ", 1)
    elif r < .17: t = t.rstrip(".") + "!!"
    return t

SEV_RANK = {"low": 0, "medium": 1, "high": 2}

def build(kind):
    """Return (title, body, rating, labels dict, date_floor)."""
    themes, tsent, sev, fr, floor = [], {}, "none", False, None
    parts = []

    def add_neg(theme, open_=True):
        nonlocal sev, floor
        text, s = rng.choice(NEG[theme])
        if "update" in text: floor = UPDATE_DATE
        themes.append(theme); tsent[theme] = "negative"
        sev = s if sev == "none" or SEV_RANK[s] > SEV_RANK[sev] else sev
        parts.append(text)
        if rng.random() < .35:                       # second paraphrase of the same issue
            t2, s2 = rng.choice([p for p in NEG[theme] if p[0] != text])
            if "update" in t2: floor = UPDATE_DATE
            sev = s2 if SEV_RANK[s2] > SEV_RANK[sev] else sev
            parts.append(t2)

    def add_pos(theme):
        themes.append(theme); tsent[theme] = "positive"
        parts.append(rng.choice(POS[theme]))

    if kind == "positive":
        title = rng.choice(POS_TITLES)
        parts.append(rng.choice(OPEN_POS))
        k = rng.choices([0, 1, 2, 3], weights=[.2, .35, .3, .15])[0]
        for t in rng.sample(THEMES, k): add_pos(t)
        if k == 0 or rng.random() < .4: parts.append(rng.choice(GENERIC_POS))
        parts.append(rng.choice(CLOSE_POS))
        rating = 5 if rng.random() < .85 else 4
        overall = "positive"
    elif kind == "neutral":
        title = rng.choice(NEUTRAL_TITLES)
        parts.append(rng.choice(NEUTRAL_BODY))
        rating = rng.choices([3, 4], weights=[.75, .25])[0]
        overall = "neutral"
    elif kind == "negative":
        th = wpick(NEG_WEIGHTS)
        title = rng.choice(NEG_TITLES[th])
        parts.append(rng.choice(OPEN_NEG)); add_neg(th); parts.append(rng.choice(CLOSE_NEG))
        rating = rng.choices([1, 2, 3], weights=[.5, .35, .15])[0]
        overall = "negative"
    elif kind == "negative_multi":
        a = wpick(NEG_WEIGHTS)
        b = wpick({k: v for k, v in NEG_WEIGHTS.items() if k != a})
        title = rng.choice(NEG_TITLES[a])
        parts.append(rng.choice(OPEN_NEG)); add_neg(a); parts.append(rng.choice(["On top of that,", "Also", "And", ""]))
        add_neg(b); parts.append(rng.choice(CLOSE_NEG))
        rating = rng.choices([1, 2], weights=[.6, .4])[0]
        overall = "negative"
    elif kind == "mixed":
        neg = wpick(NEG_WEIGHTS)
        pos = rng.choice([t for t in THEMES if t not in (neg, "security_concerns")])
        title = rng.choice(MIXED_TITLES + NEG_TITLES[neg][:2])
        if rng.random() < .5:
            add_pos(pos); parts.append(rng.choice(["But", "However,", "The problem is", "Downside:", ""])); add_neg(neg)
        else:
            add_neg(neg); parts.append(rng.choice(["That said,", "On the plus side,", "Otherwise,", ""])); add_pos(pos)
        parts.append(rng.choice(CLOSE_MIXED))
        rating = rng.choices([2, 3, 4], weights=[.15, .45, .4])[0]
        overall = "mixed"
    elif kind == "security":
        title = rng.choice(NEG_TITLES["security_concerns"])
        parts.append(rng.choice(OPEN_NEG)); add_neg("security_concerns")
        if rng.random() < .3: add_neg(wpick(NEG_WEIGHTS))
        parts.append(rng.choice(CLOSE_NEG + ["Contacted support and still waiting to hear back."]))
        rating = 1 if rng.random() < .85 else 2
        overall = "negative"
    elif kind == "off_topic":
        title, text, ratings = rng.choice(OFF_TOPIC)
        parts.append(text); rating = rng.choice(ratings)
        overall = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")
    else:
        raise ValueError(kind)

    if kind in ("positive", "mixed", "negative", "neutral") and rng.random() < .09:
        fr = True; parts.insert(-1, rng.choice(FEATURE_REQUESTS))

    # Join, tidy connectives
    out = []
    for p in parts:
        if not p: continue
        if out and out[-1] in ("But", "And", "Also", "However,", "That said,", "On the plus side,",
                               "Otherwise,", "The problem is", "Downside:", "On top of that,"):
            conn = out.pop()
            p = (conn + " " + (p[0].lower() + p[1:] if conn not in ("Downside:",) else p))
        out.append(p)
    body = casualize(" ".join(out).strip())

    neg_themes = [t for t in themes if tsent.get(t) == "negative"]
    # Primary complaint: security outranks everything, then first-mentioned
    primary = ("security_concerns" if "security_concerns" in neg_themes else
               (neg_themes[0] if neg_themes else "none"))
    labels = {
        "label_review_type": kind,
        "label_overall_sentiment": overall,
        "label_themes": ";".join(dict.fromkeys(themes)) or "none",
        "label_theme_sentiments": ";".join(f"{t}:{s}" for t, s in tsent.items()) or "none",
        "label_primary_complaint": primary,
        "label_severity": sev,
        "label_feature_request": fr,
        "label_off_topic": kind == "off_topic",
    }
    return title, body, rating, labels, floor

PLAN = {"positive": 290, "negative": 85, "mixed": 55, "negative_multi": 15,
        "neutral": 20, "off_topic": 25, "security": 10}
scale = N / sum(PLAN.values())
kinds = []
for k, v in PLAN.items(): kinds += [k] * round(v * scale)
kinds = (kinds + ["positive"] * N)[:N]
rng.shuffle(kinds)

rows, seen = [], set()
for k in kinds:
    for _ in range(50):
        title, body, rating, labels, floor = build(k)
        if body not in seen: break
    seen.add(body)
    d = rdate(floor)
    crawl = datetime(2026, 10, 1, rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
    neg = labels["label_overall_sentiment"] == "negative"
    votes = int(rng.paretovariate(1.6 if neg else 2.4)) - 1
    rows.append({
        "Uniq Id": hashlib.md5(f"{SEED}-{len(rows)}".encode()).hexdigest(),
        "Crawl Timestamp": crawl.strftime("%Y-%m-%d %H:%M:%S +0000"),
        "Product Id": PRODUCT_ID,
        "Product Name": PRODUCT_NAME,
        "Brand": BRAND,
        "Category": CATEGORY,
        "Product Description": DESCRIPTION,
        "Product Price": PRICE,
        "Product Url": URL,
        "Variant": "Finish: " + rng.choices([f for f, _ in FINISHES], weights=[w for _, w in FINISHES])[0],
        "Reviewer Name": name(),
        "Review Title": title,
        "Review Content": body,
        "Review Rating": rating,
        "Review Date": d.isoformat(),
        "Review Location": "Reviewed in the United States",
        "Verified Purchase": rng.random() < .88,
        "Helpful Votes": min(votes, 250),
        **labels,
    })

rows.sort(key=lambda r: r["Review Date"])
for i, r in enumerate(rows, 1): r["Review Id"] = f"AIQ-{i:04d}"

# Stratified 100-review held-out test split (proportional by review type)
test_n = round(N * 0.2)
by_type = {}
for r in rows: by_type.setdefault(r["label_review_type"], []).append(r)
test_ids = set()
for t, rs in by_type.items():
    n = max(1, round(len(rs) / N * test_n))
    test_ids |= {r["Review Id"] for r in rng.sample(rs, min(n, len(rs)))}
for r in rows: r["split"] = "test" if r["Review Id"] in test_ids else "train"

BASE = ["Review Id", "Uniq Id", "Crawl Timestamp", "Product Id", "Product Name", "Brand", "Category",
        "Product Description", "Product Price", "Product Url", "Variant", "Reviewer Name",
        "Review Title", "Review Content", "Review Rating", "Review Date", "Review Location",
        "Verified Purchase", "Helpful Votes"]
LABELS = ["split", "label_review_type", "label_overall_sentiment", "label_themes",
          "label_theme_sentiments", "label_primary_complaint", "label_severity",
          "label_feature_request", "label_off_topic"]

with open("accessiq_reviews_labeled.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, BASE + LABELS); w.writeheader(); w.writerows(rows)
with open("accessiq_reviews_unlabeled.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, BASE, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows, {len(test_ids)} test")
