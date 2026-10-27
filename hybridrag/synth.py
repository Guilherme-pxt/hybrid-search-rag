"""Seeded generator for a fictional company's internal docs and a gold question set.

The company, Quorvia Instruments, is invented: it sells environmental sensors
and a monitoring SaaS ("Quorvia Cloud"). Nothing here describes a real company
or a real person.

The corpus is built to make the wrong-but-similar document a real failure
mode. Most documents come in template families whose members differ only in
the entity (region, plan, product tier, service, version) and in the numbers:

* 8 regional policies x 4 regions (EU, US, UK, APAC)
* 12 product spec sheets (4 product lines x 3 tiers, each with a SKU)
* 12 pricing pages (3 plans x 4 regions)
* 3 SLA tiers, 10 service runbooks, 10 release notes, 5 office guides
* 12 one-off documents (security, password policy, rate limits, ...)

Within a family every attribute gets a distinct value, so an answer taken
from a sibling document is detectably wrong.

Questions come in four types:

* ``exact_code``   names a SKU, error code, ticket, version or RMA prefix
* ``lexical``      uses the document's own wording and names the entity
* ``paraphrase``   synonyms and indirect references ("staff in Britain",
                   "the mid-range air-quality sensor")
* ``unanswerable`` near misses that the corpus does not cover (a region,
                   SKU, service or version that does not exist, or an
                   attribute nobody documented); the right answer is "I don't know"

``python -m hybridrag synth --seed 42`` regenerates byte-identical files.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

COMPANY = "Quorvia Instruments"

REGIONS = {
    "eu": {"name": "EU", "entity": "Quorvia Instruments B.V.", "countries": "the Netherlands, Portugal and Germany",
           "cur": "EUR", "dc": "Frankfurt",
           "staff": ["someone based in Portugal", "staff in the Netherlands", "an employee in Germany"],
           "customers": ["a customer in Germany", "a client in Portugal", "a customer in the Netherlands"]},
    "us": {"name": "US", "entity": "Quorvia Instruments Inc.", "countries": "the United States",
           "cur": "USD", "dc": "Virginia",
           "staff": ["staff employed in the United States", "an American employee", "someone on the US payroll"],
           "customers": ["a customer in the United States", "an American client"]},
    "uk": {"name": "UK", "entity": "Quorvia Instruments Ltd.", "countries": "the United Kingdom",
           "cur": "GBP", "dc": "London",
           "staff": ["staff in Britain", "an employee in the United Kingdom", "someone on the British payroll"],
           "customers": ["a British customer", "a client in the United Kingdom"]},
    "apac": {"name": "APAC", "entity": "Quorvia Instruments Pte. Ltd.", "countries": "Singapore, Australia and Japan",
             "cur": "SGD", "dc": "Singapore",
             "staff": ["staff in Singapore", "an employee in Australia", "someone in the Japan team"],
             "customers": ["a customer in Australia", "a client in Japan", "a customer in Singapore"]},
}


@dataclass
class Doc:
    doc_id: str
    title: str
    family: str
    entity: str
    sections: list[tuple[str, str]]

    @property
    def text(self) -> str:
        return "\n\n".join(f"## {h}\n{t}" for h, t in self.sections)

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "family": self.family, "entity": self.entity,
                "text": self.text}


@dataclass
class Question:
    qid: str
    question: str
    qtype: str
    family: str | None
    gold_doc_id: str | None
    gold_fact: str | None
    evidence: str | None
    attribute: str | None = None
    distractor_doc_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Attr:
    """One fact slot of a template family."""

    key: str
    section: str
    sentence: str            # format string with {v} and optionally {cur}
    pool: list
    kw: str | None = None    # lexical / exact-code question template
    para: str | None = None  # paraphrase question template
    kw_type: str = "lexical"
    fact: str = "{v}"
    fixed: dict | None = None  # entity -> value, instead of sampling from pool


def _r(a, b, step=1):
    return list(range(a, b, step))


def _money(vals):
    return [f"{v:.2f}" for v in vals]


# ------------------------------------------------------------------ regional policies

REGIONAL = {
    "travel": {
        "title": "Travel and Expense Policy ({R})", "scope": "staff",
        "filler": ["Expenses are submitted in the expense tool within 30 days of the end of the trip.",
                   "Economy class is the default for all flights.",
                   "Receipts are required for every expense except the meal per diem."],
        "attrs": [
            Attr("meal", "Meals", "The meal per diem is {cur} {v} per travel day and is paid without receipts.",
                 _r(40, 100, 5), "What is the meal per diem in the {R} travel and expense policy?",
                 "How much can {scope} spend on food per day during a business trip?"),
            Attr("hotel", "Accommodation",
                 "Hotel bookings are capped at {cur} {v} per night; anything above the cap needs written approval "
                 "from a director.", _r(120, 270, 10), "What is the nightly hotel cap in the {R} travel policy?",
                 "What is the most the company will pay for one night in a hotel for {scope}?"),
            Attr("mileage", "Private car", "Business use of a private car is reimbursed at {cur} {v} per kilometre.",
                 _money([0.25 + 0.05 * i for i in range(8)]),
                 "What is the private car mileage rate in the {R} travel policy?",
                 "If {scope} drives their own car to visit a customer, how much is paid back per kilometre?"),
            Attr("booking", "Booking", "Flights must be booked at least {v} days before departure unless the trip "
                 "is urgent.", _r(7, 22), "How many days in advance must flights be booked under the {R} travel policy?",
                 "How early does {scope} have to buy plane tickets for a work trip?"),
            Attr("preapproval", "Approval", "Trips with an estimated total cost above {cur} {v} require pre-approval "
                 "from the budget owner.", _r(1500, 4250, 250),
                 "Above what trip cost is pre-approval required in the {R} travel policy?",
                 "From what total amount does {scope} need a manager's sign-off before travelling?"),
        ],
    },
    "remote": {
        "title": "Remote Work Policy ({R})", "scope": "staff",
        "filler": ["Remote days are agreed with the team lead at least one week ahead.",
                   "Employees remain reachable on chat during core collaboration hours.",
                   "Company equipment must not be used by family members."],
        "attrs": [
            Attr("remote_days", "Remote days", "Employees may work remotely for up to {v} days per calendar month.",
                 _r(6, 15), "How many remote work days per month does the {R} remote work policy allow?",
                 "How many days a month can {scope} work from home?"),
            Attr("stipend", "Home office", "New hires receive a one-time home office stipend of {cur} {v} for a desk, "
                 "chair and accessories.", _r(300, 950, 50), "What is the home office stipend in the {R} remote work policy?",
                 "How much money does {scope} get once to set up a workspace at home?"),
            Attr("internet", "Internet", "A monthly internet allowance of {cur} {v} is paid with the salary.",
                 _r(20, 65, 5), "What is the monthly internet allowance in the {R} remote work policy?",
                 "How much does the company contribute each month to the home broadband bill of {scope}?"),
            Attr("abroad", "Working abroad", "Working from a country other than your employment country is allowed "
                 "for up to {v} days per calendar year.", _r(14, 61),
                 "How many days per year can employees work from abroad under the {R} remote work policy?",
                 "For how many days a year may {scope} work while staying in another country?"),
        ],
    },
    "parental": {
        "title": "Parental Leave Policy ({R})", "scope": "staff",
        "filler": ["Leave requests are submitted to People Operations at least eight weeks before the expected date.",
                   "Leave can start up to two weeks before the expected date of birth.",
                   "Benefits and pension contributions continue during paid leave."],
        "attrs": [
            Attr("primary", "Primary caregiver", "The primary caregiver receives {v} weeks of fully paid leave after "
                 "the birth or adoption of a child.", _r(14, 27),
                 "How many weeks of fully paid leave does the primary caregiver get under the {R} parental leave policy?",
                 "How long can {scope} stay home on full pay after having a baby, as the main carer?"),
            Attr("secondary", "Secondary caregiver", "The secondary caregiver receives {v} weeks of fully paid "
                 "leave, to be taken within the first year.", _r(4, 13),
                 "How many weeks of paid leave does the secondary caregiver get under the {R} parental leave policy?",
                 "How much paid time off does the other parent get when a child arrives, for {scope}?"),
            Attr("tenure", "Eligibility", "Employees become eligible after {v} months of continuous employment.",
                 _r(3, 13), "After how many months of employment does the {R} parental leave policy apply?",
                 "How long must {scope} have worked here before qualifying for leave for a newborn?"),
            Attr("phased", "Return to work", "After leave, a phased return at reduced hours is available for up to "
                 "{v} weeks.", _r(4, 11), "How long is the phased return in the {R} parental leave policy?",
                 "After coming back from baby leave, for how many weeks can {scope} work shorter days?"),
        ],
    },
    "pto": {
        "title": "Paid Time Off Policy ({R})", "scope": "staff",
        "filler": ["Time off is requested in the HR portal and approved by the team lead.",
                   "Public holidays follow the local calendar of the employment country.",
                   "Part-time employees receive paid time off pro rata."],
        "attrs": [
            Attr("annual", "Annual allowance", "Full-time employees receive {v} days of paid time off per calendar year.",
                 _r(20, 31), "How many days of paid time off per year does the {R} PTO policy grant?",
                 "How many vacation days does {scope} get each year?"),
            Attr("carryover", "Carry-over", "Up to {v} unused days can be carried over into the next year.",
                 _r(3, 11), "How many unused days can be carried over under the {R} PTO policy?",
                 "How many leftover holiday days can {scope} move to next year?"),
            Attr("deadline", "Carry-over", "Carried-over days must be used by {v}.",
                 ["31 January", "28 February", "31 March", "30 April", "31 May", "30 June"],
                 "By what date must carried-over days be used under the {R} PTO policy?",
                 "When do leftover vacation days from last year expire for {scope}?"),
            Attr("sick", "Sick days", "Up to {v} paid sick days per year can be taken without a medical certificate.",
                 _r(3, 11), "How many sick days without a medical certificate does the {R} PTO policy allow?",
                 "How many days can {scope} be off ill in a year without bringing a doctor's note?"),
            Attr("volunteer", "Volunteering", "Each employee has {v} paid volunteering days per year.",
                 _r(1, 6), "How many paid volunteering days does the {R} PTO policy include?",
                 "How many days a year can {scope} take off, paid, to help a charity?"),
        ],
    },
    "retention": {
        "title": "Data Retention Policy ({R})", "scope": "customers",
        "filler": ["Retention periods apply to production systems and to their backups.",
                   "Legal holds override the retention periods below.",
                   "The Security team reviews this policy every year."],
        "attrs": [
            Attr("telemetry", "Telemetry", "Raw device telemetry is retained for {v} days before it is aggregated.",
                 _r(30, 400, 15), "How long is raw telemetry retained under the {R} data retention policy?",
                 "For how many days are the unaggregated sensor readings of {scope} kept?"),
            Attr("audit", "Audit logs", "Audit logs are kept for {v} months.", _r(12, 85, 6),
                 "How long are audit logs kept under the {R} data retention policy?",
                 "How many months of activity history about who did what are stored for {scope}?"),
            Attr("backup", "Backups", "Encrypted backups are retained for {v} days.", _r(7, 95, 7),
                 "How long are backups retained under the {R} data retention policy?",
                 "How far back can data be restored from a saved copy for {scope}?"),
            Attr("deletion", "Deletion requests", "Verified customer deletion requests are completed within {v} days.",
                 _r(10, 46, 5), "Within how many days are deletion requests completed under the {R} data retention policy?",
                 "If {scope} asks us to erase their data, how long do we have to do it?"),
            Attr("location", "Data location", "Customer data for this region is stored in the {v} data centre.",
                 [], "Where is customer data stored under the {R} data retention policy?",
                 "In which city are the servers that hold the data of {scope}?",
                 fixed={r: REGIONS[r]["dc"] for r in REGIONS}),
        ],
    },
    "equipment": {
        "title": "Equipment Policy ({R})", "scope": "staff",
        "filler": ["Equipment is ordered through the IT service desk.",
                   "Lost or stolen devices must be reported to Security within 24 hours.",
                   "Returned equipment is wiped and reused when possible."],
        "attrs": [
            Attr("laptop", "Laptops", "Laptops are refreshed every {v} months.", [24, 30, 36, 42, 48],
                 "How often are laptops refreshed under the {R} equipment policy?",
                 "How many months until {scope} is due a new work computer?"),
            Attr("monitor", "Monitors", "Each employee may expense one external monitor up to {cur} {v}.",
                 _r(150, 420, 25), "What is the monitor budget in the {R} equipment policy?",
                 "How much can {scope} spend on an extra screen?"),
            Attr("phone", "Mobile phones", "A mobile phone stipend of {cur} {v} per month is paid to employees who "
                 "are on call or customer facing.", _r(15, 50, 5),
                 "What is the monthly phone stipend in the {R} equipment policy?",
                 "How much is {scope} paid each month towards a mobile if they are on call?"),
            Attr("repair", "Repairs", "Broken equipment is replaced within {v} business days.", _r(2, 9),
                 "How quickly is broken equipment replaced under the {R} equipment policy?",
                 "If the laptop of {scope} breaks, how many working days until a replacement arrives?"),
        ],
    },
    "oncall": {
        "title": "On-Call Compensation Policy ({R})", "scope": "staff",
        "filler": ["On-call schedules are published four weeks in advance.",
                   "Swaps are allowed when both engineers update the schedule.",
                   "Time spent on incidents outside working hours is recorded as overtime."],
        "attrs": [
            Attr("weekday", "Weekday rate", "Each weekday on call is compensated with {cur} {v}.", _r(30, 90, 5),
                 "What is the weekday on-call rate in the {R} on-call compensation policy?",
                 "How much extra does {scope} earn for carrying the pager on a normal working day?"),
            Attr("weekend", "Weekend rate", "Each weekend day on call is compensated with {cur} {v}.",
                 _r(100, 260, 10), "What is the weekend on-call rate in the {R} on-call compensation policy?",
                 "What does {scope} get paid for being on standby on a Saturday?"),
            Attr("ack", "Response", "Pages must be acknowledged within {v} minutes.", _r(5, 21),
                 "Within how many minutes must pages be acknowledged under the {R} on-call policy?",
                 "How fast does {scope} have to respond to an alert while on standby?"),
            Attr("maxweeks", "Limits", "Nobody is scheduled for more than {v} on-call weeks per quarter.", _r(2, 7),
                 "What is the maximum number of on-call weeks per quarter in the {R} on-call policy?",
                 "How many weeks of standby duty can {scope} be given in three months, at most?"),
        ],
    },
    "rma": {
        "title": "Returns and Repairs (RMA) Policy ({R})", "scope": "customers",
        "filler": ["Devices must be returned in their original packaging when possible.",
                   "A return authorization must be requested before shipping a device back.",
                   "Refunds are paid to the original payment method."],
        "attrs": [
            Attr("prefix", "RMA numbers", "RMA numbers in this region start with {v}.", [],
                 fixed={"eu": "RMA-EU", "us": "RMA-US", "uk": "RMA-UK", "apac": "RMA-AP"}),
            Attr("window", "Return window", "Customers can return devices within {v} days of delivery for a full refund.",
                 _r(14, 46), "What is the return window for devices with an {P} return number?",
                 "How many days does {scope} have to send a sensor back for a full refund?", kw_type="exact_code"),
            Attr("turnaround", "Repairs", "Warranty repairs are completed within {v} business days of receiving the device.",
                 _r(5, 21), "How long do warranty repairs take for {P} returns?",
                 "How many working days does a warranty repair take for {scope}?", kw_type="exact_code"),
            Attr("shipping", "Shipping", "Quorvia pays return shipping for orders above {cur} {v}.", _r(200, 1050, 50),
                 "Above what order value is return shipping free in the {R} RMA policy?",
                 "From what order size does Quorvia cover the cost of sending a device back for {scope}?"),
        ],
    },
}

# ------------------------------------------------------------------ product spec sheets

LINES = [("pulse", "Pulse", "PU", "air-quality sensor", "measures particulate matter, CO2 and humidity indoors"),
         ("tide", "Tide", "TD", "water-level sensor", "measures water level and flow in tanks and channels"),
         ("ember", "Ember", "EM", "temperature logger", "logs temperature in cold chains and server rooms"),
         ("vane", "Vane", "VN", "wind sensor", "measures wind speed and direction on rooftops and masts")]
TIERS = [("lite", "Lite", "100", "entry-level"), ("pro", "Pro", "200", "mid-range"),
         ("max", "Max", "300", "top-of-the-range")]

SPEC_ATTRS = [
    Attr("battery", "Power", "Battery life is {v} months at the default sampling interval.", _r(6, 61, 3),
         "What is the battery life of the {SKU}?", "How long does the battery last on the {tierdesc} {linedesc}?",
         kw_type="exact_code"),
    Attr("range", "Connectivity", "Radio range is up to {v} metres in open field.", _r(150, 2050, 50),
         "What is the radio range of the {SKU}?", "How far from the gateway can the {tierdesc} {linedesc} be placed?",
         kw_type="exact_code"),
    Attr("sampling", "Measurement", "The default sampling interval is {v} seconds.",
         [5, 10, 15, 20, 25, 30, 40, 45, 60, 75, 90, 120], "What is the default sampling interval of the {SKU}?",
         "How often does the {tierdesc} {linedesc} take a reading by default?", kw_type="exact_code"),
    Attr("ip", "Enclosure", "The enclosure is rated {v} for dust and water.", ["IP54", "IP65", "IP66", "IP67", "IP68"],
         "What is the IP rating of the {SKU}?", "How well is the {tierdesc} {linedesc} sealed against dust and water?",
         kw_type="exact_code"),
    Attr("temp", "Environment", "Operating temperature range is -{v} °C to {w} °C.", _r(10, 41, 5),
         "What is the operating temperature range of the {SKU}?",
         "What is the coldest and hottest the {tierdesc} {linedesc} can work in?", kw_type="exact_code",
         fact="-{v} to {w}"),
    Attr("price", "Pricing", "List price is USD {v} per unit.", _r(129, 900, 10), "What is the list price of the {SKU}?",
         "How much does one {tierdesc} {linedesc} cost?", kw_type="exact_code"),
    Attr("warranty", "Warranty", "Hardware warranty is {v} years from delivery.", [1, 2, 3, 4, 5],
         "How long is the hardware warranty on the {SKU}?",
         "For how many years is the {tierdesc} {linedesc} covered against hardware faults?", kw_type="exact_code"),
]

# ------------------------------------------------------------------ pricing pages

PLANS = [("starter", "Starter", "for small teams getting started", "smallest"),
         ("team", "Team", "for growing teams that share dashboards", "middle"),
         ("enterprise", "Enterprise", "for large organisations with many sites", "largest")]

PRICING_ATTRS = [
    Attr("seat", "Price", "The {plan} plan costs {cur} {v} per user per month.", _r(8, 120),
         "How much does the {plan} plan cost per user per month in {R}?",
         "What would {scope} pay each month per seat on the {plandesc} Quorvia Cloud plan?"),
    Attr("discount", "Annual billing", "Annual billing gives a {v}% discount on the {plan} plan.", _r(8, 26),
         "What discount does annual billing give on the {plan} plan in {R}?",
         "How much cheaper is the {plandesc} plan for {scope} when paying for a year up front?"),
    Attr("devices", "Devices", "The {plan} plan includes {v} connected devices per workspace.", _r(10, 600, 5),
         "How many connected devices are included in the {plan} plan in {R}?",
         "How many sensors can {scope} connect on the {plandesc} plan without paying extra?"),
    Attr("history", "History", "Dashboards on the {plan} plan keep {v} days of history.", _r(30, 400, 30),
         "How many days of dashboard history does the {plan} plan include in {R}?",
         "How far back can {scope} look at charts on the {plandesc} plan?"),
    Attr("overage", "Extra devices", "Each additional device on the {plan} plan costs {cur} {v} per month.",
         _money([0.5 + 0.25 * i for i in range(13)]),
         "What does each additional device cost per month on the {plan} plan in {R}?",
         "What is the monthly charge for every sensor above the included amount on the {plandesc} plan for {scope}?"),
]

# ------------------------------------------------------------------ runbooks

SERVICES = [
    ("ingest-gateway", "ING", "receives telemetry uploads from devices", "the upload queue is full and new uploads are rejected",
     "devices cannot upload their readings because the intake queue has filled up", "drain", "upload queue depth"),
    ("alert-router", "ALR", "decides which notification channel each alert goes to", "alerts are stuck in the routing queue",
     "alerts are not reaching anyone because they are piling up before being dispatched", "flush-queue", "routing lag in seconds"),
    ("billing-sync", "BIL", "copies usage and invoices to the accounting system", "invoices fail to sync to the ledger",
     "invoices are not showing up in the accounting system", "resync", "count of unsynced invoices"),
    ("device-registry", "DRG", "stores the identity and certificates of every device", "device certificates fail validation",
     "sensors are being refused because their identity certificates look invalid", "reload-certs", "certificate validation failures per minute"),
    ("report-builder", "RPT", "renders scheduled PDF and CSV reports", "report rendering times out",
     "scheduled PDF reports never finish generating", "restart", "render time in seconds"),
    ("auth-broker", "AUT", "issues login tokens for the web app and the API", "token signing keys are out of sync",
     "users get logged out because the keys used to sign login tokens disagree", "rotate-keys", "token validation errors per minute"),
    ("firmware-updater", "FWU", "pushes firmware updates to devices over the air", "the update rollout has stalled",
     "over-the-air updates stop progressing across the fleet", "pause-rollout", "failed update percentage"),
    ("metrics-store", "MST", "stores time-series metrics", "disk usage on the storage nodes is too high",
     "the time-series database nodes are running out of disk", "compact", "disk usage percentage"),
    ("notification-hub", "NTF", "sends e-mails, SMS and push notifications", "the SMS provider rejects messages",
     "text messages to customers are bouncing", "failover", "SMS rejection rate percentage"),
    ("export-worker", "EXP", "produces bulk data exports for customers", "export jobs are stuck in the pending state",
     "customer data exports never leave the waiting state", "requeue", "pending export jobs"),
]
TEAMS = ["Telemetry Platform", "Alerting", "Revenue Systems", "Device Identity", "Reporting", "Identity and Access",
         "Fleet Operations", "Storage", "Messaging", "Data Products"]

# ------------------------------------------------------------------ release notes

FEATURES = [
    ("Smart Baselines", "learns a normal range for each sensor and alerts on deviations",
     "get alerted when a sensor drifts away from its usual behaviour without setting thresholds by hand"),
    ("Floor Plans", "places devices on an uploaded building plan", "see sensors drawn on top of a drawing of the building"),
    ("Shared Views", "shares a read-only dashboard link with people outside the workspace",
     "show a dashboard to an outsider without giving them an account"),
    ("Bulk Provisioning", "registers many devices at once from a CSV file", "onboard a whole batch of sensors from a spreadsheet"),
    ("Quiet Hours", "mutes non-critical notifications at night", "stop low-priority alerts from waking people up"),
    ("Audit Export", "downloads the workspace audit log as a CSV file", "save the log of who changed what to a file"),
    ("Calibration Reminders", "reminds device owners when a sensor is due for calibration",
     "get nudged when an instrument needs to be re-checked for accuracy"),
    ("Webhook Signing", "signs every webhook with an HMAC header", "let receivers prove that an outgoing event really came from us"),
    ("Data Gaps Report", "lists devices that stopped reporting", "find the sensors that went silent"),
    ("Map Clustering", "groups nearby devices on the map view", "merge pins that are close together on the map"),
]
LIMITS = [("maximum number of rows per data export", _r(100_000, 1_000_001, 50_000)),
          ("maximum number of widgets per dashboard", _r(24, 97, 4)),
          ("maximum number of alert rules per workspace", _r(200, 2001, 100)),
          ("default API page size", [50, 100, 150, 200, 250, 500]),
          ("number of webhook retry attempts", _r(3, 13)),
          ("maximum number of recipients per scheduled report", _r(10, 101, 5)),
          ("maximum number of saved filters per user", _r(20, 201, 10)),
          ("number of map markers shown before clustering starts", _r(200, 2001, 100)),
          ("minimum dashboard auto-refresh interval in seconds", [5, 10, 15, 20, 30]),
          ("maximum number of custom fields per device", _r(10, 81, 5))]
BUGS = ["CSV exports dropped the last row", "time zones were ignored in scheduled reports",
        "the map froze with more than 5,000 devices", "SSO logins failed for users with long e-mail addresses",
        "battery warnings were sent twice", "deleted dashboards still appeared in search",
        "webhooks were not retried after a timeout", "chart tooltips showed the wrong unit",
        "device names with accents broke the PDF report", "the API returned 500 for empty filters"]

OFFICES = [("lisbon", "Lisbon", "eu"), ("rotterdam", "Rotterdam", "eu"), ("austin", "Austin", "us"),
           ("leeds", "Leeds", "uk"), ("singapore", "Singapore", "apac")]


# ------------------------------------------------------------------ builder


class _Builder:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.docs: list[Doc] = []
        self.candidates: list[dict] = []

    def distinct(self, pool: list, n: int) -> list:
        if len(pool) < n:
            raise ValueError(f"pool too small: need {n}, have {len(pool)}")
        return self.rng.sample(pool, n)

    def cand(self, qtype, question, doc, attr, fact, evidence):
        self.candidates.append({"qtype": qtype, "question": question, "doc_id": doc.doc_id, "family": doc.family,
                                "attribute": attr, "fact": str(fact), "evidence": evidence})

    # -- regional policies
    def regional(self):
        regions = list(REGIONS)
        for fam, spec in REGIONAL.items():
            values = {}
            for a in spec["attrs"]:
                if a.fixed:
                    values[a.key] = dict(a.fixed)
                else:
                    values[a.key] = dict(zip(regions, self.distinct(a.pool, len(regions))))
            for r in regions:
                reg = REGIONS[r]
                R = reg["name"]
                doc_id = f"policy-{fam}-{r}"
                scope_txt = ("This policy applies to employees of {e} in {c}." if spec["scope"] == "staff"
                             else "This policy applies to customers whose contract is with {e} ({c}).")
                sections: dict[str, list[str]] = {"Scope": [scope_txt.format(e=reg["entity"], c=reg["countries"])]}
                evid = {}
                for a in spec["attrs"]:
                    v = values[a.key][r]
                    sent = a.sentence.format(v=v, cur=reg["cur"])
                    sections.setdefault(a.section, []).append(sent)
                    evid[a.key] = (v, sent)
                filler = list(spec["filler"])
                keys = [k for k in sections if k != "Scope"]
                for i, f in enumerate(filler):
                    sections[keys[(i * 2 + 1) % len(keys)]].append(f)
                doc = Doc(doc_id, spec["title"].format(R=R), f"policy-{fam}", R,
                          [(h, " ".join(s)) for h, s in sections.items()])
                self.docs.append(doc)
                prefix = values["prefix"][r] if "prefix" in values else None
                for a in spec["attrs"]:
                    v, sent = evid[a.key]
                    fact = a.fact.format(v=v)
                    if a.kw:
                        self.cand(a.kw_type, a.kw.format(R=R, P=prefix), doc, a.key, fact, sent)
                    if a.para:
                        scope = self.rng.choice(reg[spec["scope"]])
                        self.cand("paraphrase", a.para.format(scope=scope), doc, a.key, fact, sent)

    # -- product spec sheets
    def specs(self):
        n = len(LINES) * len(TIERS)
        fam_vals = {a.key: self.distinct(a.pool, n) for a in SPEC_ATTRS if a.key not in ("ip", "warranty", "temp")}
        temp_hi = self.distinct(_r(40, 100, 5), n)
        temp_lo = self.distinct(SPEC_ATTRS[4].pool, len(SPEC_ATTRS[4].pool))
        i = 0
        for lkey, lname, code, ldesc, lwhat in LINES:
            ip = self.distinct(SPEC_ATTRS[3].pool, len(TIERS))
            war = self.distinct(SPEC_ATTRS[6].pool, len(TIERS))
            for t, (tkey, tname, tnum, tdesc) in enumerate(TIERS):
                sku = f"QV-{code}{tnum}"
                doc = Doc(f"spec-{lkey}-{tkey}", f"Quorvia {lname} {tname} ({sku}) specification sheet", "spec",
                          f"{lname} {tname}", [])
                vals = {k: fam_vals[k][i] for k in fam_vals}
                vals["ip"], vals["warranty"] = ip[t], war[t]
                vals["temp"], w = temp_lo[i % len(temp_lo)], temp_hi[i]
                sections: dict[str, list[str]] = {
                    "Overview": [f"{lname} is the Quorvia {ldesc} line; it {lwhat}. The {tname} model ({sku}) is the "
                                 f"{tdesc} option of the {lname} line."]}
                evid = {}
                for a in SPEC_ATTRS:
                    sent = a.sentence.format(v=vals[a.key], w=w)
                    sections.setdefault(a.section, []).append(sent)
                    evid[a.key] = (a.fact.format(v=vals[a.key], w=w), sent)
                sections["Measurement"].append("Readings are buffered on the device for 72 hours when the gateway is offline.")
                sections["Pricing"].append("Volume discounts are available from 50 units.")
                doc.sections = [(h, " ".join(s)) for h, s in sections.items()]
                self.docs.append(doc)
                for a in SPEC_ATTRS:
                    fact, sent = evid[a.key]
                    self.cand(a.kw_type, a.kw.format(SKU=sku), doc, a.key, fact, sent)
                    self.cand("paraphrase", a.para.format(tierdesc=tdesc, linedesc=ldesc), doc, a.key, fact, sent)
                i += 1

    # -- pricing pages
    def pricing(self):
        regions = list(REGIONS)
        n = len(PLANS) * len(regions)
        fam_vals = {a.key: self.distinct(a.pool, n) for a in PRICING_ATTRS}
        i = 0
        for pkey, pname, pwhat, pdesc in PLANS:
            for r in regions:
                reg = REGIONS[r]
                R = reg["name"]
                doc = Doc(f"pricing-{pkey}-{r}", f"Quorvia Cloud {pname} plan pricing ({R})", "pricing", f"{pname} {R}", [])
                sections: dict[str, list[str]] = {
                    "Overview": [f"The {pname} plan is {pwhat}. Prices on this page apply to customers billed by "
                                 f"{reg['entity']} and are shown in {reg['cur']} excluding tax."]}
                evid = {}
                for a in PRICING_ATTRS:
                    v = fam_vals[a.key][i]
                    sent = a.sentence.format(v=v, cur=reg["cur"], plan=pname)
                    sections.setdefault(a.section, []).append(sent)
                    evid[a.key] = (v, sent)
                sections["Annual billing"].append("Annual plans are invoiced up front and renew automatically.")
                doc.sections = [(h, " ".join(s)) for h, s in sections.items()]
                self.docs.append(doc)
                for a in PRICING_ATTRS:
                    v, sent = evid[a.key]
                    self.cand("lexical", a.kw.format(plan=pname, R="the " + R if r != "apac" else "APAC"), doc, a.key, v, sent)
                    scope = self.rng.choice(reg["customers"])
                    plandesc = {"smallest": "cheapest", "middle": "mid-tier", "largest": "top"}[pdesc]
                    self.cand("paraphrase", a.para.format(scope=scope, plandesc=plandesc), doc, a.key, v, sent)
                i += 1

    # -- SLA tiers
    def sla(self):
        rows = [("starter", "Starter", "99.5", 240, 2, "business days from 09:00 to 17:00 CET"),
                ("team", "Team", "99.9", 60, 5, "business days from 07:00 to 22:00 CET"),
                ("enterprise", "Enterprise", "99.95", 15, 10, "24 hours a day, every day")]
        for key, name, up, p1, credit, hours in rows:
            sents = {"uptime": f"The monthly uptime commitment for the {name} tier is {up}%.",
                     "p1": f"Priority 1 incidents on the {name} tier get a first response within {p1} minutes.",
                     "credit": f"Each 0.1% of uptime below the commitment earns a service credit of {credit}% of the "
                               f"monthly fee, capped at 30%."}
            doc = Doc(f"sla-{key}", f"Service Level Agreement: {name} tier", "sla", name, [
                ("Uptime", sents["uptime"] + " Scheduled maintenance announced 72 hours ahead is excluded."),
                ("Support", sents["p1"] + f" Support is staffed on {hours}."),
                ("Service credits", sents["credit"] + " Credits are requested through the support portal within 30 days.")])
            self.docs.append(doc)
            self.cand("lexical", f"What is the monthly uptime commitment of the {name} SLA tier?", doc, "uptime", up, sents["uptime"])
            level = {"starter": "basic", "team": "mid-level", "enterprise": "highest"}[key]
            self.cand("paraphrase", f"How quickly will support first answer a critical outage for a customer on the "
                      f"{level} service level?", doc, "p1", p1, sents["p1"])
            self.cand("lexical", f"What service credit is earned per 0.1% below the commitment on the {name} SLA tier?",
                      doc, "credit", credit, sents["credit"])

    # -- runbooks
    def runbooks(self):
        codes = self.distinct(_r(1000, 9999), len(SERVICES))
        thresholds = self.distinct(_r(20, 980, 10), len(SERVICES))
        esc = self.distinct([10, 15, 20, 25, 30, 35, 40, 45, 50, 60], len(SERVICES))
        for i, (svc, pfx, what, symptom, para_symptom, verb, metric) in enumerate(SERVICES):
            code = f"{pfx}-{codes[i]}"
            cmd = f"qvctl {verb} {svc}"
            s_err = f"Error {code} means {symptom}. First step: run {cmd} and watch the service dashboard for five minutes."
            s_thr = f"Page the on-call engineer when the {metric} stays above {thresholds[i]} for 10 minutes."
            s_esc = f"If the issue is not resolved within {esc[i]} minutes, escalate to the {TEAMS[i]} team."
            doc = Doc(f"runbook-{svc}", f"Runbook: {svc}", "runbook", svc, [
                ("Purpose", f"The {svc} service {what}. This runbook is for the on-call engineer."),
                ("Alerts", s_thr + " Alerts are routed to the #oncall-platform channel."),
                ("Known errors", s_err + " Check recent deploys before restarting anything."),
                ("Escalation", s_esc + " Record every action in the incident timeline.")])
            self.docs.append(doc)
            self.cand("exact_code", f"What should on-call run first when error {code} appears?", doc, "error", cmd, s_err)
            self.cand("paraphrase", f"What is the first command to run when {para_symptom}?", doc, "error", cmd, s_err)
            self.cand("lexical", f"At what {metric} does the {svc} runbook page the on-call engineer?", doc,
                      "threshold", thresholds[i], s_thr)
            self.cand("lexical", f"After how many minutes should an unresolved {svc} incident be escalated?", doc,
                      "escalation", esc[i], s_esc)

    # -- release notes
    def releases(self):
        feats = self.distinct(FEATURES, 10)
        limits = self.distinct(LIMITS, 10)
        tickets = self.distinct(_r(1000, 9999), 10)
        bugs = self.distinct(BUGS, 10)
        for i in range(10):
            ver = f"7.{i}"
            fname, fwhat, fpara = feats[i]
            lname, lpool = limits[i]
            lval = self.rng.choice(lpool)
            t = f"QC-{tickets[i]}"
            s_feat = f"New: {fname}, which {fwhat}."
            s_lim = f"The {lname} was raised to {lval:,}."
            s_fix = f"Fixed {t}: {bugs[i]}."
            doc = Doc(f"release-7-{i}", f"Quorvia Cloud {ver} release notes", "release-notes", ver, [
                ("Highlights", f"Quorvia Cloud {ver} is now available to all workspaces. " + s_feat),
                ("Limits", s_lim + " Existing workspaces get the new limit automatically."),
                ("Fixes", s_fix + " Minor performance improvements across the web app.")])
            self.docs.append(doc)
            self.cand("exact_code", f"Which Quorvia Cloud release fixed {t}?", doc, "fix", ver, s_fix)
            self.cand("exact_code", f"What new feature shipped in Quorvia Cloud {ver}?", doc, "feature", fname, s_feat)
            self.cand("paraphrase", f"Which Quorvia Cloud release made it possible to {fpara}?", doc, "feature_version", ver, s_feat)
            self.cand("lexical", f"What is the {lname} after it was raised?", doc, "limit", f"{lval:,}", s_lim)

    # -- office guides
    def offices(self):
        exts = self.distinct(_r(2000, 9000), len(OFFICES))
        visitors = self.distinct([12, 24, 36, 48, 72], len(OFFICES))
        bikes = self.distinct(_r(10, 80, 5), len(OFFICES))
        for i, (key, city, r) in enumerate(OFFICES):
            wifi = f"qv-guest-{key[:3]}"
            s_wifi = f"The guest Wi-Fi network is called {wifi}; the password is printed at the front desk."
            s_ext = f"The front desk can be reached on extension {exts[i]}."
            s_vis = f"Visitors must be registered at least {visitors[i]} hours in advance."
            secs = [("About", f"The {city} office is one of the {REGIONS[r]['name']} offices of Quorvia Instruments."),
                    ("Getting in", s_vis + " Employees use their badge at the main entrance. " + s_ext),
                    ("Facilities", s_wifi)]
            if key != "singapore":  # deliberately missing: basis for an unanswerable question
                s_bike = f"There are {bikes[i]} bike parking spaces in the basement."
                secs[2] = ("Facilities", s_wifi + " " + s_bike)
            doc = Doc(f"office-{key}", f"Office guide: {city}", "office", city, secs)
            self.docs.append(doc)
            self.cand("lexical", f"What is the guest Wi-Fi network name at the {city} office?", doc, "wifi", wifi, s_wifi)
            self.cand("lexical", f"What is the front desk extension at the {city} office?", doc, "ext", exts[i], s_ext)
            self.cand("paraphrase", f"How much notice do I need to give before bringing a guest to the {city} site?",
                      doc, "visitors", visitors[i], s_vis)
            if key != "singapore":
                self.cand("paraphrase", f"Where can I leave my bicycle at the {city} office, and how many spots are there?",
                          doc, "bikes", bikes[i], s_bike)

    # -- one-off documents
    def misc(self):
        rng = self.rng
        key_days = rng.choice([60, 90, 120, 180])
        pw_len = rng.choice([12, 14, 16])
        sev = {1: rng.choice([15, 30]), 2: rng.choice([45, 60]), 3: rng.choice([120, 240]), 4: 1440}
        rate = {"Starter": rng.choice([60, 120]), "Team": rng.choice([300, 600]), "Enterprise": rng.choice([1200, 3000])}
        cal = dict(zip(["Pulse", "Tide", "Ember", "Vane"], self.distinct([90, 120, 180, 270, 365, 540], 4)))
        quotes = rng.choice([5000, 10000, 15000])
        cfo = rng.choice([50000, 75000, 100000])
        train = rng.choice([7, 14, 21, 30])
        fw = rng.choice(["3.8", "4.1", "4.4"])
        hub = rng.choice([16, 24, 32, 48])
        mount = rng.choice(["1.2", "1.5", "1.8"])
        brand = rng.choice(["#1F6F5C", "#2E5E8C", "#7A3E9D"])
        allhands = rng.choice(["first Tuesday", "second Wednesday", "last Thursday"])
        offsite = rng.choice(["September", "October", "May"])
        spec = [
            ("security-overview", "Security overview", [
                ("Encryption", "Customer data at rest is encrypted with AES-256 and data in transit uses TLS 1.2 or later. "
                               f"Encryption keys are rotated every {key_days} days."),
                ("Assurance", "Customers can request the SOC 2 Type II report through their account manager.")],
             [("lexical", "How often are encryption keys rotated?", "keys", key_days, f"Encryption keys are rotated every {key_days} days."),
              ("paraphrase", "How frequently do we swap out the cryptographic secrets that protect stored data?", "keys2", key_days,
               f"Encryption keys are rotated every {key_days} days.")]),
            ("password-policy", "Password and MFA policy", [
                ("Passwords", f"Passwords for company accounts must be at least {pw_len} characters long. Passwords are not "
                              "rotated on a schedule; they are changed after a suspected compromise."),
                ("MFA", "Multi-factor authentication with a hardware key or authenticator app is mandatory for all staff.")],
             [("lexical", "What is the minimum password length for company accounts?", "pw", pw_len,
               f"Passwords for company accounts must be at least {pw_len} characters long.")]),
            ("incident-severity", "Incident severity levels", [
                ("SEV1", f"SEV1 means a full outage for many customers. During a SEV1, status updates are posted every {sev[1]} minutes."),
                ("SEV2", f"SEV2 means a major feature is degraded. During a SEV2, status updates are posted every {sev[2]} minutes."),
                ("SEV3 and SEV4", f"SEV3 means a minor issue with a workaround; updates every {sev[3]} minutes. SEV4 is cosmetic; "
                                  f"updates every {sev[4]} minutes.")],
             [("exact_code", "How often are status updates posted during a SEV2 incident?", "sev2", sev[2],
               f"During a SEV2, status updates are posted every {sev[2]} minutes."),
              ("paraphrase", "During a total outage hitting lots of customers, how often do we publish a status update?", "sev1",
               sev[1], f"During a SEV1, status updates are posted every {sev[1]} minutes.")]),
            ("api-rate-limits", "Quorvia Cloud API rate limits", [
                ("Limits", f"The API allows {rate['Starter']} requests per minute on the Starter plan, {rate['Team']} on the Team "
                           f"plan and {rate['Enterprise']} on the Enterprise plan."),
                ("Errors", "Requests above the limit receive HTTP 429 with a Retry-After header.")],
             [("lexical", "What is the API rate limit on the Team plan?", "rate_team", rate["Team"],
               f"The API allows {rate['Starter']} requests per minute on the Starter plan, {rate['Team']} on the Team plan")]),
            ("calibration-procedure", "Sensor calibration procedure", [
                ("Intervals", f"Pulse sensors are calibrated every {cal['Pulse']} days, Tide sensors every {cal['Tide']} days, "
                              f"Ember loggers every {cal['Ember']} days and Vane sensors every {cal['Vane']} days."),
                ("Procedure", "Calibration is done with the reference kit and recorded in the device history.")],
             [("paraphrase", "How often do the temperature loggers need to be recalibrated?", "cal_ember", cal["Ember"],
               f"Ember loggers every {cal['Ember']} days"),
              ("lexical", "How often are Tide sensors calibrated?", "cal_tide", cal["Tide"], f"Tide sensors every {cal['Tide']} days")]),
            ("procurement-policy", "Procurement policy", [
                ("Quotes", f"Purchases above EUR {quotes:,} need three competing quotes."),
                ("Approval", f"Purchase orders above EUR {cfo:,} need approval from the CFO.")],
             [("lexical", "Above what amount do purchases need three competing quotes?", "quotes", f"{quotes:,}",
               f"Purchases above EUR {quotes:,} need three competing quotes."),
              ("paraphrase", "From what amount does a purchase order need the finance chief to sign it off?", "cfo", f"{cfo:,}",
               f"Purchase orders above EUR {cfo:,} need approval from the CFO.")]),
            ("onboarding-checklist", "New joiner onboarding checklist", [
                ("First day", "Laptops and badges are handed out on the first morning."),
                ("First weeks", f"Security awareness training must be completed within {train} days of the start date.")],
             [("paraphrase", "How long do new joiners have to finish the security course?", "training", train,
               f"Security awareness training must be completed within {train} days of the start date.")]),
            ("firmware-compatibility", "Firmware compatibility", [
                ("Minimum versions", f"Over-the-air updates require firmware {fw} or later on the device."),
                ("Support", "Devices more than two major firmware versions behind are not supported.")],
             [("lexical", "What is the minimum firmware version for over-the-air updates?", "fw", fw,
               f"Over-the-air updates require firmware {fw} or later on the device.")]),
            ("installation-guide", "Device installation guide", [
                ("Placement", f"Mount Pulse sensors {mount} metres above the floor, away from windows and vents."),
                ("Gateways", f"One gateway hub connects up to {hub} sensors.")],
             [("paraphrase", "At what height should the air-quality sensors be mounted on the wall?", "mount", mount,
               f"Mount Pulse sensors {mount} metres above the floor, away from windows and vents."),
              ("lexical", "How many sensors can one gateway hub connect?", "hub", hub, f"One gateway hub connects up to {hub} sensors.")]),
            ("brand-guidelines", "Brand guidelines", [
                ("Colour", f"The primary brand colour is {brand}."),
                ("Name", "Always write Quorvia with a capital Q and never abbreviate it.")],
             [("lexical", "What is the hex code of the primary brand colour?", "brand", brand.lstrip("#"),
               f"The primary brand colour is {brand}.")]),
            ("company-rituals", "Company rituals", [
                ("All-hands", f"The company all-hands meeting is held on the {allhands} of every month."),
                ("Offsite", f"The annual offsite takes place in {offsite}.")],
             [("paraphrase", "In which month do we all travel together for the yearly company retreat?", "offsite", offsite,
               f"The annual offsite takes place in {offsite}.")]),
            ("glossary", "Glossary", [
                ("Terms", "A workspace is the top-level container for devices, dashboards and users. A site is a physical "
                          "location inside a workspace. A gateway hub relays sensor readings to Quorvia Cloud.")],
             [("lexical", "In the glossary, what is a site?", "site", "physical location", "A site is a physical location inside a workspace.")]),
        ]
        for doc_id, title, sections, qs in spec:
            doc = Doc(doc_id, title, "misc", title, sections)
            self.docs.append(doc)
            for qtype, q, attr, fact, ev in qs:
                self.cand(qtype, q, doc, attr, fact, ev)


UNANSWERABLE = [
    "What is the meal per diem for employees in Brazil?",
    "How many weeks of parental leave do employees in Canada get?",
    "What is the hotel cap in the LATAM travel policy?",
    "What is the home office stipend for staff in India?",
    "What is the battery life of the QV-PU400?",
    "What is the IP rating of the QV-VN250?",
    "What is the list price of the QV-TD150?",
    "What does error GEO-4417 mean in the geo-indexer runbook?",
    "What should on-call run first when search-indexer reports error SRC-2210?",
    "What changed in Quorvia Cloud 8.1?",
    "Which Quorvia Cloud release fixed QC-0042?",
    "How much does the Business plan cost per user per month in the US?",
    "What is the monthly uptime commitment of the Premium SLA tier?",
    "How many bike parking spaces does the Singapore office have?",
    "What is the guest Wi-Fi network name at the Berlin office?",
    "Which dental insurance provider covers UK employees?",
    "What is the vesting schedule for employee stock options?",
    "What is the maximum number of API keys per workspace?",
    "How many days of sabbatical leave do employees get after five years?",
    "What is the CO2 measurement accuracy of the QV-PU200?",
    "What is the maximum altitude at which the QV-VN300 can operate?",
    "What was the maximum number of rows per data export in Quorvia Cloud 6.2?",
    "Which SMS provider does notification-hub use?",
    "Which team owns the data-lake service?",
    "What is the parking fee at the Austin office?",
]

TARGETS = {"exact_code": 35, "lexical": 35, "paraphrase": 40}


def _select(candidates: list[dict], rng: random.Random) -> list[dict]:
    """Pick TARGETS questions per type, round-robin over families, one question per (doc, attribute)."""
    used: set[tuple[str, str]] = set()
    chosen: list[dict] = []
    for qtype, n in TARGETS.items():
        pool = [c for c in candidates if c["qtype"] == qtype]
        rng.shuffle(pool)
        by_fam: dict[str, list[dict]] = {}
        for c in pool:
            by_fam.setdefault(c["family"], []).append(c)
        fams = sorted(by_fam)
        rng.shuffle(fams)
        taken = 0
        while taken < n and any(by_fam[f] for f in fams):
            for f in fams:
                while by_fam[f]:
                    c = by_fam[f].pop()
                    key = (c["doc_id"], c["attribute"].rstrip("2"))
                    if key not in used:
                        used.add(key)
                        chosen.append(c)
                        taken += 1
                        break
                if taken >= n:
                    break
    return chosen


def generate(seed: int = 42) -> tuple[list[Doc], list[Question]]:
    b = _Builder(seed)
    b.regional()
    b.specs()
    b.pricing()
    b.sla()
    b.runbooks()
    b.releases()
    b.offices()
    b.misc()
    families: dict[str, list[str]] = {}
    for d in b.docs:
        families.setdefault(d.family, []).append(d.doc_id)

    rng = random.Random(seed + 1)
    picked = _select(b.candidates, rng)
    questions: list[Question] = []
    for c in picked:
        distractors = [] if c["family"] == "misc" else [x for x in families[c["family"]] if x != c["doc_id"]]
        questions.append(Question("", c["question"], c["qtype"], c["family"], c["doc_id"], c["fact"], c["evidence"],
                                  c["attribute"], distractors))
    for q in UNANSWERABLE:
        questions.append(Question("", q, "unanswerable", None, None, None, None))
    rng.shuffle(questions)
    for i, q in enumerate(questions, 1):
        q.qid = f"q{i:03d}"
    return b.docs, questions


def write_dataset(data_dir: str | Path, seed: int = 42) -> dict:
    docs, questions = generate(seed)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    with open(data_dir / "questions.jsonl", "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.qtype] = counts.get(q.qtype, 0) + 1
    return {"documents": len(docs), "questions": len(questions), "by_type": counts,
            "families": sorted({d.family for d in docs})}
