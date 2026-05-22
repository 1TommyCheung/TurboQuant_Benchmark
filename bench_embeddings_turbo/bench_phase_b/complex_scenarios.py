"""Phase B+ — Four complex multi-turn scenarios designed to stress the
retrieval backend in ways the original pi-session replay does NOT.

Each scenario is a multi-turn task where the agent must:
  - cross-reference evidence across documents/dates (multi-hop fact-tracing)
  - reconstruct date-anchored chronologies (recall + ordering)
  - filter privileged/WP content for court output (metadata-aware retrieval)
  - correct wrong premises pushed by the user (refinement under pressure)

Designed so that retrieval quality is the binding factor: a backend that
embeds Singapore-legal text poorly will visibly fail.
"""
from __future__ import annotations
from typing import TypedDict


class Turn(TypedDict):
    idx: int
    user_text: str
    needs_search: bool
    skill: str  # tag for analysis: "multi_hop" | "timeline" | "privilege" | "correction"


class Scenario(TypedDict):
    id: str
    title: str
    skill: str
    description: str
    turns: list[Turn]


# ─────────────────────────────────────────────────────────────────────────
# Scenario 1: Multi-hop fact-tracing — counselling refusal claim
# ─────────────────────────────────────────────────────────────────────────
SCENARIO_MULTI_HOP: Scenario = {
    "id": "scenario_1_multi_hop",
    "title": "Multi-hop fact-tracing: Respondent's counselling-refusal claim",
    "skill": "multi_hop",
    "description": (
        "User pushes the agent to trace a specific contested claim across "
        "multiple correspondence dates and parties, ending with a fact-saved "
        "rebuttal. Requires retrieving (a) the Respondent's L&L 6 Jan 2026 "
        "letter claim, (b) Tommy's own contemporaneous emails about counselling, "
        "(c) GJC's responses, (d) any third-party (counsellor/MontfortCare) "
        "records, and synthesising them."
    ),
    "turns": [
        {"idx": 1, "user_text": "Tracy's L&L letter on 6 Jan 2026 claims I refused all counselling sessions. Pull her exact wording on that — I need the quote.", "needs_search": True, "skill": "multi_hop"},
        {"idx": 2, "user_text": "Now find what I actually said to GJC about counselling between November and December 2025. I want my real position in my own words.", "needs_search": True, "skill": "multi_hop"},
        {"idx": 3, "user_text": "Did GJC respond in writing to her 6 Jan claim? I'm looking for a solicitor letter from us between January and February 2026 that pushes back specifically on the counselling angle.", "needs_search": True, "skill": "multi_hop"},
        {"idx": 4, "user_text": "What about MontfortCare or any counsellor — do we have any third-party records of sessions actually happening or being declined?", "needs_search": True, "skill": "multi_hop"},
        {"idx": 5, "user_text": "Pull this together as a single rebuttal paragraph for the affidavit. Cite chunk_ids inline.", "needs_search": False, "skill": "multi_hop"},
        {"idx": 6, "user_text": "Good. Save the key facts (her claim, my actual position, MontfortCare evidence) so this stops being a vulnerable point going forward.", "needs_search": False, "skill": "multi_hop"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────
# Scenario 2: Date-anchored timeline reconstruction — access incidents
# ─────────────────────────────────────────────────────────────────────────
SCENARIO_TIMELINE: Scenario = {
    "id": "scenario_2_timeline",
    "title": "Date-anchored timeline: every access incident Jun-Oct 2025",
    "skill": "timeline",
    "description": (
        "Forces recall + chronology over a specific 5-month window. Backend "
        "must surface all access-related events in the window. Tests whether "
        "vector retrieval misses incidents that BM25 keyword search would "
        "catch (or vice versa). Then asks for documentary trail per incident."
    ),
    "turns": [
        {"idx": 1, "user_text": "Walk me through every access incident with the children between June 2025 and October 2025. I want a chronological list with dates, what happened, and who was involved.", "needs_search": True, "skill": "timeline"},
        {"idx": 2, "user_text": "For each incident in your list, was Lee & Lee notified at the time? Show me the corresponding solicitor correspondence — date and chunk_id.", "needs_search": True, "skill": "timeline"},
        {"idx": 3, "user_text": "Did any of these incidents become the subject of a separate summons or a court application? I'm thinking summons for variation or contempt — check.", "needs_search": True, "skill": "timeline"},
        {"idx": 4, "user_text": "Tabulate it: column 1 date, column 2 what happened, column 3 our solicitor's response, column 4 Lee & Lee's response if any, column 5 outcome.", "needs_search": False, "skill": "timeline"},
        {"idx": 5, "user_text": "Which 3 incidents in your table have the strongest documentary trail? I want to lead with those in the affidavit.", "needs_search": False, "skill": "timeline"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────
# Scenario 3: Privilege-sensitive court drafting — matrimonial valuation
# ─────────────────────────────────────────────────────────────────────────
SCENARIO_PRIVILEGE: Scenario = {
    "id": "scenario_3_privilege",
    "title": "Privilege-aware drafting: matrimonial home valuation paragraph",
    "skill": "privilege",
    "description": (
        "Tests whether the backend surfaces non-privileged content suitable "
        "for court filing. Privileged GJC strategy memos and WP correspondence "
        "must NOT enter the final draft. Backend must distinguish source "
        "categories (court_doc, document_exchange, public records) from "
        "privileged GJC emails."
    ),
    "turns": [
        {"idx": 1, "user_text": "Find all GJC correspondence and internal notes about the matrimonial home at 31 Alexandra Road — valuation, strategy, anything we've discussed with counsel.", "needs_search": True, "skill": "privilege"},
        {"idx": 2, "user_text": "Now find any non-privileged sources for the valuation: third-party valuer reports, IRAS or HDB records, public listings, anything that's filed or in the open record.", "needs_search": True, "skill": "privilege"},
        {"idx": 3, "user_text": "What's the safest evidentiary anchor that is neither privileged nor WP? I want one source we can stake the valuation paragraph on.", "needs_search": True, "skill": "privilege"},
        {"idx": 4, "user_text": "Draft a 200-word court-ready paragraph supporting our position on the matrimonial home value. Cite only non-privileged evidence.", "needs_search": False, "skill": "privilege"},
        {"idx": 5, "user_text": "Cross-check your draft — does it reference any chunk where is_privileged or is_wp is true? If yes, replace those citations.", "needs_search": True, "skill": "privilege"},
        {"idx": 6, "user_text": "OK looks clean. Save this draft paragraph and the safe-citations list as a verified fact so we lock it in.", "needs_search": False, "skill": "privilege"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────
# Scenario 4: Fact-correction loop — wrong-date pushback
# ─────────────────────────────────────────────────────────────────────────
SCENARIO_CORRECTION: Scenario = {
    "id": "scenario_4_correction",
    "title": "Fact-correction loop: when did indirect-contribution claim start",
    "skill": "correction",
    "description": (
        "Tests refinement-under-pressure: user states a wrong date, then "
        "corrects it, forcing the agent to chase the right anchor. Realistic "
        "user behaviour — recalls 'something happened around March' but "
        "actually it was earlier. Backend must support multiple refinement "
        "searches with shifting date filters."
    ),
    "turns": [
        {"idx": 1, "user_text": "When did Lee & Lee first raise the indirect-contribution argument? I think it was around March 2025 — find me the letter or affidavit.", "needs_search": True, "skill": "correction"},
        {"idx": 2, "user_text": "No, that's wrong — March 2025 is just the response. The first time she raised indirect contribution was in her reply affidavit. When was that filed?", "needs_search": True, "skill": "correction"},
        {"idx": 3, "user_text": "Cross-check that with my own October 2025 Request for Further and Better Particulars — what did I demand on indirect contribution, and what did she say in reply?", "needs_search": True, "skill": "correction"},
        {"idx": 4, "user_text": "Show me her verbatim words on indirect contribution — I want the exact phrasing from her reply affidavit, not paraphrased.", "needs_search": True, "skill": "correction"},
        {"idx": 5, "user_text": "OK so the correct anchor is her reply affidavit, not anything in March 2025. Update our timeline fact for indirect-contribution to use that earlier date and save it.", "needs_search": False, "skill": "correction"},
    ],
}


ALL_SCENARIOS: list[Scenario] = [
    SCENARIO_MULTI_HOP,
    SCENARIO_TIMELINE,
    SCENARIO_PRIVILEGE,
    SCENARIO_CORRECTION,
]
