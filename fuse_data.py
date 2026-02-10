"""
fuse_data.py - Elden Ring Data Fusion & Enrichment Pipeline
============================================================
Loads all CSVs + lore JSON, parses nested structures, builds
cross-reference indexes, and outputs elden_ring_enriched.json.

Pipeline: extract_lore.py → master_lore.json
                                   ↓
          fuse_data.py     → elden_ring_enriched.json
                                   ↓
          generate_qa.py   → elden_ring_final_train.jsonl
"""

import os
import json
import ast
import re
import pandas as pd
from difflib import get_close_matches
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "data"
LORE_PATH = "master_lore.json"
OUTPUT_PATH = "elden_ring_enriched.json"
FUZZY_CUTOFF = 0.75

CSV_FILES = {
    "weapons": "weapons.csv",
    "weapons_stats": "elden_ring_weapon.csv",
    "boss_stats": "elden_ring_boss_stats_clean.csv",
    "bosses": "bosses.csv",
    "armors": "armors.csv",
    "incantations": "incantations.csv",
    "sorceries": "sorceries.csv",
    "npcs": "npcs.csv",
    "locations": "locations.csv",
    "creatures": "creatures.csv",
    "ashes_of_war": "ashesOfWar.csv",
    "skills": "skills.csv",
}

# ============================================================
# UTILITIES
# ============================================================
def safe_literal_eval(val):
    """Safely parse string-encoded dicts/lists."""
    if pd.isna(val):
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return str(val)

def norm(name):
    """Normalize entity name for matching."""
    if pd.isna(name):
        return ""
    return re.sub(r'\s+', ' ', str(name).lower().strip())

def fuzzy_get(lookup_dict, key, cutoff=FUZZY_CUTOFF):
    """Fuzzy match a key against a dict's keys."""
    k = norm(key)
    if k in lookup_dict:
        return lookup_dict[k]
    matches = get_close_matches(k, lookup_dict.keys(), n=1, cutoff=cutoff)
    return lookup_dict[matches[0]] if matches else None

def safe_str(val, default="Unknown"):
    """Convert to string, handling NaN."""
    if pd.isna(val):
        return default
    return str(val).strip()

def safe_float(val, default=None):
    """Convert to float, handling commas and NaN."""
    if pd.isna(val):
        return default
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return default

def parse_list_col(val):
    """Parse a string-encoded list column."""
    if pd.isna(val):
        return []
    parsed = safe_literal_eval(val)
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if pd.notna(x)]
    return [str(parsed).strip()] if parsed else []

def load_csv(name):
    """Load a CSV from data dir, return df or None."""
    path = os.path.join(DATA_DIR, CSV_FILES[name])
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping {name}")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {name}: {len(df)} rows")
    return df

# ============================================================
# STEP 1: LOAD ALL DATA
# ============================================================
def load_all_data():
    print("=" * 60)
    print("STEP 1: Loading all data sources")
    print("=" * 60)

    data = {}
    for key in CSV_FILES:
        data[key] = load_csv(key)

    # Load lore library
    lore_lib = {}
    if os.path.exists(LORE_PATH):
        with open(LORE_PATH, "r", encoding="utf-8") as f:
            raw_lore = json.load(f)
        # Normalize keys
        lore_lib = {norm(k): v for k, v in raw_lore.items()}
        print(f"  Loaded lore library: {len(lore_lib)} entries")
    else:
        print(f"  WARNING: {LORE_PATH} not found, proceeding without lore")

    return data, lore_lib

# ============================================================
# STEP 2a: PARSE & ENRICH WEAPONS
# ============================================================
def enrich_weapons(data, lore_lib):
    print("\n" + "=" * 60)
    print("STEP 2a: Enriching weapons")
    print("=" * 60)

    df = data["weapons"]
    if df is None:
        return []

    # Build scaling lookup from elden_ring_weapon.csv
    scaling_lookup = {}
    dmg_lookup = {}
    if data["weapons_stats"] is not None:
        ws = data["weapons_stats"]
        for _, r in ws.iterrows():
            n = norm(r.get("Name", ""))
            scaling_lookup[n] = {
                "Str": safe_str(r.get("Str"), "-"),
                "Dex": safe_str(r.get("Dex"), "-"),
                "Int": safe_str(r.get("Int"), "-"),
                "Fai": safe_str(r.get("Fai"), "-"),
                "Arc": safe_str(r.get("Arc"), "-"),
            }
            dmg_lookup[n] = {
                "Phy": safe_str(r.get("Phy"), "0"),
                "Mag": safe_str(r.get("Mag"), "0"),
                "Fir": safe_str(r.get("Fir"), "0"),
                "Lit": safe_str(r.get("Lit"), "0"),
                "Hol": safe_str(r.get("Hol"), "0"),
            }
        print(f"  Built scaling lookup: {len(scaling_lookup)} weapons")

    weapons = []
    for _, row in df.iterrows():
        name = safe_str(row.get("name"), "Unknown Weapon")
        n = norm(name)

        # Parse requirements dict
        reqs = safe_literal_eval(row.get("requirements"))
        if not isinstance(reqs, dict):
            reqs = {}

        # Fuzzy match lore
        lore = fuzzy_get(lore_lib, name) or ""

        # Fuzzy match scaling + base damage from stats CSV
        scaling = fuzzy_get(scaling_lookup, name) or {}
        base_dmg = fuzzy_get(dmg_lookup, name) or {}

        # Determine primary scaling stat
        grade_order = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "-": 0}
        primary_scaling = "None"
        if scaling:
            best = max(scaling.items(), key=lambda x: grade_order.get(x[1], 0))
            if grade_order.get(best[1], 0) > 0:
                primary_scaling = best[0]

        weapons.append({
            "name": name,
            "description": safe_str(row.get("description"), ""),
            "lore": lore,
            "category": safe_str(row.get("category"), "Unknown"),
            "damage_type": safe_str(row.get("damage type"), "Standard"),
            "requirements": reqs,
            "scaling": scaling,
            "primary_scaling": primary_scaling,
            "base_damage": base_dmg,
            "passive_effect": safe_str(row.get("passive effect"), "None"),
            "skill": safe_str(row.get("skill"), "None"),
            "fp_cost": safe_str(row.get("FP cost"), "0"),
            "weight": safe_float(row.get("weight"), 0.0),
            "dlc": int(row.get("dlc", 0)),
        })

    print(f"  Enriched {len(weapons)} weapons")
    return weapons

# ============================================================
# STEP 2b: PARSE & ENRICH BOSSES + VULNERABILITY ANALYSIS
# ============================================================
def enrich_bosses(data, lore_lib, weapon_index):
    print("\n" + "=" * 60)
    print("STEP 2b: Enriching bosses with vulnerability analysis")
    print("=" * 60)

    df_bosses = data["bosses"]
    df_stats = data["boss_stats"]
    if df_bosses is None:
        return []

    # Build stats lookup by normalized name
    stats_lookup = {}
    if df_stats is not None:
        for _, r in df_stats.iterrows():
            n = norm(r.get("boss", ""))
            stats_lookup[n] = r.to_dict()
        print(f"  Built boss stats lookup: {len(stats_lookup)} entries")

    bosses = []
    for _, row in df_bosses.iterrows():
        name = safe_str(row.get("name"), "Unknown Boss")

        # Parse Locations & Drops
        loc_drops_raw = safe_literal_eval(row.get("Locations & Drops"))
        locations = []
        drops = []
        if isinstance(loc_drops_raw, dict):
            for loc, items in loc_drops_raw.items():
                loc_clean = str(loc).rstrip(":").strip()
                locations.append(loc_clean)
                if isinstance(items, list):
                    # First item is usually runes amount, rest are drops
                    for item in items:
                        item_str = str(item).strip()
                        # Skip pure rune amounts like "120,000"
                        if not re.match(r'^[\d,]+$', item_str):
                            drops.append(item_str)

        # Fuzzy match stats
        stats = fuzzy_get(stats_lookup, name)

        # Vulnerability analysis
        vuln = analyze_boss_vulnerability(stats, weapon_index) if stats else {
            "weakest_physical": "Unknown",
            "status_vulnerabilities": [],
            "inflicts": [],
            "dominant_damage": "Unknown",
            "parryable": "Unknown",
            "stance": None,
            "defense": None,
            "recommended_weapons": {},
        }

        lore_text = fuzzy_get(lore_lib, name) or ""
        blockquote = safe_str(row.get("blockquote"), "")

        bosses.append({
            "name": name,
            "description": blockquote if blockquote else lore_text,
            "lore": lore_text,
            "hp": safe_str(row.get("HP"), "Unknown"),
            "locations": locations,
            "drops": drops,
            "dlc": int(row.get("dlc", 0)),
            **vuln,
        })

    print(f"  Enriched {len(bosses)} bosses")
    return bosses


def analyze_boss_vulnerability(stats, weapon_index):
    """Analyze a boss's weaknesses and recommend weapons."""

    # --- Physical weakness ---
    phys_neg = {
        "Standard": safe_float(stats.get("neg_standard")),
        "Slash": safe_float(stats.get("neg_slash")),
        "Strike": safe_float(stats.get("neg_strike")),
        "Pierce": safe_float(stats.get("neg_pierce")),
    }
    # Filter out None values, find lowest negation = most vulnerable
    valid_phys = {k: v for k, v in phys_neg.items() if v is not None}
    weakest_physical = "Unknown"
    if valid_phys:
        weakest_physical = min(valid_phys, key=valid_phys.get)

    # --- Elemental weakness (from dmg_ columns, 1 = takes this type) ---
    elem_vuln = {}
    for elem in ["magic", "fire", "lightning", "holy"]:
        val = stats.get(f"dmg_{elem}", 0)
        # dmg_ columns indicate damage type the boss DEALS, not weakness
        # But neg_ for elemental isn't in this CSV, so we skip elemental weakness

    # --- Status vulnerabilities ---
    status_map = {
        "Hemorrhage": "res_hemorrhage",
        "Frostbite": "res_frostbite",
        "Poison": "res_poison",
        "Scarlet Rot": "res_scarlet_rot",
    }
    status_vulns = []
    status_details = {}
    for status_name, col in status_map.items():
        val = stats.get(col)
        if pd.notna(val) and str(val).strip().lower() != "immune":
            status_vulns.append(status_name)
            status_details[status_name] = str(val).strip()

    # --- What boss inflicts ---
    inflict_map = {
        "Bleed": "inflicts_bleed",
        "Frostbite": "inflicts_frostbite",
        "Scarlet Rot": "inflicts_scarlet_rot",
        "Poison": "inflicts_poison",
        "Madness": "inflicts_madness",
        "Sleep": "inflicts_sleep",
    }
    inflicts = [name for name, col in inflict_map.items() if stats.get(col, 0) == 1]

    # --- Dominant damage type boss deals ---
    dmg_types = {
        "Standard": stats.get("dmg_standard", 0),
        "Slash": stats.get("dmg_slash", 0),
        "Strike": stats.get("dmg_strike", 0),
        "Pierce": stats.get("dmg_pierce", 0),
        "Magic": stats.get("dmg_magic", 0),
        "Fire": stats.get("dmg_fire", 0),
        "Lightning": stats.get("dmg_lightning", 0),
        "Holy": stats.get("dmg_holy", 0),
    }
    active_dmg = [k for k, v in dmg_types.items() if v == 1]
    dominant_damage = ", ".join(active_dmg) if active_dmg else "Unknown"

    # --- Parryable ---
    parryable_val = safe_float(stats.get("parryable"))
    if parryable_val is not None:
        parryable = True if parryable_val == 1.0 else False
    else:
        parryable = "Unknown"

    # --- Weapon recommendations by build ---
    recommended = recommend_weapons(
        weakest_physical, status_vulns, weapon_index,
        status_details=status_details,
        physical_negation=valid_phys,
    )

    return {
        "weakest_physical": weakest_physical,
        "physical_negation": {k: v for k, v in phys_neg.items() if v is not None},
        "status_vulnerabilities": status_vulns,
        "status_resistance_values": status_details,
        "inflicts": inflicts,
        "dominant_damage": dominant_damage,
        "parryable": parryable,
        "stance": safe_float(stats.get("stance")),
        "defense": safe_float(stats.get("defense")),
        "recommended_weapons": recommended,
    }


def parse_first_resistance(res_str):
    """Parse '290 / 332 / 430 / 720' → 290.0, or return infinity if Immune/invalid."""
    if pd.isna(res_str) or str(res_str).strip().lower() == "immune":
        return float("inf")
    try:
        first_val = str(res_str).split("/")[0].strip().replace(",", "")
        return float(first_val)
    except (ValueError, IndexError):
        return float("inf")


def rank_status_vulnerabilities(status_vulns, status_details):
    """
    Rank status effects by how easy they are to proc (lowest base resistance first).
    Returns list of (status_name, base_resistance) sorted ascending.
    """
    ranked = []
    for status in status_vulns:
        base_res = parse_first_resistance(status_details.get(status, ""))
        ranked.append((status, base_res))
    ranked.sort(key=lambda x: x[1])
    return ranked


def recommend_weapons(weakest_physical, status_vulns, weapon_index,
                      status_details=None, physical_negation=None):
    """
    Recommend 3-5 weapons across different build types.
    Scoring:
      - Status effect match weighted by 1/base_resistance (lower res = higher score)
      - Physical type match weighted by how much lower the negation is vs others
      - Weapons with both a status AND physical match score highest
    """
    if status_details is None:
        status_details = {}
    if physical_negation is None:
        physical_negation = {}

    # Rank statuses by effectiveness (lowest resistance = best)
    ranked_statuses = rank_status_vulnerabilities(status_vulns, status_details)

    # Assign status weights: best status gets 10, next 7, next 4, next 2
    status_weights = {}
    tier_scores = [10, 7, 4, 2]
    for i, (status_name, base_res) in enumerate(ranked_statuses):
        status_weights[status_name] = tier_scores[i] if i < len(tier_scores) else 1

    # Physical type weight: bonus if weakest_physical is strictly lower than others
    phys_bonus = 1
    if physical_negation and weakest_physical != "Unknown":
        vals = [v for v in physical_negation.values() if v is not None]
        weakest_val = physical_negation.get(weakest_physical)
        if weakest_val is not None and vals:
            avg_others = sum(v for v in vals if v != weakest_val) / max(len(vals) - 1, 1)
            if weakest_val < avg_others:
                phys_bonus = 3  # meaningful physical weakness

    # Status keyword mapping for matching weapon passive effects
    status_keywords = {
        "Hemorrhage": ["blood loss", "hemorrhage", "bleed"],
        "Frostbite": ["frostbite", "frost"],
        "Poison": ["poison"],
        "Scarlet Rot": ["scarlet rot", "rot"],
        "Madness": ["madness"],
        "Sleep": ["sleep"],
        "Death Blight": ["death", "blight"],
    }

    def get_weapon_status(weapon):
        """Determine which status effect a weapon applies."""
        passive = weapon.get("passive_effect", "None").lower()
        if passive in ["none", "no passive effects", ""]:
            return None
        for status_name, keywords in status_keywords.items():
            if any(kw in passive for kw in keywords):
                return status_name
        return None

    # Score ALL weapons
    all_weapons_scored = []
    for w in weapon_index.get("by_damage_type", {}).values():
        for weapon in w:
            all_weapons_scored.append(weapon)
    for w in weapon_index.get("by_status", {}).values():
        for weapon in w:
            all_weapons_scored.append(weapon)

    # Deduplicate
    seen = set()
    unique_weapons = []
    for w in all_weapons_scored:
        if w["name"] not in seen:
            seen.add(w["name"])
            unique_weapons.append(w)

    # Score each weapon
    scored = []
    for w in unique_weapons:
        score = 0
        reasons = []

        # Physical type match
        w_dmg_types = [d.strip().capitalize() for d in w.get("damage_type", "").split("/")]
        if weakest_physical in w_dmg_types:
            score += phys_bonus
            if phys_bonus > 1:
                reasons.append(f"Exploits {weakest_physical} weakness")

        # Status match (weighted by boss vulnerability)
        w_status = get_weapon_status(w)
        if w_status and w_status in status_weights:
            score += status_weights[w_status]
            res_val = status_details.get(w_status, "")
            reasons.append(f"Applies {w_status} (boss resistance: {res_val})")

        # Small bonus for having any passive when boss has vulnerabilities
        if w_status and w_status in status_vulns and not reasons:
            score += 1

        if score > 0:
            scored.append({
                **w,
                "score": score,
                "reason": "; ".join(reasons) if reasons else f"Deals {w.get('damage_type', 'Unknown')} damage",
            })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Distribute into build categories
    scaling_to_build = {
        "Str": "strength", "Dex": "dexterity", "Int": "intelligence",
        "Fai": "faith", "Arc": "arcane",
    }
    recs = {"strength": [], "dexterity": [], "intelligence": [], "faith": [], "arcane": []}

    for w in scored:
        build = scaling_to_build.get(w.get("primary_scaling"), None)
        if build and len(recs[build]) < 2:
            recs[build].append({
                "name": w["name"],
                "category": w.get("category", ""),
                "damage_type": w.get("damage_type", ""),
                "passive_effect": w.get("passive_effect", "None"),
                "reason": w.get("reason", ""),
            })

    # Fallback: if a build has 0 recs, pick best overall that scales with it
    for stat, build in scaling_to_build.items():
        if not recs[build] and scored:
            for w in scored[:10]:
                scaling = w.get("scaling", {})
                grade_order = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "-": 0}
                if grade_order.get(scaling.get(stat, "-"), 0) >= 2 and len(recs[build]) < 2:
                    recs[build].append({
                        "name": w["name"],
                        "category": w.get("category", ""),
                        "damage_type": w.get("damage_type", ""),
                        "passive_effect": w.get("passive_effect", "None"),
                        "reason": w.get("reason", ""),
                    })

    return {k: v for k, v in recs.items() if v}


# ============================================================
# STEP 2c: BUILD WEAPON INDEX (needed before boss enrichment)
# ============================================================
def build_weapon_index(weapons):
    """Build lookup indexes for weapons by damage type and status effect."""
    print("\n  Building weapon cross-reference indexes...")

    by_damage_type = defaultdict(list)
    by_status = defaultdict(list)
    by_category = defaultdict(list)
    by_scaling = defaultdict(list)

    # Status effect keyword mapping from passive_effect strings
    status_keywords = {
        "Blood Loss": "Hemorrhage",
        "Bleed": "Hemorrhage",
        "Frost": "Frostbite",
        "Frostbite": "Frostbite",
        "Poison": "Poison",
        "Scarlet Rot": "Scarlet Rot",
        "Rot": "Scarlet Rot",
        "Madness": "Madness",
        "Sleep": "Sleep",
        "Death": "Death Blight",
    }

    for w in weapons:
        # Index by damage type (can be "Standard/Pierce" → split)
        dmg_types = [d.strip() for d in w.get("damage_type", "Standard").split("/")]
        for dt in dmg_types:
            # Normalize: "Standard" from "Standard/Pierce"
            dt_clean = dt.strip().capitalize()
            if dt_clean in ["Standard", "Slash", "Strike", "Pierce"]:
                by_damage_type[dt_clean].append(w)

        # Index by status effect
        passive = w.get("passive_effect", "None")
        if passive and passive != "None" and passive != "No passive effects":
            for keyword, status_name in status_keywords.items():
                if keyword.lower() in passive.lower():
                    by_status[status_name].append(w)
                    break

        # Index by category
        by_category[w.get("category", "Unknown")].append(w)

        # Index by primary scaling
        by_scaling[w.get("primary_scaling", "None")].append(w)

    print(f"    Damage type index: {', '.join(f'{k}: {len(v)}' for k, v in by_damage_type.items())}")
    print(f"    Status index: {', '.join(f'{k}: {len(v)}' for k, v in by_status.items())}")

    return {
        "by_damage_type": dict(by_damage_type),
        "by_status": dict(by_status),
        "by_category": dict(by_category),
        "by_scaling": dict(by_scaling),
    }


# ============================================================
# STEP 2d: ENRICH SORCERIES & INCANTATIONS
# ============================================================
def enrich_magic(data, lore_lib, category_name):
    print(f"\n  Enriching {category_name}...")
    df = data[category_name]
    if df is None:
        return []

    items = []
    for _, row in df.iterrows():
        name = safe_str(row.get("name"), "Unknown")
        lore = fuzzy_get(lore_lib, name) or ""

        reqs = {}
        for stat in ["INT", "FAI", "ARC"]:
            val = safe_float(row.get(stat))
            if val is not None and val > 0:
                reqs[stat] = int(val)

        items.append({
            "name": name,
            "type": category_name,  # "sorceries" or "incantations"
            "description": safe_str(row.get("description"), ""),
            "lore": lore,
            "effect": safe_str(row.get("effect"), ""),
            "fp_cost": safe_str(row.get("FP"), "0"),
            "slot": int(row.get("slot", 1)),
            "requirements": reqs,
            "stamina_cost": safe_str(row.get("stamina cost"), "0"),
            "bonus": safe_str(row.get("bonus"), "None"),
            "group": safe_str(row.get("group"), ""),  # incantations only
            "location": safe_str(row.get("location"), "Unknown"),
            "dlc": safe_str(row.get("dlc"), "0"),
        })

    print(f"    Enriched {len(items)} {category_name}")
    return items


# ============================================================
# STEP 2e: ENRICH NPCS
# ============================================================
def enrich_npcs(data, lore_lib):
    print("\n  Enriching NPCs...")
    df = data["npcs"]
    if df is None:
        return []

    npcs = []
    for _, row in df.iterrows():
        name = safe_str(row.get("name"), "Unknown")
        lore = fuzzy_get(lore_lib, name) or ""

        npcs.append({
            "name": name,
            "description": safe_str(row.get("description"), ""),
            "lore": lore,
            "location": safe_str(row.get("location"), "Unknown"),
            "role": safe_str(row.get("role"), "Unknown"),
            "voiced_by": safe_str(row.get("voiced by"), "Unknown"),
            "dlc": int(row.get("dlc", 0)),
        })

    print(f"    Enriched {len(npcs)} NPCs")
    return npcs


# ============================================================
# STEP 2f: ENRICH LOCATIONS
# ============================================================
def enrich_locations(data, lore_lib):
    print("\n  Enriching locations...")
    df = data["locations"]
    if df is None:
        return []

    locations = []
    for _, row in df.iterrows():
        name = safe_str(row.get("name"), "Unknown")

        locations.append({
            "name": name,
            "description": safe_str(row.get("description"), ""),
            "region": safe_str(row.get("region"), "Unknown"),
            "items": parse_list_col(row.get("items")),
            "npcs": parse_list_col(row.get("npcs")),
            "creatures": parse_list_col(row.get("creatures")),
            "bosses": parse_list_col(row.get("bosses")),
            "dlc": int(row.get("dlc", 0)),
        })

    print(f"    Enriched {len(locations)} locations")
    return locations


# ============================================================
# STEP 2g: ENRICH ARMORS
# ============================================================
def enrich_armors(data, lore_lib):
    print("\n  Enriching armors...")
    df = data["armors"]
    if df is None:
        return []

    armors = []
    for _, row in df.iterrows():
        name = safe_str(row.get("name"), "Unknown")
        lore = fuzzy_get(lore_lib, name) or ""

        # Parse damage negation
        dmg_neg = safe_literal_eval(row.get("damage negation"))
        if isinstance(dmg_neg, list) and len(dmg_neg) > 0:
            dmg_neg = dmg_neg[0]  # It's a list of one dict
        if not isinstance(dmg_neg, dict):
            dmg_neg = {}

        # Parse resistance
        res = safe_literal_eval(row.get("resistance"))
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        if not isinstance(res, dict):
            res = {}

        armors.append({
            "name": name,
            "description": safe_str(row.get("description"), ""),
            "lore": lore,
            "type": safe_str(row.get("type"), "Unknown"),
            "damage_negation": dmg_neg,
            "resistance": res,
            "weight": safe_float(row.get("weight"), 0.0),
            "special_effect": safe_str(row.get("special effect"), "None"),
            "how_to_acquire": safe_str(row.get("how to acquire"), "Unknown"),
            "dlc": safe_str(row.get("dlc"), "0"),
        })

    print(f"    Enriched {len(armors)} armors")
    return armors


# ============================================================
# STEP 2h: ENRICH CREATURES
# ============================================================
def enrich_creatures(data, lore_lib):
    print("\n  Enriching creatures...")
    df = data["creatures"]
    if df is None:
        return []

    creatures = []
    for _, row in df.iterrows():
        name = safe_str(row.get("name"), "Unknown")
        lore = fuzzy_get(lore_lib, name) or ""

        creatures.append({
            "name": name,
            "description": safe_str(row.get("blockquote"), ""),
            "lore": lore,
            "locations": parse_list_col(row.get("locations")),
            "drops": parse_list_col(row.get("drops")),
            "dlc": int(row.get("dlc", 0)),
        })

    print(f"    Enriched {len(creatures)} creatures")
    return creatures


# ============================================================
# STEP 2i: ENRICH ASHES OF WAR
# ============================================================
def enrich_ashes_of_war(data, lore_lib):
    print("\n  Enriching ashes of war...")
    df = data["ashes_of_war"]
    if df is None:
        return []

    ashes = []
    for _, row in df.iterrows():
        name = safe_str(row.get("name"), "Unknown")
        lore = fuzzy_get(lore_lib, name) or ""

        ashes.append({
            "name": name,
            "description": safe_str(row.get("description"), ""),
            "lore": lore,
            "affinity": safe_str(row.get("affinity"), "Standard"),
            "skill": safe_str(row.get("skill"), "Unknown"),
            "dlc": int(row.get("dlc", 0)),
        })

    print(f"    Enriched {len(ashes)} ashes of war")
    return ashes


# ============================================================
# STEP 2j: ENRICH SKILLS
# ============================================================
def enrich_skills(data, lore_lib):
    print("\n  Enriching skills...")
    df = data["skills"]
    if df is None:
        return []

    skills = []
    for _, row in df.iterrows():
        name = safe_str(row.get("name"), "Unknown")

        skills.append({
            "name": name,
            "type": safe_str(row.get("type"), "Regular"),
            "equipment": safe_str(row.get("equipament"), "Unknown"),
            "chargeable": safe_str(row.get("charge"), "No"),
            "fp_cost": safe_str(row.get("FP"), "0"),
            "effect": safe_str(row.get("effect"), ""),
            "location": safe_str(row.get("locations"), "Unknown"),
            "dlc": int(row.get("dlc", 0)),
        })

    print(f"    Enriched {len(skills)} skills")
    return skills


# ============================================================
# STEP 2k: BUILD LOCATION CROSS-REFERENCE INDEX
# ============================================================
def build_location_index(locations):
    """Build reverse lookups: entity_name → list of locations."""
    print("\n  Building location cross-reference index...")

    boss_to_locs = defaultdict(list)
    npc_to_locs = defaultdict(list)
    creature_to_locs = defaultdict(list)

    for loc in locations:
        loc_name = loc["name"]
        for boss in loc.get("bosses", []):
            boss_to_locs[norm(boss)].append(loc_name)
        for npc in loc.get("npcs", []):
            npc_to_locs[norm(npc)].append(loc_name)
        for creature in loc.get("creatures", []):
            creature_to_locs[norm(creature)].append(loc_name)

    print(f"    Boss→location entries: {len(boss_to_locs)}")
    print(f"    NPC→location entries: {len(npc_to_locs)}")
    print(f"    Creature→location entries: {len(creature_to_locs)}")

    return {
        "boss_to_locations": dict(boss_to_locs),
        "npc_to_locations": dict(npc_to_locs),
        "creature_to_locations": dict(creature_to_locs),
    }


# ============================================================
# STEP 2l: BUILD ARMOR RECOMMENDATION INDEX
# ============================================================
def build_armor_index(armors):
    """Index armors by type and notable resistances for boss counter recs."""
    print("\n  Building armor recommendation index...")

    by_type = defaultdict(list)
    for a in armors:
        by_type[a.get("type", "Unknown")].append(a)

    # We store the full list; generate_qa.py can sort/filter as needed
    print(f"    Armor types: {', '.join(f'{k}: {len(v)}' for k, v in by_type.items())}")
    return {"by_type": dict(by_type)}


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("\n🗡️  Elden Ring Data Fusion Pipeline")
    print("=" * 60)

    # Step 1: Load everything
    data, lore_lib = load_all_data()

    # Step 2a: Enrich weapons
    weapons = enrich_weapons(data, lore_lib)

    # Step 2c: Build weapon index (needed for boss recommendations)
    weapon_index = build_weapon_index(weapons)

    # Step 2b: Enrich bosses (uses weapon index)
    bosses = enrich_bosses(data, lore_lib, weapon_index)

    # Step 2d: Enrich magic
    sorceries = enrich_magic(data, lore_lib, "sorceries")
    incantations = enrich_magic(data, lore_lib, "incantations")

    # Step 2e-j: Enrich remaining entities
    npcs = enrich_npcs(data, lore_lib)
    locations = enrich_locations(data, lore_lib)
    armors = enrich_armors(data, lore_lib)
    creatures = enrich_creatures(data, lore_lib)
    ashes_of_war = enrich_ashes_of_war(data, lore_lib)
    skills = enrich_skills(data, lore_lib)

    # Step 2k-l: Build cross-reference indexes
    location_index = build_location_index(locations)
    armor_index = build_armor_index(armors)

    # Assemble final enriched output
    enriched = {
        "metadata": {
            "total_weapons": len(weapons),
            "total_bosses": len(bosses),
            "total_sorceries": len(sorceries),
            "total_incantations": len(incantations),
            "total_npcs": len(npcs),
            "total_locations": len(locations),
            "total_armors": len(armors),
            "total_creatures": len(creatures),
            "total_ashes_of_war": len(ashes_of_war),
            "total_skills": len(skills),
        },
        "weapons": weapons,
        "bosses": bosses,
        "sorceries": sorceries,
        "incantations": incantations,
        "npcs": npcs,
        "locations": locations,
        "armors": armors,
        "creatures": creatures,
        "ashes_of_war": ashes_of_war,
        "skills": skills,
        "indexes": {
            "weapon_index_summary": {
                "damage_types": list(weapon_index["by_damage_type"].keys()),
                "status_effects": list(weapon_index["by_status"].keys()),
                "categories": list(weapon_index["by_category"].keys()),
            },
            "location_index": location_index,
            "armor_types": list(armor_index["by_type"].keys()),
        },
    }

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("✅ FUSION COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total entities enriched: {sum(v for k, v in enriched['metadata'].items())}")
    for k, v in enriched["metadata"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()