"""
generate_qa.py - Elden Ring QA Dataset Generator
=================================================
Loads elden_ring_enriched.json and generates diverse QA pairs
with varied phrasings, tagged for stratified train/val/test splitting.

Pipeline: fuse_data.py → elden_ring_enriched.json
                                ↓
          generate_qa.py → elden_ring_final_train.jsonl
                                ↓
          train.py       → fine-tuned model
"""

import json
import random
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = "elden_ring_enriched.json"
OUTPUT_PATH = "elden_ring_final_train.jsonl"
SEED = 42
random.seed(SEED)

# ============================================================
# UTILITIES
# ============================================================
def pick(options):
    """Pick a random phrasing from a list."""
    return random.choice(options)

def safe(val, default="Unknown"):
    """Return val if truthy, else default."""
    if val is None or val == "" or val == "Unknown":
        return default
    return str(val).strip()

def fmt_reqs(reqs):
    """Format requirements dict to readable string."""
    if not reqs or not isinstance(reqs, dict):
        return "no special requirements"
    parts = [f"{v} {k}" for k, v in reqs.items() if v and str(v) != "0"]
    return ", ".join(parts) if parts else "no special requirements"

def fmt_list(items, limit=8):
    """Format a list to readable string, capped."""
    if not items:
        return "none known"
    clean = [str(x).strip() for x in items if x and str(x).strip()]
    if not clean:
        return "none known"
    if len(clean) > limit:
        return ", ".join(clean[:limit]) + f", and {len(clean) - limit} more"
    return ", ".join(clean)

def fmt_scaling(scaling):
    """Format scaling dict to readable string."""
    if not scaling or not isinstance(scaling, dict):
        return "unknown scaling"
    parts = [f"{k}: {v}" for k, v in scaling.items() if v and v != "-"]
    return ", ".join(parts) if parts else "no notable scaling"

def fmt_negation(neg):
    """Format damage negation dict."""
    if not neg or not isinstance(neg, dict):
        return "unknown"
    parts = [f"{k}: {v}" for k, v in neg.items()]
    return ", ".join(parts)

def fmt_resistance(res):
    """Format resistance dict."""
    if not res or not isinstance(res, dict):
        return "unknown"
    parts = [f"{k}: {v}" for k, v in res.items()]
    return ", ".join(parts)

def make_entry(instruction, output, entity_type, question_type, entity_name):
    """Create a single QA entry with metadata tags."""
    return {
        "instruction": instruction,
        "input": "",
        "output": output.strip(),
        "metadata": {
            "entity_type": entity_type,
            "question_type": question_type,
            "entity_name": entity_name,
        }
    }

# ============================================================
# WEAPON QA TEMPLATES
# ============================================================
def generate_weapon_qa(weapons):
    entries = []
    for w in weapons:
        n = w["name"]
        desc = safe(w.get("description"), "")
        lore = safe(w.get("lore"), "")
        cat = safe(w.get("category"))
        dmg = safe(w.get("damage_type"))
        reqs = w.get("requirements", {})
        scaling = w.get("scaling", {})
        passive = safe(w.get("passive_effect"), "None")
        skill = safe(w.get("skill"), "None")
        fp = safe(w.get("fp_cost"), "0")
        weight = w.get("weight", 0)
        base_dmg = w.get("base_damage", {})

        # 1. Lore / Description
        if desc and desc != "Unknown":
            q = pick([
                f"What is the {n} in Elden Ring?",
                f"Tell me about the {n}.",
                f"Describe the {n} from Elden Ring.",
                f"What's the lore behind the {n}?",
            ])
            full_desc = desc
            if lore and lore != desc:
                full_desc += f" {lore}"
            entries.append(make_entry(q, full_desc, "weapon", "lore", n))

        # 2. Category / Type
        q = pick([
            f"What type of weapon is the {n}?",
            f"What category does {n} fall under?",
            f"What kind of weapon is {n}?",
        ])
        a = f"The {n} is a {cat} that deals {dmg} damage."
        entries.append(make_entry(q, a, "weapon", "category", n))

        # 3. Requirements
        q = pick([
            f"What stats do I need to use {n}?",
            f"What are the requirements for the {n}?",
            f"Can you tell me the stat requirements for {n}?",
        ])
        a = f"The {n} requires {fmt_reqs(reqs)} to wield."
        entries.append(make_entry(q, a, "weapon", "requirements", n))

        # 4. Scaling
        if scaling:
            q = pick([
                f"What does {n} scale with?",
                f"How does the {n} scale?",
                f"What are the scaling grades for {n}?",
            ])
            a = f"The {n} has the following scaling: {fmt_scaling(scaling)}."
            entries.append(make_entry(q, a, "weapon", "scaling", n))

        # 5. Passive effect
        if passive and passive not in ["None", "No passive effects"]:
            q = pick([
                f"Does {n} have any passive effects?",
                f"What passive effect does {n} have?",
                f"Does the {n} cause any status buildup?",
            ])
            a = f"Yes, the {n} has the passive effect: {passive}."
            entries.append(make_entry(q, a, "weapon", "passive", n))
        else:
            q = f"Does {n} have any passive effects?"
            a = f"No, the {n} does not have any passive effects."
            entries.append(make_entry(q, a, "weapon", "passive", n))

        # 6. Skill
        if skill and skill != "None":
            q = pick([
                f"What skill does {n} have?",
                f"What's the weapon skill on {n}?",
                f"What ash of war comes with {n}?",
            ])
            a = f"The {n} has the skill {skill}"
            if fp and fp != "0":
                a += f", which costs {fp} FP to use."
            else:
                a += "."
            entries.append(make_entry(q, a, "weapon", "skill", n))

        # 7. Weight
        q = pick([
            f"How heavy is the {n}?",
            f"What's the weight of {n}?",
        ])
        a = f"The {n} weighs {weight} units."
        entries.append(make_entry(q, a, "weapon", "weight", n))

        # 8. Base damage (if available)
        if base_dmg:
            active_dmg = {k: v for k, v in base_dmg.items() if v and v not in ["0", "-", "0.0"]}
            if active_dmg:
                q = pick([
                    f"What's the base damage of {n}?",
                    f"How much damage does {n} do?",
                ])
                dmg_str = ", ".join(f"{v} {k}" for k, v in active_dmg.items())
                a = f"The {n} has base damage of {dmg_str}."
                entries.append(make_entry(q, a, "weapon", "base_damage", n))

    return entries


# ============================================================
# BOSS QA TEMPLATES
# ============================================================
def generate_boss_qa(bosses):
    entries = []
    for b in bosses:
        n = b["name"]
        desc = safe(b.get("description"), "")
        hp = safe(b.get("hp"), "Unknown")
        locs = b.get("locations", [])
        drops = b.get("drops", [])
        weak_phys = safe(b.get("weakest_physical"), "Unknown")
        phys_neg = b.get("physical_negation", {})
        status_vulns = b.get("status_vulnerabilities", [])
        status_res = b.get("status_resistance_values", {})
        inflicts = b.get("inflicts", [])
        dominant_dmg = safe(b.get("dominant_damage"), "Unknown")
        parryable = b.get("parryable", "Unknown")
        stance = b.get("stance")
        recs = b.get("recommended_weapons", {})

        # 1. Description / Lore
        if desc and desc != "Unknown":
            q = pick([
                f"Who is {n} in Elden Ring?",
                f"Tell me about {n}.",
                f"What do you know about {n}?",
            ])
            entries.append(make_entry(q, desc, "boss", "lore", n))

        # 2. Location + Drops
        if locs:
            q = pick([
                f"Where do I find {n}?",
                f"What's the location of {n}?",
                f"Where is {n} located?",
            ])
            a = f"{n} can be found at {fmt_list(locs)}."
            if drops:
                a += f" Defeating them rewards: {fmt_list(drops)}."
            entries.append(make_entry(q, a, "boss", "location", n))

        # 3. HP
        if hp != "Unknown":
            q = pick([
                f"How much health does {n} have?",
                f"What's the HP of {n}?",
            ])
            a = f"{n} has {hp} HP."
            entries.append(make_entry(q, a, "boss", "hp", n))

        # 4. Weakness analysis
        if status_vulns or weak_phys != "Unknown":
            q = pick([
                f"What is {n} weak to?",
                f"What are the weaknesses of {n}?",
                f"How can I exploit {n}'s weaknesses?",
            ])
            parts = []
            if weak_phys != "Unknown":
                # Check if there's an actual difference in negation values
                vals = list(phys_neg.values())
                if vals and min(vals) < max(vals):
                    parts.append(f"most vulnerable to {weak_phys} damage (negation: {phys_neg.get(weak_phys, '?')})")
                else:
                    parts.append(f"equally resistant to all physical types (negation: {vals[0] if vals else '?'})")
            if status_vulns:
                # Sort by resistance for answer quality
                def parse_res_val(s):
                    raw = status_res.get(s, "999").split("/")[0].strip().replace(",", "")
                    try:
                        return float(raw)
                    except (ValueError, TypeError):
                        return 999.0
                sorted_status = sorted(status_vulns, key=parse_res_val)
                parts.append(f"susceptible to {fmt_list(sorted_status)}")
                if sorted_status and sorted_status[0] in status_res:
                    parts.append(f"{sorted_status[0]} is the most effective (resistance: {status_res[sorted_status[0]]})")
            a = f"{n} is {'; '.join(parts)}."
            entries.append(make_entry(q, a, "boss", "weakness", n))

        # 5. Weapon recommendations
        if recs:
            q = pick([
                f"What weapons are good against {n}?",
                f"What should I use to fight {n}?",
                f"Best weapons for {n}?",
                f"How do I beat {n}?",
            ])
            rec_parts = []
            for build, weapons in recs.items():
                for w in weapons:
                    rec_parts.append(f"{w['name']} ({build} build - {w.get('reason', '')})")
            a = f"Effective weapons against {n} include: {'; '.join(rec_parts[:5])}."
            entries.append(make_entry(q, a, "boss", "weapon_recommendation", n))

            # Per-build questions
            for build, weapons in recs.items():
                if weapons:
                    q = pick([
                        f"What {build} weapons work against {n}?",
                        f"Best {build} build weapons for {n}?",
                        f"I'm running a {build} build, what should I use against {n}?",
                    ])
                    wnames = [f"{w['name']} ({w.get('reason', '')})" for w in weapons]
                    a = f"For a {build} build against {n}, try {'; '.join(wnames)}."
                    entries.append(make_entry(q, a, "boss", f"weapon_rec_{build}", n))

        # 6. Status vulnerability specific
        for status in status_vulns:
            status_lower = status.lower()
            q = pick([
                f"Is {n} weak to {status_lower}?",
                f"Can I use {status_lower} on {n}?",
                f"Does {status_lower} work against {n}?",
            ])
            res_val = status_res.get(status, "unknown")
            a = f"Yes, {n} is vulnerable to {status} with a resistance of {res_val}."
            entries.append(make_entry(q, a, "boss", "status_check", n))

        # Immune statuses
        all_statuses = ["Hemorrhage", "Frostbite", "Poison", "Scarlet Rot"]
        immune_to = [s for s in all_statuses if s not in status_vulns]
        for status in immune_to:
            status_lower = status.lower()
            q = f"Is {n} weak to {status_lower}?"
            a = f"No, {n} is immune to {status}."
            entries.append(make_entry(q, a, "boss", "status_check", n))

        # 7. What boss inflicts
        if inflicts:
            q = pick([
                f"What status effects does {n} inflict?",
                f"What should I watch out for against {n}?",
                f"Does {n} cause any status effects?",
            ])
            a = f"{n} can inflict {fmt_list(inflicts)}. Prepare accordingly with the right resistances."
            entries.append(make_entry(q, a, "boss", "inflicts", n))

        # 8. Parryable
        if parryable != "Unknown":
            q = pick([
                f"Can {n} be parried?",
                f"Is {n} parryable?",
            ])
            if parryable:
                a = f"Yes, {n} can be parried."
                if stance:
                    a += f" They have a stance value of {int(stance)}."
            else:
                a = f"No, {n} cannot be parried."
            entries.append(make_entry(q, a, "boss", "parry", n))

        # 9. Damage the boss deals
        if dominant_dmg and dominant_dmg != "Unknown":
            q = pick([
                f"What type of damage does {n} deal?",
                f"What damage should I prepare for against {n}?",
            ])
            a = f"{n} primarily deals {dominant_dmg} damage."
            if inflicts:
                a += f" They also inflict {fmt_list(inflicts)}."
            entries.append(make_entry(q, a, "boss", "boss_damage", n))

        # 10. Drops only
        if drops:
            q = pick([
                f"What does {n} drop?",
                f"What rewards do I get for beating {n}?",
                f"What loot does {n} give?",
            ])
            a = f"Defeating {n} rewards: {fmt_list(drops)}."
            entries.append(make_entry(q, a, "boss", "drops", n))

    return entries


# ============================================================
# SORCERY & INCANTATION QA TEMPLATES
# ============================================================
def generate_magic_qa(spells, spell_type):
    """spell_type: 'sorcery' or 'incantation'"""
    entries = []
    for s in spells:
        n = s["name"]
        desc = safe(s.get("description"), "")
        effect = safe(s.get("effect"), "")
        fp = safe(s.get("fp_cost"), "0")
        slot = s.get("slot", 1)
        reqs = s.get("requirements", {})
        bonus = safe(s.get("bonus"), "None")
        loc = safe(s.get("location"), "Unknown")
        group = safe(s.get("group"), "")

        # 1. Description / Lore
        if desc and desc != "Unknown":
            q = pick([
                f"What is {n} in Elden Ring?",
                f"Tell me about the {spell_type} {n}.",
                f"Describe the {n} {spell_type}.",
            ])
            entries.append(make_entry(q, desc, spell_type, "lore", n))

        # 2. Effect
        if effect and effect != "Unknown":
            q = pick([
                f"What does {n} do?",
                f"What's the effect of {n}?",
                f"How does {n} work?",
            ])
            a = f"{n} {effect}."
            entries.append(make_entry(q, a, spell_type, "effect", n))

        # 3. Requirements
        q = pick([
            f"What do I need to cast {n}?",
            f"What are the requirements for {n}?",
            f"What stats do I need for {n}?",
        ])
        a = f"{n} requires {fmt_reqs(reqs)} and uses {slot} slot(s). It costs {fp} FP to cast."
        entries.append(make_entry(q, a, spell_type, "requirements", n))

        # 4. Location
        if loc and loc != "Unknown":
            q = pick([
                f"Where can I find {n}?",
                f"How do I get {n}?",
                f"Where is the {spell_type} {n} located?",
            ])
            a = f"{n} can be obtained: {loc}"
            entries.append(make_entry(q, a, spell_type, "location", n))

        # 5. Bonus / School
        if bonus and bonus != "None":
            q = pick([
                f"What school does {n} belong to?",
                f"What type of {spell_type} is {n}?",
            ])
            a = f"{n} belongs to the {bonus} school"
            if group:
                a += f" and is categorized as {group}"
            a += "."
            entries.append(make_entry(q, a, spell_type, "school", n))

    return entries


# ============================================================
# NPC QA TEMPLATES
# ============================================================
def generate_npc_qa(npcs):
    entries = []
    for npc in npcs:
        n = npc["name"]
        desc = safe(npc.get("description"), "")
        lore = safe(npc.get("lore"), "")
        loc = safe(npc.get("location"), "Unknown")
        role = safe(npc.get("role"), "Unknown")

        # 1. Description
        if desc and desc != "Unknown":
            q = pick([
                f"Who is {n} in Elden Ring?",
                f"Tell me about {n}.",
                f"What do you know about {n}?",
            ])
            full = desc
            if lore and lore != desc:
                full += f" {lore}"
            entries.append(make_entry(q, full, "npc", "lore", n))

        # 2. Location
        if loc and loc != "Unknown":
            q = pick([
                f"Where can I find {n}?",
                f"Where is {n} located?",
                f"What's the location of {n}?",
            ])
            a = f"{n} can be found at {loc}."
            entries.append(make_entry(q, a, "npc", "location", n))

        # 3. Role
        if role and role != "Unknown":
            q = pick([
                f"What does {n} do?",
                f"What is {n}'s role?",
                f"What services does {n} offer?",
            ])
            a = f"{n} serves as a {role}."
            if loc and loc != "Unknown":
                a += f" They can be found at {loc}."
            entries.append(make_entry(q, a, "npc", "role", n))

    return entries


# ============================================================
# LOCATION QA TEMPLATES
# ============================================================
def generate_location_qa(locations):
    entries = []
    for loc in locations:
        n = loc["name"]
        desc = safe(loc.get("description"), "")
        region = safe(loc.get("region"), "Unknown")
        bosses = loc.get("bosses", [])
        npcs_list = loc.get("npcs", [])
        creatures = loc.get("creatures", [])
        items = loc.get("items", [])

        # 1. Description
        if desc and desc != "Unknown":
            q = pick([
                f"Tell me about {n}.",
                f"Describe {n} in Elden Ring.",
                f"What is {n}?",
            ])
            entries.append(make_entry(q, desc, "location", "lore", n))

        # 2. Region
        if region and region != "Unknown":
            q = pick([
                f"What region is {n} in?",
                f"Where is {n} located?",
            ])
            a = f"{n} is located in the {region} region."
            entries.append(make_entry(q, a, "location", "region", n))

        # 3. Bosses at location
        if bosses:
            q = pick([
                f"What bosses are in {n}?",
                f"Are there any bosses at {n}?",
                f"Who do I fight at {n}?",
            ])
            a = f"The bosses found at {n} include: {fmt_list(bosses)}."
            entries.append(make_entry(q, a, "location", "bosses", n))

        # 4. NPCs at location
        if npcs_list:
            q = pick([
                f"What NPCs are at {n}?",
                f"Who can I find at {n}?",
                f"What characters are in {n}?",
            ])
            a = f"NPCs found at {n} include: {fmt_list(npcs_list)}."
            entries.append(make_entry(q, a, "location", "npcs", n))

        # 5. Notable items
        if items:
            q = pick([
                f"What items can I find at {n}?",
                f"What loot is at {n}?",
                f"What can I pick up at {n}?",
            ])
            a = f"Notable items at {n} include: {fmt_list(items)}."
            entries.append(make_entry(q, a, "location", "items", n))

        # 6. Creatures
        if creatures:
            q = pick([
                f"What enemies are at {n}?",
                f"What creatures lurk in {n}?",
            ])
            a = f"Enemies found at {n} include: {fmt_list(creatures)}."
            entries.append(make_entry(q, a, "location", "creatures", n))

    return entries


# ============================================================
# ARMOR QA TEMPLATES
# ============================================================
def generate_armor_qa(armors):
    entries = []
    for a in armors:
        n = a["name"]
        desc = safe(a.get("description"), "")
        lore = safe(a.get("lore"), "")
        atype = safe(a.get("type"), "Unknown")
        weight = a.get("weight", 0)
        dmg_neg = a.get("damage_negation", {})
        res = a.get("resistance", {})
        special = safe(a.get("special_effect"), "None")
        acquire = safe(a.get("how_to_acquire"), "Unknown")

        # 1. Description
        if desc and desc != "Unknown":
            q = pick([
                f"What is {n} in Elden Ring?",
                f"Tell me about the {n} armor.",
                f"Describe {n}.",
            ])
            full = desc
            if lore and lore != desc:
                full += f" {lore}"
            entries.append(make_entry(q, full, "armor", "lore", n))

        # 2. Stats
        if dmg_neg:
            q = pick([
                f"What are the defensive stats of {n}?",
                f"How good is {n} for defense?",
                f"What protection does {n} offer?",
            ])
            ans = f"{n} is a {atype} weighing {weight} units. Damage negation: {fmt_negation(dmg_neg)}."
            if res:
                ans += f" Resistances: {fmt_resistance(res)}."
            entries.append(make_entry(q, ans, "armor", "stats", n))

        # 3. How to acquire
        if acquire and acquire != "Unknown":
            q = pick([
                f"How do I get {n}?",
                f"Where can I find {n}?",
                f"How do I obtain the {n}?",
            ])
            entries.append(make_entry(q, acquire, "armor", "acquisition", n))

        # 4. Special effect
        if special and special != "None":
            q = pick([
                f"Does {n} have any special effects?",
                f"What's special about {n}?",
            ])
            ans = f"{n} has the following special effect: {special}."
            entries.append(make_entry(q, ans, "armor", "special", n))

    return entries


# ============================================================
# CREATURE QA TEMPLATES
# ============================================================
def generate_creature_qa(creatures):
    entries = []
    for c in creatures:
        n = c["name"]
        desc = safe(c.get("description"), "")
        lore = safe(c.get("lore"), "")
        locs = c.get("locations", [])
        drops = c.get("drops", [])

        # 1. Description
        if desc and desc != "Unknown":
            q = pick([
                f"What is a {n} in Elden Ring?",
                f"Tell me about the {n} enemy.",
                f"Describe the {n}.",
            ])
            full = desc
            if lore and lore != desc:
                full += f" {lore}"
            entries.append(make_entry(q, full, "creature", "lore", n))

        # 2. Location
        if locs:
            q = pick([
                f"Where can I find {n}?",
                f"Where do {n} enemies appear?",
                f"What locations have {n}?",
            ])
            a = f"{n} can be found at: {fmt_list(locs)}."
            entries.append(make_entry(q, a, "creature", "location", n))

        # 3. Drops
        if drops:
            q = pick([
                f"What does {n} drop?",
                f"What loot do I get from {n}?",
                f"What items does {n} drop?",
            ])
            a = f"{n} can drop: {fmt_list(drops)}."
            entries.append(make_entry(q, a, "creature", "drops", n))

    return entries


# ============================================================
# ASHES OF WAR QA TEMPLATES
# ============================================================
def generate_ash_qa(ashes):
    entries = []
    for a in ashes:
        n = a["name"]
        desc = safe(a.get("description"), "")
        affinity = safe(a.get("affinity"), "Standard")
        skill = safe(a.get("skill"), "Unknown")

        # 1. Description
        if desc and desc != "Unknown":
            q = pick([
                f"What is {n}?",
                f"Tell me about {n}.",
                f"What does {n} do?",
            ])
            entries.append(make_entry(q, desc, "ash_of_war", "lore", n))

        # 2. Affinity + Skill
        q = pick([
            f"What skill does {n} grant?",
            f"What affinity does {n} give?",
        ])
        ans = f"{n} grants the {affinity} affinity and the {skill} skill."
        entries.append(make_entry(q, ans, "ash_of_war", "skill", n))

    return entries


# ============================================================
# SKILL QA TEMPLATES
# ============================================================
def generate_skill_qa(skills):
    entries = []
    for s in skills:
        n = s["name"]
        effect = safe(s.get("effect"), "")
        fp = safe(s.get("fp_cost"), "0")
        equip = safe(s.get("equipment"), "Unknown")
        chargeable = safe(s.get("chargeable"), "No")
        stype = safe(s.get("type"), "Regular")

        # 1. Effect
        if effect and effect != "Unknown":
            q = pick([
                f"What does the {n} skill do?",
                f"How does {n} work?",
                f"Tell me about the {n} skill.",
            ])
            a = f"{n} is a {stype} skill. {effect}"
            if fp and fp != "0":
                a += f" It costs {fp} FP."
            entries.append(make_entry(q, a, "skill", "effect", n))

        # 2. Equipment compatibility
        if equip and equip != "Unknown":
            q = pick([
                f"What weapons can use {n}?",
                f"What's {n} compatible with?",
            ])
            a = f"{n} is {equip}."
            if chargeable == "Yes":
                a += " This skill can be charged."
            entries.append(make_entry(q, a, "skill", "equipment", n))

    return entries


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("🗡️  Elden Ring QA Generator")
    print("=" * 60)

    # Load enriched data
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded enriched data: {INPUT_PATH}")
    for k, v in data.get("metadata", {}).items():
        print(f"  {k}: {v}")

    # Generate QA pairs for each entity type
    all_entries = []

    print("\nGenerating QA pairs...")

    weapons_qa = generate_weapon_qa(data.get("weapons", []))
    print(f"  Weapons: {len(weapons_qa)} pairs")
    all_entries.extend(weapons_qa)

    bosses_qa = generate_boss_qa(data.get("bosses", []))
    print(f"  Bosses: {len(bosses_qa)} pairs")
    all_entries.extend(bosses_qa)

    sorceries_qa = generate_magic_qa(data.get("sorceries", []), "sorcery")
    print(f"  Sorceries: {len(sorceries_qa)} pairs")
    all_entries.extend(sorceries_qa)

    incantations_qa = generate_magic_qa(data.get("incantations", []), "incantation")
    print(f"  Incantations: {len(incantations_qa)} pairs")
    all_entries.extend(incantations_qa)

    npcs_qa = generate_npc_qa(data.get("npcs", []))
    print(f"  NPCs: {len(npcs_qa)} pairs")
    all_entries.extend(npcs_qa)

    locations_qa = generate_location_qa(data.get("locations", []))
    print(f"  Locations: {len(locations_qa)} pairs")
    all_entries.extend(locations_qa)

    armors_qa = generate_armor_qa(data.get("armors", []))
    print(f"  Armors: {len(armors_qa)} pairs")
    all_entries.extend(armors_qa)

    creatures_qa = generate_creature_qa(data.get("creatures", []))
    print(f"  Creatures: {len(creatures_qa)} pairs")
    all_entries.extend(creatures_qa)

    ashes_qa = generate_ash_qa(data.get("ashes_of_war", []))
    print(f"  Ashes of War: {len(ashes_qa)} pairs")
    all_entries.extend(ashes_qa)

    skills_qa = generate_skill_qa(data.get("skills", []))
    print(f"  Skills: {len(skills_qa)} pairs")
    all_entries.extend(skills_qa)

    # Shuffle
    random.shuffle(all_entries)

    # Print distribution summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total QA pairs: {len(all_entries)}")

    type_counts = defaultdict(int)
    qtype_counts = defaultdict(int)
    for e in all_entries:
        type_counts[e["metadata"]["entity_type"]] += 1
        qtype_counts[e["metadata"]["question_type"]] += 1

    print("\nBy entity type:")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print("\nBy question type:")
    for k, v in sorted(qtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Write JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Written {len(all_entries)} QA pairs to {OUTPUT_PATH}")

    # Print a few samples
    print("\n" + "=" * 60)
    print("SAMPLE ENTRIES")
    print("=" * 60)
    samples = random.sample(all_entries, min(5, len(all_entries)))
    for s in samples:
        print(f"\n  [Entity: {s['metadata']['entity_type']}, QType: {s['metadata']['question_type']}]")
        print(f"  Q: {s['instruction']}")
        print(f"  A: {s['output'][:150]}...")


if __name__ == "__main__":
    main()