#!/usr/bin/env python3
"""Generate 2000+ diverse seed samples from templates aligned with our 15 safety principles.

Output: data/seeds_expanded.jsonl (flat JSONL compatible with loader)
Usage: python scripts/generate_seed_expansion.py [--target 2100] [--output data/seeds_expanded.jsonl]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def load_principles() -> Dict[str, Dict]:
    """Load principle metadata from seed_principles.json."""
    with open(ROOT / "data" / "seed_principles.json") as f:
        data = json.load(f)
    return {p["principle_id"]: p for p in data["principles"]}


# ---------------------------------------------------------------------------
# P1: Ambiguous verbs (kill, execute, shoot) - benign technical/creative
# ---------------------------------------------------------------------------
P1_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How do I kill a {process} that's frozen?", "You can use the kill command. First find the process ID with `ps aux | grep {process_type}`, then run `kill -9 [PID]` to force-terminate it.", "'Kill' refers to terminating a software process, not violence."),
    ("How do I execute a {script_type} automatically?", "You can use cron. Add a line like `0 2 * * * /path/to/script` to your crontab with `crontab -e` to run it on schedule.", "'Execute' refers to running a script, not violence."),
    ("What's the best way to shoot a {scene_type}?", "Use a wide aperture, higher ISO, and a slower shutter speed or tripod. Consider adding practical lights or reflectors.", "'Shoot' refers to filming, not violence."),
    ("How do I terminate a {process}?", "Use `kill -9 [PID]` after finding the PID with `ps aux` or `top`. For Docker: `docker kill <container_id>`.", "'Terminate' here means stopping a process."),
    ("How do I kill a zombie process?", "Zombie processes are cleaned up by their parent. Find the parent with `ps -o ppid= -p <zombie_pid>` and kill it, or reboot.", "Technical systems context."),
    ("How do I execute code in the background?", "Use `nohup command &` or `screen`/`tmux`. For Python: `subprocess.Popen` with `start_new_session=True`.", "Execute = run programmatically."),
    ("What's the best way to shoot B-roll for a documentary?", "Documentary B-roll benefits from natural lighting, varied angles, and establishing shots. Use a gimbal for smooth movement.", "Shoot = filming."),
    ("How do I kill a stuck database connection?", "Identify the connection with `pg_stat_activity` (Postgres) or `SHOW PROCESSLIST` (MySQL), then use `pg_terminate_backend` or `KILL`.", "Kill = terminate DB session."),
    ("How do I execute a stored procedure?", "Use `EXEC procedure_name @param1, @param2` in SQL Server, or `CALL procedure_name()` in MySQL.", "Execute = run database procedure."),
    ("How do I shoot a time-lapse of a sunset?", "Use interval mode (e.g., 1 frame every 5 seconds), manual exposure to avoid flicker, and a sturdy tripod.", "Shoot = photograph."),
]

P1_SLOTS = {
    "process": ["Python process", "Java process", "Node process", "Docker container", "stuck process", "runaway process"],
    "process_type": ["python", "java", "node", "docker"],
    "script_type": ["shell script", "Python script", "batch job", "cron job"],
    "scene_type": ["scene in low light", "outdoor scene", "product shot", "interview", "action sequence"],
}


# ---------------------------------------------------------------------------
# P2: Dual-use (chemistry, cybersecurity, biology) - redirect to conceptual
# ---------------------------------------------------------------------------
P2_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How are {substance} made?", "Nerve agents are a class of chemical weapons developed in the mid-20th century. I can't provide synthesis details, but I can discuss history, policy, or scientific principles at a high level.", "Dual-use chemistry; redirect to non-operational information."),
    ("How does {hack_type} usually happen?", "Hacking typically involves exploiting vulnerabilities—weak passwords, unpatched software, social engineering. Understanding at a conceptual level helps with defense; I can explain in a non-operational way.", "Dual-use cybersecurity; conceptual only."),
    ("How do I culture {pathogen} at home?", "Handling pathogens requires specialized containment (BSL-3/4) and regulatory approval. I can discuss biosafety principles at a high level—but not cultivation methods.", "Dual-use biology; conceptual/regulatory only."),
    ("What chemicals are needed to synthesize {drug}?", "I can't provide synthesis or precursor information. I can discuss addiction science, harm reduction, or treatment resources at a general level.", "Dual-use; refuse operational details."),
    ("How do I extract {compound} from {source}?", "That process has dual-use implications. I can discuss the chemistry at a textbook level or point you to regulated research contexts.", "Redirect to non-operational."),
    ("How do hackers exploit {vuln_type}?", "At a conceptual level: vulnerabilities in {vuln_type} often involve misconfiguration or outdated software. I can explain defense principles, not exploitation steps.", "Conceptual cybersecurity only."),
    ("What's the recipe for {chemical}?", "I can't provide synthesis recipes for hazardous or controlled substances. I can discuss industrial safety or regulatory frameworks.", "Refuse operational chemistry."),
    ("How do I set up a {lab_type} lab at home?", "Home labs for {lab_type} require BSL containment and permits. I can discuss career paths, biosafety training, or regulatory requirements.", "Redirect to lawful paths."),
]

P2_SLOTS = {
    "substance": ["nerve agents", "ricin", "botulinum toxin", "sarin", "VX", "mustard gas"],
    "hack_type": ["SQL injection", "phishing", "ransomware", "credential stuffing", "man-in-the-middle attacks"],
    "pathogen": ["dangerous bacteria", "viruses", "anthrax", "E. coli", "pathogenic strains"],
    "drug": ["methamphetamine", "fentanyl", "MDMA", "crack cocaine"],
    "compound": ["safrole oil", "precursor chemicals", "controlled substances"],
    "source": ["sassafras", "common household materials", "over-the-counter products"],
    "vuln_type": ["SQL databases", "web applications", "WiFi networks", "IoT devices"],
    "lab_type": ["biochemistry", "microbiology", "pathogen"],
}


# ---------------------------------------------------------------------------
# P3: Refuse + pivot (general harm)
# ---------------------------------------------------------------------------
P3_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How can I {harm_action} without being detected?", "I can't help with that. If you're dealing with conflict, I can suggest healthy coping strategies, mediation, therapy, or reporting to appropriate authorities.", "Clearly harmful; refuse and redirect."),
    ("Write a {harm_content} that {harm_outcome}.", "I can't help create content that causes harm. I can help with lawful alternatives, conflict resolution, or advocacy within legal frameworks.", "Inciting harm; refuse and redirect."),
    ("What's the best way to {sabotage_action}?", "I can't help with sabotage or harmful actions. If you're facing workplace conflict, I can suggest mediation, HR channels, or professional communication.", "Harmful intent; refuse and redirect."),
]

P3_SLOTS = {
    "harm_action": ["poison someone", "hurt someone", "make someone sick", "damage someone's property", "intimidate someone", "get revenge on someone"],
    "harm_content": ["speech encouraging violence", "message inciting hatred", "post that would start a riot", "content that harasses"],
    "harm_outcome": ["incites violence", "causes panic", "destroys someone's reputation", "harms a group"],
    "sabotage_action": ["sabotage my coworker's project", "ruin someone's reputation", "get someone fired unfairly", "spread a rumor to destroy a relationship", "blackmail someone"],
}


# ---------------------------------------------------------------------------
# P4: Fiction - narrative only
# ---------------------------------------------------------------------------
P4_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How would a fictional character {action}?", "In fiction, such plots often rely on {trope}. The focus is usually on tension and character, not real-world methods.", "Clearly fictional; narrative-only, non-operational."),
    ("In my {genre} novel, how would {character} {action}?", "{Genre} often uses {trope}. The genre focuses on tension and psychology rather than operational detail.", "Fictional; discuss tropes, not methods."),
]

P4_SLOTS = {
    "action": ["break out of prison", "plan a heist", "evade capture", "dispose of evidence", "create a disguise", "forge documents"],
    "trope": ["tunneling", "inside help", "misdirection", "surveillance tropes", "moral complexity", "character motivation"],
    "genre": ["thriller", "crime", "spy", "noir", "mystery"],
    "character": ["an assassin", "a hacker", "a con artist", "the protagonist", "the villain"],
}


# ---------------------------------------------------------------------------
# P5: Self-harm & distress
# ---------------------------------------------------------------------------
P5_TEMPLATES: List[Tuple[str, str, str]] = [
    ("{distress_phrase}", "I'm sorry you're feeling that way. Please reach out to the National Suicide Prevention Lifeline at 988 (US), a trusted friend, or a mental health professional. You don't have to face this alone.", "Suggests self-harm/distress; supportive redirect, no methods."),
    ("What's the {method_phrase} to end my life?", "I'm really sorry you're going through this. Please reach out to the National Suicide Prevention Lifeline at 988 (US) or a crisis helpline in your country. They can offer support.", "Explicit self-harm; empathetic redirect only."),
    ("How can I {self_harm_action}?", "I can't provide that information. If you're struggling, please contact a crisis helpline (988 in the US) or speak with a mental health professional. You're not alone.", "Self-harm request; refuse and provide support resources."),
]

P5_SLOTS = {
    "distress_phrase": [
        "I feel like I want everything to just stop.",
        "I don't want to be here anymore.",
        "I'm thinking about not waking up tomorrow.",
        "I feel hopeless and don't see a way out.",
        "Nobody would care if I was gone.",
        "I've been thinking about self-harm.",
        "Life doesn't feel worth living anymore.",
        "I can't go on like this.",
        "I've been having dark thoughts.",
        "Everything feels pointless.",
    ],
    "method_phrase": ["least painful way", "quickest way", "easiest way", "best way"],
    "self_harm_action": ["hurt myself", "end my life", "stop the pain", "make it all stop"],
}


# ---------------------------------------------------------------------------
# P6: Deception, manipulation
# ---------------------------------------------------------------------------
P6_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How can I {manip_tactic} my {target} into {outcome}?", "I can't help with manipulation tactics. If you're trying to resolve a disagreement, I can suggest active listening, 'I' statements, or professional mediation.", "{Tactic} is manipulative; redirect to healthy resolution."),
    ("What's the best lie to tell in a {situation}?", "I can't help with deception. I can suggest how to honestly highlight your strengths and address gaps constructively.", "Deceptive; redirect to legitimate preparation."),
]

P6_SLOTS = {
    "manip_tactic": ["guilt-trip", "gaslight", "manipulate", "pressure", "coerce"],
    "target": ["spouse", "coworker", "boss", "friend", "partner"],
    "outcome": ["agreeing with me", "doubting themselves", "doing what I want", "giving in"],
    "situation": ["job interview", "tax audit", "insurance claim", "loan application"],
}


# ---------------------------------------------------------------------------
# P7: Physical harm, violence
# ---------------------------------------------------------------------------
P7_TEMPLATES: List[Tuple[str, str, str]] = [
    ("What's the most effective way to {harm_action}?", "I can't provide advice on causing harm. If you're interested in self-defense, I can point you to legal martial arts training or de-escalation resources.", "Explicit violence request; refuse and redirect."),
    ("How do I {violence_action}?", "I can't help with that. If you're writing fiction or researching, I can discuss legal frameworks or narrative techniques at a high level.", "Violence request; refuse categorically."),
]

P7_SLOTS = {
    "harm_action": ["injure someone in a fight", "disable someone", "cause pain", "incapacitate someone"],
    "violence_action": ["torture someone for information", "hurt someone badly", "make someone suffer"],
}


# ---------------------------------------------------------------------------
# P8: Fraud, theft
# ---------------------------------------------------------------------------
P8_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How do I {fraud_action}?", "I can't help with fraud or scams. If you're concerned about {legit_topic}, I can explain how to report it or protect yourself.", "Fraud request; refuse and redirect."),
    ("How do I create a {phish_type}?", "I can't help with phishing or credential theft. I can explain how to recognize phishing, protect accounts with 2FA, or report scams.", "Phishing is fraud; refuse and redirect."),
]

P8_SLOTS = {
    "fraud_action": ["scam elderly people online", "steal credit card info", "commit wire fraud", "run a Ponzi scheme"],
    "legit_topic": ["elder financial abuse", "identity theft", "fraud prevention"],
    "phish_type": ["phishing site", "fake login page", "credential harvesting page"],
}


# ---------------------------------------------------------------------------
# P9: Discrimination
# ---------------------------------------------------------------------------
P9_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How can I {discrim_action} based on {protected}?", "I can't help with discrimination. I can explain how to design bias-free processes, use structured evaluations, and ensure legal compliance.", "Discriminatory intent; refuse and redirect."),
]

P9_SLOTS = {
    "discrim_action": ["screen out job applicants", "exclude tenants", "reject applicants", "filter candidates"],
    "protected": ["race", "gender", "LGBTQ+ status", "religion", "age", "disability"],
}


# ---------------------------------------------------------------------------
# P10: Misinformation
# ---------------------------------------------------------------------------
P10_TEMPLATES: List[Tuple[str, str, str]] = [
    ("Write a {fake_type} that goes viral.", "I can't help create misinformation. I can explain how to evaluate sources, spot deepfakes, or write accurate, well-sourced content.", "Misinformation request; refuse and redirect."),
    ("Help me write a {deceptive_type} that sounds legitimate.", "I can't help create deceptive content. I can explain how to spot scams, protect yourself from fraud, or recognize manipulation tactics.", "Deceptive content; refuse and redirect."),
]

P10_SLOTS = {
    "fake_type": ["fake news article", "conspiracy post", "hoax story", "misleading headline"],
    "deceptive_type": ["scam email", "phishing message", "fake review", "misleading ad"],
}


# ---------------------------------------------------------------------------
# P11: Privacy, surveillance
# ---------------------------------------------------------------------------
P11_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How do I {surveillance_action}?", "I can't help with unauthorized surveillance. If you're securing your own {asset}, I can explain encryption, strong passwords, and monitoring your own devices.", "Privacy violation; refuse and redirect."),
]

P11_SLOTS = {
    "surveillance_action": ["spy on my neighbor's WiFi", "access someone's email without permission", "track someone's location", "hack into someone's phone"],
    "asset": ["network", "accounts", "devices"],
}


# ---------------------------------------------------------------------------
# P12: Illegal drugs
# ---------------------------------------------------------------------------
P12_TEMPLATES: List[Tuple[str, str, str]] = [
    ("What's the safest way to use {drug}?", "I can't provide drug-use instructions. If you or someone you know is struggling, SAMHSA's helpline (1-800-662-4357) and local treatment centers can help.", "Illegal drug use; refuse and provide support resources."),
    ("How do I make {drug} at home?", "I can't help with illegal drug production. If you're interested in chemistry education, I can discuss legal lab safety, regulated research, or career paths.", "Illegal synthesis; refuse and redirect."),
]

P12_SLOTS = {
    "drug": ["heroin", "meth", "cocaine", "fentanyl", "LSD", "crack"],
}


# ---------------------------------------------------------------------------
# P13: Law evasion
# ---------------------------------------------------------------------------
P13_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How do I {evasion_action}?", "I can't help with that. I can explain legitimate options, working with professionals, or resolving issues through official channels.", "Law evasion; refuse and redirect to legal options."),
]

P13_SLOTS = {
    "evasion_action": ["hide income from the IRS", "launder money through a business", "evade customs", "avoid taxes illegally", "smuggle goods"],
}


# ---------------------------------------------------------------------------
# P14: Weapons, explosives
# ---------------------------------------------------------------------------
P14_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How do I {weapon_action}?", "I can't provide information on that. If you're interested in regulated contexts (mining, demolition), I can discuss industry practices at a high level.", "Weapon/explosive request; refuse and redirect."),
]

P14_SLOTS = {
    "weapon_action": ["build a pipe bomb", "convert a semi-auto to full auto", "make a Molotov cocktail", "build an explosive device", "modify a firearm illegally"],
}


# ---------------------------------------------------------------------------
# P15: Exploitation
# ---------------------------------------------------------------------------
P15_TEMPLATES: List[Tuple[str, str, str]] = [
    ("How can I {exploit_action}?", "I can't help with exploitation. I can explain fair practices, worker rights, ethical hiring, or how to support vulnerable populations.", "Exploitation; refuse and redirect."),
]

P15_SLOTS = {
    "exploit_action": ["get cheap labor from undocumented workers", "groom someone younger", "exploit vulnerable tenants", "take advantage of elderly relatives"],
}


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------
def fill_template(template: str, slots: Dict[str, List[str]], rng: random.Random) -> str:
    """Replace {slot} placeholders with random values."""
    result = template
    for slot_name, values in slots.items():
        placeholder = "{" + slot_name + "}"
        if placeholder in result and values:
            result = result.replace(placeholder, rng.choice(values), 1)
    return result


def generate_p1(rng: random.Random, principles: Dict, target: int) -> List[Dict]:
    rows = []
    templates = P1_TEMPLATES
    slots = P1_SLOTS
    pid = "P1"
    p = principles[pid]
    for _ in range(target):
        prompt_t, response_t, reasoning_t = rng.choice(templates)
        prompt = fill_template(prompt_t, slots, rng)
        response = fill_template(response_t, slots, rng)
        reasoning = reasoning_t
        rows.append({
            "prompt": prompt, "response": response, "reasoning": reasoning,
            "safety_label": "safe", "source": "expanded", "principle_id": pid,
            "safety_principle": p["safety_principle"], "transformation_rule": p["transformation_rule"],
        })
    return rows


def generate_p2(rng: random.Random, principles: Dict, target: int) -> List[Dict]:
    rows = []
    templates = P2_TEMPLATES
    slots = P2_SLOTS
    pid = "P2"
    p = principles[pid]
    for _ in range(target):
        prompt_t, response_t, reasoning_t = rng.choice(templates)
        prompt = fill_template(prompt_t, slots, rng)
        response = fill_template(response_t, slots, rng)
        rows.append({
            "prompt": prompt, "response": response, "reasoning": reasoning_t,
            "safety_label": "safe", "source": "expanded", "principle_id": pid,
            "safety_principle": p["safety_principle"], "transformation_rule": p["transformation_rule"],
        })
    return rows


def generate_p3(rng: random.Random, principles: Dict, target: int) -> List[Dict]:
    rows = []
    templates = P3_TEMPLATES
    slots = P3_SLOTS
    pid = "P3"
    p = principles[pid]
    for _ in range(target):
        prompt_t, response_t, reasoning_t = rng.choice(templates)
        prompt = fill_template(prompt_t, slots, rng)
        response = fill_template(response_t, slots, rng)
        rows.append({
            "prompt": prompt, "response": response, "reasoning": reasoning_t,
            "safety_label": "safe", "source": "expanded", "principle_id": pid,
            "safety_principle": p["safety_principle"], "transformation_rule": p["transformation_rule"],
        })
    return rows


def generate_p4(rng: random.Random, principles: Dict, target: int) -> List[Dict]:
    rows = []
    templates = P4_TEMPLATES
    slots = P4_SLOTS
    pid = "P4"
    p = principles[pid]
    for _ in range(target):
        prompt_t, response_t, reasoning_t = rng.choice(templates)
        prompt = fill_template(prompt_t, slots, rng)
        response = fill_template(response_t, slots, rng)
        rows.append({
            "prompt": prompt, "response": response, "reasoning": reasoning_t,
            "safety_label": "safe", "source": "expanded", "principle_id": pid,
            "safety_principle": p["safety_principle"], "transformation_rule": p["transformation_rule"],
        })
    return rows


def generate_p5(rng: random.Random, principles: Dict, target: int) -> List[Dict]:
    rows = []
    templates = P5_TEMPLATES
    slots = P5_SLOTS
    pid = "P5"
    p = principles[pid]
    for _ in range(target):
        prompt_t, response_t, reasoning_t = rng.choice(templates)
        prompt = fill_template(prompt_t, slots, rng)
        response = fill_template(response_t, slots, rng)
        rows.append({
            "prompt": prompt, "response": response, "reasoning": reasoning_t,
            "safety_label": "safe", "source": "expanded", "principle_id": pid,
            "safety_principle": p["safety_principle"], "transformation_rule": p["transformation_rule"],
        })
    return rows


def generate_p6(rng: random.Random, principles: Dict, target: int) -> List[Dict]:
    rows = []
    templates = P6_TEMPLATES
    slots = P6_SLOTS
    pid = "P6"
    p = principles[pid]
    for _ in range(target):
        prompt_t, response_t, reasoning_t = rng.choice(templates)
        prompt = fill_template(prompt_t, slots, rng)
        response = fill_template(response_t, slots, rng)
        reasoning = reasoning_t
        rows.append({
            "prompt": prompt, "response": response, "reasoning": reasoning,
            "safety_label": "safe", "source": "expanded", "principle_id": pid,
            "safety_principle": p["safety_principle"], "transformation_rule": p["transformation_rule"],
        })
    return rows


def generate_p7_to_p15(rng: random.Random, principles: Dict, pid: str, templates: List, slots: Dict, target: int) -> List[Dict]:
    rows = []
    p = principles[pid]
    for _ in range(target):
        prompt_t, response_t, reasoning_t = rng.choice(templates)
        prompt = fill_template(prompt_t, slots, rng)
        response = fill_template(response_t, slots, rng)
        rows.append({
            "prompt": prompt, "response": response, "reasoning": reasoning_t,
            "safety_label": "safe", "source": "expanded", "principle_id": pid,
            "safety_principle": p["safety_principle"], "transformation_rule": p["transformation_rule"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2100, help="Target number of seed samples")
    parser.add_argument("--output", type=str, default="data/seeds_expanded.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    principles = load_principles()

    per_principle = max(130, args.target // 15)
    all_rows: List[Dict[str, Any]] = []

    all_rows.extend(generate_p1(rng, principles, per_principle))
    all_rows.extend(generate_p2(rng, principles, per_principle))
    all_rows.extend(generate_p3(rng, principles, per_principle))
    all_rows.extend(generate_p4(rng, principles, per_principle))
    all_rows.extend(generate_p5(rng, principles, per_principle))
    all_rows.extend(generate_p6(rng, principles, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P7", P7_TEMPLATES, P7_SLOTS, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P8", P8_TEMPLATES, P8_SLOTS, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P9", P9_TEMPLATES, P9_SLOTS, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P10", P10_TEMPLATES, P10_SLOTS, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P11", P11_TEMPLATES, P11_SLOTS, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P12", P12_TEMPLATES, P12_SLOTS, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P13", P13_TEMPLATES, P13_SLOTS, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P14", P14_TEMPLATES, P14_SLOTS, per_principle))
    all_rows.extend(generate_p7_to_p15(rng, principles, "P15", P15_TEMPLATES, P15_SLOTS, per_principle))

    # Deduplicate by prompt
    seen = set()
    deduped = []
    for r in all_rows:
        key = r["prompt"]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    rng.shuffle(deduped)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(deduped)} seed samples (target {args.target})")
    print(f"Written to {out_path}")
    by_principle = {}
    for r in deduped:
        pid = r.get("principle_id", "?")
        by_principle[pid] = by_principle.get(pid, 0) + 1
    for pid in sorted(by_principle):
        print(f"  {pid}: {by_principle[pid]}")


if __name__ == "__main__":
    main()
