# Principles vs. HarmBench Cross-Check

Internal reference: how our 15 seed principles align with HarmBench's semantic categories.

## HarmBench Semantic Categories

From `harmbench_behaviors_text_all.csv` (400 total behaviors), verified counts:

| Category | Count | Description |
|----------|-------|-------------|
| copyright | 100 | Reproduction/transformation of copyrighted works |
| cybercrime_intrusion | 67 | Hacking, intrusion, credential theft |
| illegal | 65 | Fraud, theft, violence, smuggling, exploitation, etc. |
| misinformation_disinformation | 65 | Defamation, false claims, conspiracy narratives |
| chemical_biological | 56 | Synthesis, pathogens, toxins, dual-use chemistry/biology |
| harassment_bullying | 25 | Coercion, insults, hate speech, discrimination |
| harmful | 22 | Domestic abuse, dangerous product mixing, etc. |

## Our Principles → HarmBench Mapping

| Principle | Focus | HarmBench categories covered |
|-----------|-------|------------------------------|
| P1 | Ambiguous verbs (kill, execute, shoot) | Edge case; helps with benign technical prompts |
| P2 | Dual-use (chemistry, cybersecurity, biology) | chemical_biological, cybercrime_intrusion |
| P3 | Refuse + pivot (general harm) | illegal, harmful |
| P4 | Fiction (narrative-only, no operational) | Contextual / edge case |
| P5 | Self-harm & distress | harmful |
| P6 | Deception, manipulation, gaslighting | harassment_bullying |
| P7 | Physical harm, violence, weapons | illegal, harmful |
| P8 | Fraud, theft, phishing | illegal |
| P9 | Discrimination, bias | harassment_bullying, misinformation |
| P10 | Misinformation creation | misinformation_disinformation |
| P11 | Privacy, surveillance | illegal, cybercrime_intrusion |
| P12 | Illegal drugs, substance abuse | chemical_biological, illegal |
| P13 | Law evasion (tax, laundering) | illegal |
| P14 | Weapons, explosives, hazardous materials | illegal, chemical_biological |
| P15 | Exploitation of vulnerable populations | illegal, harassment_bullying |

## Gap

- **Copyright**: HarmBench has 100 copyright prompts (reproduction of copyrighted works). We do not have an explicit copyright-refusal principle. P13 (law evasion) may partially cover some cases but is not targeted at copyright.

## Usage

Run `python scripts/analyze_seed_overlap.py` with HarmBench behaviors as the eval set to quantify prompt overlap. To use HarmBench CSV as eval prompts:

```bash
# After downloading harmbench_behaviors_text_all.csv to data/harmbench_cache/
# extend analyze_seed_overlap.py to load Behavior column from CSV
```
