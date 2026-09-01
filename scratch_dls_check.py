import json, glob

files = glob.glob('data/raw/t20s_male_json/*.json')
count = 0
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    info = d.get('info', {})
    outcome = info.get('outcome', {})
    method = outcome.get('method', None)
    if method == 'D/L' and count < 3:
        innings = d.get('innings', [])
        print(f"=== {f} ===")
        print(f"  Outcome: {outcome}")
        print(f"  Innings count: {len(innings)}")
        for i, inn in enumerate(innings):
            team = inn.get('team', '?')
            target = inn.get('target', 'NONE')
            overs_data = inn.get('overs', [])
            print(f"  Innings {i}: team={team}, target={target}, overs_count={len(overs_data)}")
        print()
        count += 1
    if count >= 3:
        break

# Also check a non-DLS match for target key
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    info = d.get('info', {})
    outcome = info.get('outcome', {})
    method = outcome.get('method', None)
    if method is None and len(d.get('innings', [])) == 2:
        innings = d.get('innings', [])
        print(f"=== NON-DLS: {f} ===")
        for i, inn in enumerate(innings):
            team = inn.get('team', '?')
            target = inn.get('target', 'NONE')
            print(f"  Innings {i}: team={team}, target={target}")
        print()
        break

# Check how many innings have a 'target' key
has_target = 0
no_target = 0
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    innings = d.get('innings', [])
    if len(innings) >= 2:
        if 'target' in innings[1]:
            has_target += 1
        else:
            no_target += 1
print(f"2nd innings with target key: {has_target}")
print(f"2nd innings without target key: {no_target}")
