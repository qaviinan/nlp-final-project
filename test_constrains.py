from inspect_ai.log import read_eval_log
import collections

log = read_eval_log("evaluation/inspect-logs/2026-04-11T00-52-06+00-00_ifeval_VEwuBk2XaNvhScGvhVtP5K.eval")

failed_constraints = collections.Counter()
passed_constraints = collections.Counter()

for sample in log.samples:
    score = sample.scores.get("instruction_following")
    if not score:
        continue
    
    constraints = sample.metadata.get("instruction_id_list", [])
    inst_strict = score.value.get("inst_level_strict")
    prompt_pass = score.value.get("prompt_level_strict", False)

    # Single constraint case — inst_level_strict is just 0 or 1
    if len(constraints) == 1:
        if inst_strict == 1:
            passed_constraints[constraints[0]] += 1
        else:
            failed_constraints[constraints[0]] += 1
    # Multiple constraints — prompt_level_strict tells us if ALL passed
    # but we don't know which individual ones failed, so track at prompt level
    elif len(constraints) > 1:
        for c in constraints:
            if prompt_pass:
                passed_constraints[c] += 1
            else:
                failed_constraints[c] += 1

print("TOP FAILED CONSTRAINTS:")
for constraint, count in failed_constraints.most_common(20):
    total = failed_constraints[constraint] + passed_constraints[constraint]
    pct = count / total * 100
    print(f"  {constraint}: {count}/{total} failed ({pct:.0f}%)")

print("\nTOP PASSED CONSTRAINTS:")
for constraint, count in passed_constraints.most_common(15):
    total = failed_constraints[constraint] + passed_constraints[constraint]
    pct = count / total * 100
    print(f"  {constraint}: {count}/{total} passed ({pct:.0f}%)")