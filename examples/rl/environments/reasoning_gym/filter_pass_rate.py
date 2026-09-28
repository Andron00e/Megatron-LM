# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep the reasoning_gym prompts a checkpoint solves sometimes but not always.

Reads the JSONL written by ReasoningGymAgent(probe_output=...) during a probe run
(train_rl.py --perform-rl-step --skip-train, k samples per prompt) and writes the entries with
min_pass < pass rate < max_pass, in the format ReasoningGymAgent(prompts_file=...) reads.

    python filter_pass_rate.py probe.jsonl filtered.jsonl [--min-pass 0] [--max-pass 1]
"""

import argparse
import collections
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", nargs="+", help="probe JSONL file(s)")
    parser.add_argument("output")
    parser.add_argument("--min-pass", type=float, default=0.0)
    parser.add_argument("--max-pass", type=float, default=1.0)
    args = parser.parse_args()

    rewards, entries = collections.defaultdict(list), {}
    for path in args.probe:
        with open(path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    rewards[record["problem_id"]].extend(record["rewards"])
                    entries[record["problem_id"]] = record["entry"]

    kept = 0
    by_task = collections.defaultdict(lambda: [0, 0, 0.0])
    with open(args.output, "w") as out:
        for problem_id, rs in rewards.items():
            # A reward counts as a pass only at full score (format rewards are partial).
            pass_rate = sum(r >= 1.0 for r in rs) / len(rs)
            task = by_task[problem_id.split(":")[0]]
            task[0] += 1
            task[2] += pass_rate
            if args.min_pass < pass_rate < args.max_pass:
                out.write(json.dumps({"problem_id": problem_id, "pass_rate": pass_rate,
                                      "samples": len(rs), "entry": entries[problem_id]}) + "\n")
                task[1] += 1
                kept += 1
    for name, (n, n_kept, total) in sorted(by_task.items()):
        print(f"{name}: {n} prompts, mean pass rate {total / n:.3f}, kept {n_kept}")
    print(f"kept {kept} of {len(rewards)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
