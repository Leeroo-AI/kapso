"""Draw the sealed set: conversations the build never sees, for the check after it.

    python runner/split.py <runner dir> --development repo/bench/dev_tasks.json --sealed 20 --seed 7

The development conversations are the ones the repository lists (the recorded runs under bench/reference/ are
on them). The sealed set is a seeded sample of the banking domain's other tasks. The split is written to
<runner dir>/split.json (`seed`, `dev`, `sealed`, `unused`), inside the runner's own directory, which the
repository's protection list names. Prints counts and a hash of the sealed list, never a sealed id: commit
the hash wherever you keep such records, so the set cannot be changed after a result is known.
"""
import argparse
import hashlib
import json
import pathlib
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runner_dir", type=pathlib.Path, help="where runner/install.sh put the runner")
    parser.add_argument("--development", type=pathlib.Path, required=True, help="the repository's bench/dev_tasks.json")
    parser.add_argument("--sealed", type=int, required=True, help="how many conversations to seal")
    parser.add_argument("--seed", type=int, required=True, help="the seed of the draw")
    args = parser.parse_args()
    registry = args.runner_dir / "tau2-bench-2.0.0" / "data" / "tau2" / "domains" / "banking_knowledge" / "tasks.json"
    target = args.runner_dir / "split.json"
    if not registry.is_file():
        raise SystemExit(f"no task registry at {registry}; run runner/install.sh first")
    if target.exists():
        raise SystemExit(f"{target} exists: a split is drawn once; remove it by hand to draw another")
    every = sorted(task["id"] for task in json.loads(registry.read_text(encoding="utf-8")))
    development = json.loads(args.development.read_text(encoding="utf-8"))["development"]
    unknown = sorted(set(development) - set(every))
    if unknown:
        raise SystemExit(f"{len(unknown)} development conversation(s) are not in the runner's registry of {len(every)}; "
                         "is this the full registry, before runner/seal.py?")
    rest = [identifier for identifier in every if identifier not in set(development)]
    if not 0 < args.sealed <= len(rest):
        raise SystemExit(f"--sealed must be between 1 and {len(rest)}, the tasks outside the development set")
    sealed = sorted(random.Random(args.seed).sample(rest, args.sealed))
    unused = [identifier for identifier in rest if identifier not in set(sealed)]
    target.write_text(json.dumps({"seed": args.seed, "dev": sorted(development), "sealed": sealed, "unused": unused},
                                 indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(json.dumps(sealed).encode("utf-8")).hexdigest()
    print(f"seed {args.seed}: development {len(development)}, sealed {len(sealed)}, unused {len(unused)}, of {len(every)} tasks")
    print(f"sealed sha256 {digest}")
    print(f"written to {target}; next: python runner/seal.py {args.runner_dir} <vault dir>")


if __name__ == "__main__":
    main()
