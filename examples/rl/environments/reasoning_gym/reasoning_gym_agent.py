# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""reasoning_gym procedural tasks with exact verifiers, as few-shot raw-text prompts for base models.

Prompts use the pretraining text format (no chat template):

    Question: <q1>\\nAnswer: <a1>\\n\\nQuestion: <q2>\\nAnswer: <a2>\\n\\nQuestion: <q>\\nAnswer:

The answer is the generated text up to the first blank line (the next few-shot block starts there)
and is scored by the task's own reasoning_gym verifier. The reward is binary by default. Generation
stops at `stop` (default: the blank line), so a rollout is the answer, not max-length text; such a
rollout is closed with an EOD that is not a sampled token (no loss on it), as the RL data path
expects every rollout to end in EOD or fill the sequence.

Item i of a reasoning_gym dataset is generated from Random(seed + i). Training draws items
[0, size) at `seed`, validation the next `size` items and few-shot examples the `size` after that,
so the three are disjoint.

A pass-rate probe runs this agent with `sequential: true` and `probe_output: <file>` under
train_rl.py --skip-train: every group (one prompt, k samples) is appended to the file with its
rewards; filter_pass_rate.py then keeps the prompts with 0 < pass rate < 1, and
`prompts_file: <filtered file>` trains on exactly those.
"""

import json
import os
import random
from typing import Any

import reasoning_gym

from megatron.rl import GenericGenerationArgs
from megatron.rl.agent.api import TokenRollout
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent

# Sequential position and sampling RNG per item stream. train_rl.py rebuilds the agent for every
# rollout collection (without --rl-partial-rollouts), so both must outlive the instance.
_SEQUENTIAL_POSITION: dict = {}
_STREAM_RNG: dict = {}


class ReasoningGymAgent(RewardOnlyAgent):
    env_id: str = "reasoning_gym"

    def __init__(
        self,
        tasks: list[dict] | None = None,
        size: int = 1_000_000,
        seed: int = 0,
        num_shots: int = 3,
        format_reward: float = 0.0,
        binary_reward: bool = True,
        prompts_file: str | None = None,
        sequential: bool = False,
        probe_output: str | None = None,
        stop: list[str] | None = ("\n\n", "\n\n\n", "\nQuestion:"),
        **kwargs,
    ):
        """
        Args:
            tasks: [{name, weight (default 1), config (reasoning_gym dataset kwargs, i.e. the
                difficulty knobs)}]. Default: basic_arithmetic with its default config.
            size: Items per task in the training stream (and in the disjoint validation stream).
            seed: Base seed of the item streams.
            num_shots: Few-shot examples per prompt, from the task of the question.
            format_reward: Reward for a wrong but terminated, non-empty answer.
            binary_reward: Reward 1 only for a full-score answer (the default verifier of some
                tasks gives partial credit).
            prompts_file: JSONL with an "entry" per line (filter_pass_rate.py output); replaces
                the task streams for training. Few-shot examples still come from `tasks`.
            sequential: Visit items in order (task round-robin) instead of sampling.
            probe_output: Append {problem_id, rewards, responses (first 200 characters), entry} per
                group to this JSONL file.
            stop: Stop sequences for generation (None: generate until EOD or the length limit).
                Matching is on token ids, so "\n\n\n" (its own token) and a next "\nQuestion:"
                block are listed next to the blank line.
            Paths may contain environment variables ($VAR).
        """
        super().__init__(**kwargs)
        self.tasks = tasks or [{"name": "basic_arithmetic"}]
        self.size = size
        self.seed = seed
        self.num_shots = num_shots
        self.format_reward = format_reward
        self.binary_reward = binary_reward
        self.sequential = sequential
        self.probe_output = os.path.expandvars(probe_output) if probe_output else None
        self.stop = list(stop) if stop else None
        self._datasets = {}
        for split, offset in (("train", 0), ("validation", size), ("shots", 2 * size)):
            self._datasets[split] = [
                reasoning_gym.create_dataset(
                    task["name"], seed=seed + offset, size=size, **task.get("config", {})
                )
                for task in self.tasks
            ]
        self._weights = [float(task.get("weight", 1.0)) for task in self.tasks]
        self._entries = None
        if prompts_file is not None:
            with open(os.path.expandvars(prompts_file)) as f:
                self._entries = [json.loads(line)["entry"] for line in f if line.strip()]
        self._stream = repr((self.tasks, seed, size, prompts_file))
        self._rng = _STREAM_RNG.setdefault(self._stream, random.Random(self._stream))
        self._probe_responses = {}

    def _item(self, split: str, task_idx: int, idx: int) -> dict:
        entry = self._datasets[split][task_idx][idx]
        entry["problem_id"] = f"{self.tasks[task_idx]['name']}:{split}:{idx}"
        entry["task_idx"] = task_idx
        return entry

    def _sample(self, validation: bool) -> dict:
        if self._entries is not None and not validation:
            if self.sequential:
                i = self._advance() % len(self._entries)
            else:
                i = self._rng.randrange(len(self._entries))
            return dict(self._entries[i])
        split = "validation" if validation else "train"
        if self.sequential:
            k = self._advance()
            task_idx, idx = k % len(self.tasks), k // len(self.tasks)
        else:
            task_idx = self._rng.choices(range(len(self.tasks)), weights=self._weights)[0]
            idx = self._rng.randrange(self.size)
        return self._item(split, task_idx, idx)

    def _advance(self) -> int:
        k = _SEQUENTIAL_POSITION.get(self._stream, 0)
        _SEQUENTIAL_POSITION[self._stream] = k + 1
        return k

    def make_prefix(self, entry: dict) -> str:
        names = [task["name"] for task in self.tasks]
        source = entry["metadata"]["source_dataset"]
        task_idx = entry.get("task_idx", names.index(source) if source in names else 0)
        rng = random.Random(entry["problem_id"])
        shots = [self._datasets["shots"][task_idx][rng.randrange(self.size)] for _ in range(self.num_shots)]
        blocks = [f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots]
        return "".join(blocks) + f"Question: {entry['question']}\nAnswer:"

    @staticmethod
    def extract_answer(response: str) -> tuple[str, bool]:
        """The answer text before the next blank line, and whether the answer was terminated."""
        head, sep, _ = response.partition("\n\n")
        head = head.split("\nQuestion:")[0]
        return head.strip(), bool(sep)

    def score(self, response: str, entry: dict) -> float:
        answer, terminated = self.extract_answer(response)
        task_idx = entry.get("task_idx")
        if task_idx is not None:
            score_fn = self._datasets["train"][task_idx].score_answer
        else:
            score_fn = reasoning_gym.get_score_answer_fn(entry["metadata"]["source_dataset"])
        try:
            score = float(score_fn(answer, entry)) if answer else 0.0
        except Exception:
            score = 0.0
        if self.binary_reward:
            score = 1.0 if score >= 1.0 else 0.0
        if score == 0.0 and answer and terminated:
            return self.format_reward
        return score

    async def get_prompt(self, validation: bool = False) -> tuple[str, dict]:
        entry = self._sample(validation)
        return self.make_prefix(entry), entry

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> list[tuple[str, Any]]:
        split = "validation" if validation else "train"
        entries = [
            self._item(split, k % len(self.tasks), k // len(self.tasks)) for k in range(num_prompts)
        ]
        return [(self.make_prefix(entry), entry) for entry in entries]

    async def get_reward(self, response: str, golden: dict) -> float:
        if self.probe_output is not None:
            self._probe_responses.setdefault(golden["problem_id"], []).append(response[:200])
        return self.score(response, golden)

    async def rollout_from_response(self, request, response, golden):
        stopped = self._stopped_on_stop_sequence(response)
        if stopped:
            # The engine drops the stop sequence; keep the end of the answer visible to score().
            message = response.response.model_copy(
                update={"content": response.response.content + "\n\n"}
            )
            response = response.model_copy(update={"response": message})
        rollout = await super().rollout_from_response(request, response, golden)
        if stopped and isinstance(rollout, TokenRollout):
            from megatron.training import get_tokenizer

            rollout.trajectory[0].append(get_tokenizer().eod)
            rollout.generation_mask[0].append(False)
        return rollout

    def _stopped_on_stop_sequence(self, response) -> bool:
        if not self.stop or not response.token_ids:
            return False
        from megatron.training import get_args, get_tokenizer

        return (
            response.token_ids[-1] != get_tokenizer().eod
            and len(response.token_ids) < get_args().seq_length
        )

    async def group_rollout(self, request):
        if self.stop:
            request = request.model_copy(
                update={"generation_args": request.generation_args.add(
                    GenericGenerationArgs(stop=self.stop)
                )}
            )
        rollouts = await super().group_rollout(request)
        if self.probe_output is not None:
            # Rollouts carry the golden's problem_id; the probe needs the whole entry back.
            record = {
                "problem_id": rollouts[0].problem_id,
                "rewards": [rollout.reward for rollout in rollouts],
                "responses": self._probe_responses.pop(rollouts[0].problem_id, []),
                "entry": self._probe_entry(rollouts[0].problem_id),
            }
            with open(self.probe_output, "a") as f:
                f.write(json.dumps(record) + "\n")
        return rollouts

    def _probe_entry(self, problem_id: str) -> dict:
        name, split, idx = problem_id.rsplit(":", 2)
        task_idx = [task["name"] for task in self.tasks].index(name)
        entry = self._item(split, task_idx, int(idx))
        entry.pop("task_idx")
        return entry
