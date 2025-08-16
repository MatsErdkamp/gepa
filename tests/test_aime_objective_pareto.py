import os
import sys
import types

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter


def test_aime_objective_pareto_example():
    # Stub out the datasets module to avoid network calls
    dummy_records = [
        {"problem": "P1", "solution": "S1", "answer": 1},
        {"problem": "P2", "solution": "S2", "answer": 2},
    ]

    datasets_stub = types.ModuleType("datasets")

    def fake_load_dataset(name):
        return {"train": [dict(r) for r in dummy_records]}

    datasets_stub.load_dataset = fake_load_dataset
    sys.modules["datasets"] = datasets_stub

    from gepa.examples import aime

    trainset, valset, _ = aime.init_dataset()

    seed_prompt = {
        "system_prompt": (
            "You are a helpful assistant. You are given a question and you need to answer it. "
            "The answer should be given at the end of your response in exactly the format '### <final answer>'"
        )
    }

    class DummyAIMEAdapter(GEPAAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            outputs = [
                {"full_assistant_response": candidate["system_prompt"]} for _ in batch
            ]
            trajectories = [] if capture_traces else None
            scores = []
            for data in batch:
                reasoning = 1.0 if "improved" in candidate["system_prompt"] else 0.0
                answer = 0.5
                scores.append({"reasoning": reasoning, "answer": answer})
                if capture_traces:
                    trajectories.append({"data": data})
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {components_to_update[0]: [{"input": "dummy", "feedback": "dummy"}]}

    adapter = DummyAIMEAdapter()

    result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=lambda prompt: "```improved```",
        candidate_selection_strategy="pareto",
        objectives=["reasoning", "answer"],
        selection_strategy="objective_pareto",
        reflection_minibatch_size=1,
        max_metric_calls=10,
    )

    # Ensure multi-objective scores are tracked and multiple candidates are generated
    assert isinstance(result.val_subscores[0][0], dict)
    assert set(result.val_subscores[0][0].keys()) == {"reasoning", "answer"}
    assert len(result.candidates) >= 2
