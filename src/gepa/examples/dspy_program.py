"""Minimal DSPy program demonstrating usage of the custom GEPA optimizer.

This example shows how to optimize the instruction of a simple DSPy
predictor using the GEPA implementation from this repository instead of
the version bundled with DSPy.

Run this script directly to execute the optimization. It requires a DSPy
language model to be configured via ``dspy.settings.configure``.
"""
from __future__ import annotations

import dspy
from dspy import Example

from gepa.api import optimize
from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter, ScoreWithFeedback


class AddModule(dspy.Module):
    """A tiny DSPy program that adds two integers."""

    def __init__(self) -> None:
        # The predictor is named ``add`` so GEPA can target it by name.
        self.add = dspy.Predict("a:int, b:int -> sum:int")

    def forward(self, a: int, b: int):
        return self.add(a=a, b=b)


def addition_metric(example: Example, prediction: dspy.Prediction) -> float:
    """Score 1.0 when the predicted sum matches the reference."""
    return float(example.sum == prediction.sum)


def addition_feedback(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
    """Provide textual feedback for GEPA based on model correctness."""
    correct = predictor_output["sum"] == module_inputs["sum"]
    if correct:
        feedback = "The prediction was correct."
    else:
        feedback = (
            f"Expected {module_inputs['sum']} but the model predicted {predictor_output['sum']}."
        )
    return ScoreWithFeedback(score=float(correct), feedback=feedback)


def run() -> None:
    """Run GEPA optimization on the ``AddModule`` DSPy program."""
    # Configure DSPy with an LM of your choice.
    dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))

    program = AddModule()

    # Tiny synthetic dataset for demonstration.
    trainset = [
        Example(a=1, b=2, sum=3),
        Example(a=5, b=7, sum=12),
        Example(a=2, b=9, sum=11),
    ]
    valset = list(trainset)

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=addition_metric,
        feedback_map={"add": addition_feedback},
    )

    seed_candidate = {
        "add": "Compute the integer sum of a and b and return it as `sum`.",
    }

    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm="gpt-4o-mini",
        max_metric_calls=20,
    )
    print("Best candidate:", result.best_candidate)
    print("Best score:", result.best_score)


if __name__ == "__main__":
    run()
