"""PolicyShift command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from policyshift import __version__
from policyshift.agents import BaselineAgent, OracleAgent, RAGAgent
from policyshift.data_generation import (
    check_split_leakage,
    generate_cases,
    write_cases,
    write_policies,
)
from policyshift.environment import PolicyShiftEnvironment, PolicyStore
from policyshift.evaluation import run_phase2_smoke
from policyshift.retrieval import PolicyRetriever, evaluate_retrieval_result
from policyshift.schemas import Split, export_json_schemas
from policyshift.training import run_phase3_smoke

app = typer.Typer(help="PolicyShift: continual post-training under evolving enterprise policies.")
console = Console()


@app.callback()
def main() -> None:
    """PolicyShift CLI."""


@app.command("version")
def version_cmd() -> None:
    console.print(f"policyshift {__version__}")


@app.command("generate-policies")
def generate_policies_cmd(
    out: Path = typer.Option(Path("policies"), help="Directory for policy JSON files"),
    export_json: Path = typer.Option(
        Path("data/generated/policies"), help="Aggregate export directory"
    ),
) -> None:
    paths = write_policies(out, export_json_dir=export_json)
    export_json_schemas(Path("policies/schemas"))
    console.print(f"Wrote {len(paths)} policy documents to {out}")


@app.command("generate-cases")
def generate_cases_cmd(
    seed: int = typer.Option(42, help="RNG seed"),
    n_cases: int = typer.Option(120, help="Number of cases to generate"),
    out: Path = typer.Option(Path("data/generated/cases"), help="Output directory"),
) -> None:
    cases = generate_cases(seed=seed, n_cases=n_cases)
    paths = write_cases(cases, out)
    report = check_split_leakage(cases)
    console.print(f"Wrote {len(cases)} cases to {out}")
    console.print(f"Leakage check ok={report['ok']} counts={report['template_counts']}")
    console.print(f"Artifacts: {', '.join(str(p) for p in paths.values())}")


@app.command("demo")
def demo_cmd(
    seed: int = typer.Option(42, help="Case generation seed"),
    case_index: int = typer.Option(0, help="Index into generated cases"),
    n_cases: int = typer.Option(120, min=1),
) -> None:
    """Run the oracle agent on one synthetic case and print the trajectory."""
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=seed, n_cases=n_cases)
    case = cases[case_index % len(cases)]
    agent = OracleAgent(store)
    trajectory = agent.resolve(case)

    table = Table(title=f"Oracle demo: {case.case_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Domain", case.domain.value)
    table.add_row("Difficulty", case.difficulty.value)
    table.add_row("Occurred at", case.occurred_at.isoformat())
    table.add_row("Expected policy", case.expected_policy_key)
    table.add_row("Expected resolution", case.expected_resolution)
    table.add_row("Success", str(trajectory.success))
    table.add_row("Total reward", f"{trajectory.total_reward:.3f}")
    table.add_row("Steps", str(len(trajectory.actions)))
    console.print(table)

    for action in trajectory.actions:
        ok = action.tool_output.get("ok", True) if action.tool_output else False
        console.print(
            f"[cyan]step {action.step_number}[/cyan] {action.tool_name} ok={ok} — {action.thought_summary}"
        )


@app.command("resolve-all")
def resolve_all_cmd(
    seed: int = typer.Option(42),
    n_cases: int = typer.Option(120),
    fail_fast: bool = typer.Option(False, help="Stop on first oracle failure"),
) -> None:
    """Resolve all generated cases with the oracle; report success rate."""
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=seed, n_cases=n_cases)
    agent = OracleAgent(store)
    successes = 0
    failures: list[str] = []
    for case in cases:
        traj = agent.resolve(case)
        if traj.success:
            successes += 1
        else:
            failures.append(
                f"{case.case_id} expected={case.expected_resolution} "
                f"final={traj.final_answer} cats={[c.value for c in traj.failure_categories]}"
            )
            if fail_fast:
                break
    console.print(f"Oracle success: {successes}/{len(cases)}")
    if failures:
        console.print(f"Failures ({len(failures)}):")
        for line in failures[:20]:
            console.print(f"  - {line}")
        raise typer.Exit(code=1)


@app.command("smoke-tools")
def smoke_tools_cmd(case_index: int = 0) -> None:
    """Exercise core tools on a single case."""
    store = PolicyStore.from_builtin()
    case = generate_cases(seed=42, n_cases=20)[case_index]
    env = PolicyShiftEnvironment(store, [case])
    calls = [
        ("inspect_case", {"case_id": case.case_id}),
        (
            "list_available_policies",
            {"domain": case.domain.value, "occurred_at": case.occurred_at.isoformat()},
        ),
        (
            "check_policy_effective_date",
            {
                "policy_id": case.expected_policy_id,
                "version": case.expected_policy_version,
                "occurred_at": case.occurred_at.isoformat(),
            },
        ),
        ("unknown_tool", {}),
    ]
    for name, args in calls:
        result = env.call_tool(name, args)
        console.print(f"{name}: ok={result.get('ok')} keys={list(result.keys())[:6]}")


@app.command("evaluate-phase2")
def evaluate_phase2_cmd(
    seed: int = typer.Option(42),
    n_cases: int = typer.Option(40, help="Cases to evaluate (from split)"),
    split: str = typer.Option("validation"),
    artifact_root: Path = typer.Option(Path("artifacts/experiments")),
) -> None:
    """Run Phase 2 smoke: retrieval ablations + baseline/RAG agents + artifact export."""
    result = run_phase2_smoke(
        seed=seed,
        n_cases=n_cases,
        split=Split(split),
        artifact_root=artifact_root,
    )
    console.print(f"Experiment: {result['experiment_id']}")
    console.print(f"Trajectories: {result['n_trajectories']}")
    for name, summary in result["summary"]["conditions"].items():
        console.print(
            f"  {name}: success={summary.get('success', 0):.3f} "
            f"stale_error={summary.get('stale_policy_error', 0):.3f} "
            f"n={summary.get('n', 0)}"
        )
    console.print("Retrieval ablation (recall_policy@5):")
    for mode, summary in result["summary"]["retrieval_ablation"].items():
        console.print(f"  {mode}: {summary.get('recall_policy@5', 0):.3f}")
    console.print(f"Artifacts: {result['paths'].get('manifest')}")


@app.command("compare-agents")
def compare_agents_cmd(
    seed: int = typer.Option(42),
    case_index: int = typer.Option(0),
    n_cases: int = typer.Option(120),
) -> None:
    """Compare baseline vs RAG vs oracle on one case."""
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=seed, n_cases=n_cases)
    case = cases[case_index % len(cases)]
    retriever = PolicyRetriever.from_store(store)
    agents = {
        "baseline": BaselineAgent(store),
        "rag": RAGAgent(store, retriever=retriever),
        "oracle": OracleAgent(store),
    }
    table = Table(title=f"Agent comparison: {case.case_id}")
    table.add_column("Agent")
    table.add_column("Success")
    table.add_column("Final")
    table.add_column("Expected")
    table.add_column("Reward")
    table.add_column("Steps")
    for name, agent in agents.items():
        traj = agent.resolve(case)
        table.add_row(
            name,
            str(traj.success),
            str(traj.final_answer),
            case.expected_resolution,
            f"{traj.total_reward:.2f}",
            str(len(traj.actions)),
        )
    console.print(table)
    result = retriever.retrieve(case, mode="date_filtered_rerank", top_k=3)
    metrics = evaluate_retrieval_result(case, result)
    console.print(
        f"Retrieval top={metrics.get('top_policy')} expected={metrics.get('expected_policy')} "
        f"recall@5={metrics.get('recall_policy@5')}"
    )


@app.command("evaluate-phase3")
def evaluate_phase3_cmd(
    seed: int = typer.Option(42),
    n_cases: int = typer.Option(40, help="Train cases for teacher/SFT"),
    n_eval: int = typer.Option(12, help="Eval cases for base vs distilled"),
    artifact_root: Path = typer.Option(Path("artifacts/experiments")),
) -> None:
    """Run Phase 3 smoke: teacher → filter → SFT → base/distilled comparison."""
    result = run_phase3_smoke(
        seed=seed,
        n_cases=n_cases,
        n_eval=n_eval,
        artifact_root=artifact_root,
    )
    console.print(f"Experiment: {result['experiment_id']}")
    console.print(
        f"Teacher accepted={result['summary']['teacher']['n_accepted']}/"
        f"{result['summary']['teacher']['n_cases']} "
        f"SFT examples={result['summary']['sft_examples']}"
    )
    console.print(f"Checkpoint: {result['summary']['checkpoint_loaded']['path']}")
    for name, summary in result["summary"]["conditions"].items():
        console.print(
            f"  {name}: success={summary.get('success', 0):.3f} "
            f"task_success={summary.get('task_success', 0):.3f} n={summary.get('n', 0)}"
        )


if __name__ == "__main__":
    app()
