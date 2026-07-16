import argparse
import json
from pathlib import Path
import sys

from ..paths import DEFAULT_SCENARIO_PATH
from .agents import load_agent_manifest, validate_agent_manifest
from .environment import CompetitionEnv
from .runner import CompetitionRunner
from .schema import generate_schemas


def _build_parser():
    parser = argparse.ArgumentParser(description="AFSIM air-combat competition baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-agent", help="Validate an explicit agent manifest")
    validate_parser.add_argument("manifest")

    schema_parser = subparsers.add_parser("schema", help="Generate versioned JSON schemas")
    schema_parser.add_argument("--output", default="docs/schemas")

    check_parser = subparsers.add_parser("check-afsim", help="Check the AFSIM gRPC server")
    check_parser.add_argument("--ip", default="127.0.0.1")
    check_parser.add_argument("--port", default="19920")
    check_parser.add_argument("--timeout", type=float, default=5.0)

    run_parser = subparsers.add_parser("run", help="Run a competition episode")
    run_parser.add_argument("--blue-agent", required=True)
    run_parser.add_argument("--red-agent", required=True)
    run_parser.add_argument("--ip", default="127.0.0.1")
    run_parser.add_argument("--port", default="19920")
    run_parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO_PATH))
    run_parser.add_argument("--episodes", type=int, default=1)
    run_parser.add_argument("--steps", type=int, default=2000)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--step-delay", type=float, default=0.0)
    run_parser.add_argument("--output", default="runs")
    run_parser.add_argument("--global-view", action="store_true")
    return parser


def main(argv=None, env_factory=None, runner_factory=CompetitionRunner, client_factory=None):
    args = _build_parser().parse_args(argv)
    if args.command == "validate-agent":
        print(json.dumps(validate_agent_manifest(args.manifest), ensure_ascii=False, indent=2))
        return 0
    if args.command == "schema":
        paths = generate_schemas(args.output)
        print(json.dumps({"schemas": [str(path) for path in paths]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "check-afsim":
        if client_factory is None:
            from ..client import EnvClient

            client_factory = EnvClient
        from ..protocol import afsim_pb2

        client = client_factory(
            afsim_ip=args.ip,
            afsim_port=args.port,
            rpc_timeout=args.timeout,
        )
        try:
            client.connect_server()
            state = client.get_server_state()
        except Exception as error:
            print(
                json.dumps(
                    {"ready": False, "error": str(error)},
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1
        finally:
            client.close()

        ready = state == afsim_pb2.active
        print(
            json.dumps(
                {
                    "ready": ready,
                    "state": afsim_pb2.SimServerStateEnum.Name(state),
                    "state_code": state,
                    "endpoint": f"{args.ip}:{args.port}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if ready else 1

    if env_factory is None:
        from ..env import AFsimEnv

        env_factory = AFsimEnv
    blue_agent = load_agent_manifest(args.blue_agent)
    red_agent = load_agent_manifest(args.red_agent)
    raw_env = env_factory(args.ip, args.port, args.scenario)
    competition_env = CompetitionEnv(
        raw_env,
        max_steps=args.steps,
        global_view=args.global_view,
    )
    summaries = []
    for episode_index in range(args.episodes):
        seed = None if args.seed is None else args.seed + episode_index
        runner = runner_factory(
            competition_env,
            blue_agent=blue_agent,
            red_agent=red_agent,
            output_dir=Path(args.output),
            max_steps=args.steps,
            step_delay_s=args.step_delay,
        )
        summaries.append(runner.run_episode(seed=seed))
    print(json.dumps({"episodes": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
