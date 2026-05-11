"""
Tulbase CLI — Command line interface.

Usage:
    tulbase serve                    Start the proxy server
    tulbase serve --port 9000        Custom port
    tulbase check "your prompt"      Check for injection
    tulbase optimize "your prompt"   Preview optimization
    tulbase --version                Show version
"""

import argparse
import sys

from . import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="tulbase",
        description="Tulbase — LLM API proxy with prompt optimization & injection protection",
    )
    parser.add_argument("--version", action="version", version=f"tulbase {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the proxy server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")

    # check
    check_parser = subparsers.add_parser("check", help="Check text for injection")
    check_parser.add_argument("text", help="Text to check")

    # optimize
    opt_parser = subparsers.add_parser("optimize", help="Preview optimization")
    opt_parser.add_argument("text", help="Text to optimize")

    args = parser.parse_args()

    if args.command == "serve":
        import os
        os.environ["TULBASE_HOST"] = args.host
        os.environ["TULBASE_PORT"] = str(args.port)
        if args.reload:
            os.environ["TULBASE_DEBUG"] = "true"
        from .main import serve
        serve()

    elif args.command == "check":
        from .injection import check_injection
        result = check_injection(args.text)
        if result.is_injection:
            print(f"INJECTION DETECTED (score: {result.score:.2f})")
            print(f"  Patterns: {', '.join(result.matched_patterns)}")
            print(f"  Layer: {result.layer}")
            sys.exit(1)
        else:
            print(f"Clean (score: {result.score:.2f})")

    elif args.command == "optimize":
        from .optimizer import optimize_messages
        messages = [{"role": "user", "content": args.text}]
        optimized, _ = optimize_messages(messages)
        original = args.text
        result = optimized[0]["content"] if optimized else ""
        saved = len(original) - len(result)
        pct = (saved / len(original) * 100) if len(original) > 0 else 0
        print(f"Original:  {original}")
        print(f"Optimized: {result}")
        print(f"Saved:     {saved} chars ({pct:.1f}%)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
