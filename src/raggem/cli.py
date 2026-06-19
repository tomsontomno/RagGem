"""``raggem`` command-line interface.

Subcommands:
  brains              list configured brains
  list      BRAIN     list source files in a brain
  upload    BRAIN F+  add one or more files to a brain and index them
  delete    BRAIN F   delete a single file from a brain
  query     BRAIN Q   ask a question
  rebuild   BRAIN     drop the vector store and reindex every file
  destroy   BRAIN     erase a brain (documents AND vector store)
  serve               start the HTTP API server
  mcp                 start the MCP stdio server (optional integration)

Every subcommand returns process exit code 0 on success, 1 on user error
(bad input, file not found, ...) and 2 on unexpected internal errors. ``--json``
switches output to one-line JSON for machine consumers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import APP_NAME, APP_VERSION


_EXIT_OK = 0
_EXIT_USER = 1
_EXIT_INTERNAL = 2


def _print(payload: Dict[str, Any], *, as_json: bool, human: str) -> None:
    """Emit either JSON or a human-readable line, controlled by --json."""
    if as_json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(human)


def _setup_logging(verbose: bool) -> None:
    """Configure root logging.

    Defaults to WARN so the CLI stays quiet by default; verbose lifts to
    INFO. Errors always go to stderr because Python's default config sends
    everything to stderr already.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def _cmd_brains(args: argparse.Namespace) -> int:
    from .core.brains import list_brains

    brains = list_brains()
    if args.json:
        print(json.dumps({"brains": brains}, ensure_ascii=False))
    else:
        if not brains:
            print("(no brains yet - upload a file to create one)")
        else:
            for b in brains:
                print(b)
    return _EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    from .core.brains import list_files

    files = list_files(args.brain)
    if args.json:
        print(
            json.dumps(
                {
                    "brain": args.brain,
                    "files": [
                        {
                            "filename": f.filename,
                            "size_bytes": f.size_bytes,
                            "extension": f.extension,
                        }
                        for f in files
                    ],
                },
                ensure_ascii=False,
            )
        )
    else:
        if not files:
            print(f"(brain {args.brain!r} has no documents)")
        else:
            for f in files:
                print(f"{f.filename}\t{f.size_bytes} bytes")
    return _EXIT_OK


def _cmd_upload(args: argparse.Namespace) -> int:
    from .core.brains import save_uploaded_file
    from .core.engine import index_one, invalidate_brain

    results: List[Dict[str, Any]] = []
    for raw_path in args.files:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            print(f"error: {raw_path!r} is not a file", file=sys.stderr)
            return _EXIT_USER
        with path.open("rb") as fh:
            stored = save_uploaded_file(args.brain, path.name, fh)
        chunks = index_one(args.brain, stored.name)
        results.append(
            {"filename": stored.name, "chunks_indexed": chunks}
        )
    invalidate_brain(args.brain)
    if args.json:
        print(json.dumps({"brain": args.brain, "uploaded": results}, ensure_ascii=False))
    else:
        for r in results:
            print(f"uploaded {r['filename']} -> {r['chunks_indexed']} chunks")
    return _EXIT_OK


def _cmd_delete(args: argparse.Namespace) -> int:
    from .core.brains import delete_file
    from .core.engine import invalidate_brain, remove_source

    removed_chunks = remove_source(args.brain, args.filename)
    removed_file = delete_file(args.brain, args.filename)
    invalidate_brain(args.brain)
    if not removed_file and removed_chunks == 0:
        print(f"error: {args.filename!r} not found in brain {args.brain!r}", file=sys.stderr)
        return _EXIT_USER
    if args.json:
        print(
            json.dumps(
                {
                    "brain": args.brain,
                    "filename": args.filename,
                    "file_removed": removed_file,
                    "chunks_removed": removed_chunks,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(
            f"deleted {args.filename} "
            f"(file: {'yes' if removed_file else 'no'}, chunks: {removed_chunks})"
        )
    return _EXIT_OK


def _cmd_query(args: argparse.Namespace) -> int:
    from .core.engine import query_brain

    result = query_brain(args.brain, args.question)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["answer"])
        if args.show_sources and result.get("sources"):
            print("\n--- sources ---")
            for s in result["sources"]:
                print(f"[{s['source_id']}] {s['source']} (page {s['page']}, distance {s['distance']:.3f})")
    return _EXIT_OK


def _cmd_rebuild(args: argparse.Namespace) -> int:
    from .core.brains import brain_paths, reset_vector_store
    from .core.engine import get_engine, invalidate_brain

    if not brain_paths(args.brain).knowledge_dir.exists():
        print(f"error: brain {args.brain!r} has no knowledge directory", file=sys.stderr)
        return _EXIT_USER
    reset_vector_store(args.brain)
    invalidate_brain(args.brain)
    counts = get_engine(args.brain).index_all()
    total = sum(counts.values())
    if args.json:
        print(json.dumps({"brain": args.brain, "reindexed": counts, "total_chunks": total}, ensure_ascii=False))
    else:
        for fn, n in counts.items():
            print(f"{fn}\t{n} chunks")
        print(f"total: {total} chunks across {len(counts)} files")
    return _EXIT_OK


def _cmd_destroy(args: argparse.Namespace) -> int:
    from .core.brains import brain_paths, destroy_brain
    from .core.engine import invalidate_brain

    paths = brain_paths(args.brain)
    existed = paths.exists()
    if existed and not args.yes:
        confirm = input(
            f"This will permanently delete brain {args.brain!r} "
            f"(documents and vector store). Type the brain id to confirm: "
        )
        if confirm.strip() != args.brain:
            print("aborted", file=sys.stderr)
            return _EXIT_USER
    destroy_brain(args.brain)
    invalidate_brain(args.brain)
    if args.json:
        print(json.dumps({"brain": args.brain, "destroyed": existed}, ensure_ascii=False))
    else:
        print(f"destroyed {args.brain}" if existed else f"brain {args.brain} did not exist")
    return _EXIT_OK


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .config import admin_key_required

    admin_key_required()  # fail fast if unconfigured
    uvicorn.run(
        "raggem.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return _EXIT_OK


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import run_stdio

    run_stdio()
    return _EXIT_OK


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON output instead of human-readable text",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="raggem",
        description=f"{APP_NAME} v{APP_VERSION} - hardened RAG service.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="enable info logs")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("brains", help="list all brains")
    _add_common(sp)
    sp.set_defaults(func=_cmd_brains)

    sp = sub.add_parser("list", help="list files in a brain")
    sp.add_argument("brain", help="brain id")
    _add_common(sp)
    sp.set_defaults(func=_cmd_list)

    sp = sub.add_parser("upload", help="upload one or more files into a brain")
    sp.add_argument("brain", help="brain id")
    sp.add_argument("files", nargs="+", help="files to upload")
    _add_common(sp)
    sp.set_defaults(func=_cmd_upload)

    sp = sub.add_parser("delete", help="delete a single file from a brain")
    sp.add_argument("brain", help="brain id")
    sp.add_argument("filename", help="filename (no path) inside the brain")
    _add_common(sp)
    sp.set_defaults(func=_cmd_delete)

    sp = sub.add_parser("query", help="ask a question of a brain")
    sp.add_argument("brain", help="brain id")
    sp.add_argument("question", help="the question, in quotes")
    sp.add_argument(
        "--show-sources",
        action="store_true",
        help="also print retrieved source snippets",
    )
    _add_common(sp)
    sp.set_defaults(func=_cmd_query)

    sp = sub.add_parser("rebuild", help="drop the vector store and reindex everything")
    sp.add_argument("brain", help="brain id")
    _add_common(sp)
    sp.set_defaults(func=_cmd_rebuild)

    sp = sub.add_parser("destroy", help="erase a brain entirely")
    sp.add_argument("brain", help="brain id")
    sp.add_argument(
        "-y", "--yes", action="store_true", help="skip the confirmation prompt"
    )
    _add_common(sp)
    sp.set_defaults(func=_cmd_destroy)

    sp = sub.add_parser("serve", help="start the HTTP API server")
    sp.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0)")
    sp.add_argument("--port", type=int, default=8100, help="bind port (default 8100)")
    sp.add_argument(
        "--reload",
        action="store_true",
        help="auto-reload on code changes (development only)",
    )
    sp.set_defaults(func=_cmd_serve)

    sp = sub.add_parser("mcp", help="start the MCP stdio server")
    sp.set_defaults(func=_cmd_mcp)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: optional argument list (for testing). Defaults to sys.argv[1:].

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return _EXIT_USER
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USER
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USER
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logging.getLogger("raggem.cli").exception("Unhandled error")
        print(f"internal error: {exc}", file=sys.stderr)
        return _EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
