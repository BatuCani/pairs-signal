import sys

from pair_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["research", *sys.argv[1:]]))
