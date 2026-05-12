#!/usr/bin/env python
from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from diamondworld.eval.harness import main


if __name__ == "__main__":
    main()
