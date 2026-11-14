from scripts.compute_tex import policy_sentence


def test_policy_sentence_reads_the_bench_cells():
    cell = lambda dev, th, n, med: {"device": dev, "threads": th, "cars": n, "median": med, "lo": med * 0.9,
                                    "hi": med * 1.2, "n": 300}
    p = {"W": 12, "reps": 300, "gpu_shared": False,
         "cells": [cell("cuda", None, 1, 4e-4), cell("cuda", None, 64, 5e-4), cell("cpu", 1, 1, 2e-4),
                   cell("cpu", 1, 64, 9e-4), cell("cpu", 24, 1, 3e-4), cell("cpu", 24, 64, 6e-4)]}
    s = policy_sentence(p)
    assert "0.200 [0.180, 0.240]\\,ms for one car" in s and "0.500 [0.450, 0.600]\\,ms for 64 cars" in s
    assert "N = 300" in s and "otherwise idle" in s
