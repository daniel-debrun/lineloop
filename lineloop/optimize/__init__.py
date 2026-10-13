"""Optimisation: the inspection policy (MILP + twin loop) and the solver layer it runs on.

MIP solvers live in `lineloop.optimize.mip` (cuOpt on GPU, HiGHS on CPU, the same model for both).
"""
