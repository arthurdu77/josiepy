# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.solver import Solver
from josie.mesh import Mesh

from .schemes import TsCapScheme
from .state import Q


class TsCapSolver(Solver):
    """A solver for the TsCap system"""

    def __init__(self, mesh: Mesh, scheme: TsCapScheme):
        super().__init__(mesh, Q, scheme)
