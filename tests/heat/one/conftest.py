# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from josie.bc import Dirichlet
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.heat.state import Q


@pytest.fixture
def boundaries():
    """ 1D problem along x """
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    left.bc = Dirichlet(Q(0))
    right.bc = Dirichlet(Q(0))
    top.bc = None
    bottom.bc = None

    yield (left, bottom, right, top)


@pytest.fixture()
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(100, 1)
    mesh.generate()

    yield mesh
