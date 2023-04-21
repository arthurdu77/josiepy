from josie.euler.schemes.hllx import HLLC
from josie.general.schemes.space.muscl import MUSCL
from josie.general.schemes.space.limiters import MinMod
from josie.general.schemes.time.rk import RK2

from josie.bc import Neumann
from josie.boundary import Line
from josie.math import Direction
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.euler.eos import PerfectGas
from josie.euler.solver import EulerSolver
from josie.euler.state import EulerState
from josie.io.write.writer import XDMFWriter
from josie.io.write.strategy import TimeStrategy

from josie.bc import BoundaryCondition
from josie.euler.eos import EOS
from josie.euler.fields import EulerFields as fields


from josie.boundary import Boundary

import numpy as np


def init2Q(p: float, rho: float, U: float, V: float, eos):
    rhoe = eos.rhoe(rho, p)
    rhoE = 0.5 * rho * (np.power(U, 2) + np.power(V, 2)) + rhoe
    c = eos.sound_velocity(rho, p)

    return EulerState(rho, rho * U, rho * V, rhoE, rhoe, U, V, p, c, rhoe / rho)


eos = PerfectGas()

Q_TL = init2Q(0.3, 0.5323, 1.206, 0, eos)
Q_TR = init2Q(1.5, 1.5, 0, 0, eos)
Q_BL = init2Q(0.029, 0.138, 1.206, 1.206, eos)
Q_BR = init2Q(0.3, 0.5323, 0, 1.206, eos)


def init_fun(cells: MeshCellSet):
    x = cells.centroids[..., Direction.X]
    y = cells.centroids[..., Direction.Y]
    cells.values[np.where((x > 0.5) * (y > 0.5))] = Q_TR
    cells.values[np.where((x <= 0.5) * (y > 0.5))] = Q_TL
    cells.values[np.where((x > 0.5) * (y <= 0.5))] = Q_BR
    cells.values[np.where((x <= 0.5) * (y <= 0.5))] = Q_BL


class EulerNeumannBC(BoundaryCondition):
    # TODO: Add 3D
    def __init__(
        self,
        eos: EOS,
    ):
        # The partial set of BCs to impose
        self.p_bc = Neumann(0)
        self.rho_bc = Neumann(0)
        self.U_bc = Neumann(0)
        self.V_bc = Neumann(0)

        self.eos = eos

    def init(self, cells: MeshCellSet, boundary: Boundary):
        boundary_idx = boundary.cells_idx
        boundary_cells = cells[boundary_idx]

        self.p_bc.init(boundary_cells)
        self.rho_bc.init(boundary_cells)
        self.U_bc.init(boundary_cells)
        self.V_bc.init(boundary_cells)

    def __call__(self, cells: MeshCellSet, boundary: Boundary, t: float):
        ghost_idx = boundary.ghost_cells_idx
        boundary_idx = boundary.cells_idx

        boundary_cells = cells[boundary_idx]
        ghost_cells = cells[ghost_idx]

        # Compute all the derived ghost values
        p_ghost = self.p_bc(boundary_cells, ghost_cells, fields.p, t)
        rho_ghost = self.rho_bc(boundary_cells, ghost_cells, fields.rho, t)
        U_ghost = self.U_bc(boundary_cells, ghost_cells, fields.U, t)
        V_ghost = self.V_bc(boundary_cells, ghost_cells, fields.V, t)
        rhoe_ghost = self.eos.rhoe(rho_ghost, p_ghost)
        rhoU_ghost = rho_ghost * U_ghost
        rhoV_ghost = rho_ghost * V_ghost
        e_ghost = rhoe_ghost / rho_ghost
        rhoE_ghost = rho_ghost * (e_ghost + (U_ghost**2 + V_ghost**2) / 2)
        c_ghost = self.eos.sound_velocity(rho_ghost, p_ghost)

        # Impose the ghost values
        for field, ghost_value in (
            (fields.rho, rho_ghost),
            (fields.rhoU, rhoU_ghost),
            (fields.rhoV, rhoV_ghost),
            (fields.rhoE, rhoE_ghost),
            (fields.rhoe, rhoe_ghost),
            (fields.U, U_ghost),
            (fields.V, V_ghost),
            (fields.p, p_ghost),
            (fields.c, c_ghost),
            (fields.e, e_ghost),
        ):
            cells._values[ghost_idx[0], ghost_idx[1], :, field] = ghost_value


def test_shock():
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    left.bc = EulerNeumannBC(eos)
    right.bc = EulerNeumannBC(eos)
    top.bc = EulerNeumannBC(eos)
    bottom.bc = EulerNeumannBC(eos)

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(100, 100)
    mesh.generate()

    class Scheme(MUSCL, MinMod, RK2, HLLC):
        pass

    scheme = Scheme(eos)
    solver = EulerSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = 3e-1
    CFL = 0.5

    write_strategy = TimeStrategy(dt_save=1e-2, animate=True)
    writer = XDMFWriter("shock.xdmf", write_strategy, solver, final_time, CFL)
    writer.solve()
