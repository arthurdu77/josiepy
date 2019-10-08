# josiepy
# Copyright © 2019 Ruben Di Battista
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Ruben Di Battista ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Ruben Di Battista BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of Ruben Di Battista.
from __future__ import annotations

import abc
import numpy as np
import os

from meshio import XdmfTimeSeriesWriter
from typing import Callable, List, NoReturn, Union, TYPE_CHECKING

from .state import StateTemplate
from .scheme import Scheme

if TYPE_CHECKING:
    from josie.mesh import Mesh


class Solver(metaclass=abc.ABCMeta):
    def __init__(self, mesh: Mesh, Q: StateTemplate):
        """ This class is used to solve a problem governed by PDEs.

        The internal state of the mesh is stored in :attr:`values`, while the
        values of the ghost cells (used to apply the
        :class:`BoundaryCondition`) are stored respectively in
        :attr:`left_ghost`, :attr:`btm_ghost`, :attr:`right_ghost`,
        :attr:`top_ghost`. They're all numpy arrays.

        Parameters
        ----------
        mesh
            An instance of the mesh to compute the solution on
        Q
            A :class:`StateTemplate` representing the variables of the problem
            to be solved

        Attributes
        ----------
        values
            An array of dimensions (num_cells_x, num_cells_y, state_size)
            storing the value of the :class:`State` for each cell of the
            :class:`Mesh`
        left_ghost
            An array of dimensions (num_cells_y, state_size) storing the values
            of the :class:`State` on the ghost cells that are used to apply
            the :class:`BoundaryCondition` for the left boundary
        btm_ghost
            An array of dimensions (num_cells_y, state_size) storing the values
            of the :class:`State` on the ghost cells that are used to apply
            the :class:`BoundaryCondition` for the bottom boundary
        right_ghost
            An array of dimensions (num_cells_y, state_size) storing the values
            of the :class:`State` on the ghost cells that are used to apply
            the :class:`BoundaryCondition` for the right boundary
        top_ghost
            An array of dimensions (num_cells_y, state_size) storing the values
            of the :class:`State` on the ghost cells that are used to apply
            the :class:`BoundaryCondition` for the top boundary
        """

        self.mesh = mesh
        self.Q = Q

        # Initializing empty arrays for type checking
        self.values: np.ndarray = np.empty(0)
        self.left_ghost: np.ndarray = np.empty(0)
        self.right_ghost: np.ndarray = np.empty(0)
        if not (self.mesh.oneD):
            self.top_ghost: np.ndarray = np.empty(0)
            self.btm_ghost: np.ndarray = np.empty(0)

    def init(self, init_fun: Callable[[Solver], NoReturn]):
        """
        This method initialize the internal values of the cells of the
        :class:`Mesh` and the values of the ghost cells that apply the
        :class:`BoundaryCondition` for each boundary of the domain

        Parameters
        ---------
        init_fun
            The function to use to initialize the value in the domain

        """
        num_cells_x = self.mesh.num_cells_x
        num_cells_y = self.mesh.num_cells_y
        state_size = len(self.Q.fields)

        # First set all the values for the internal cells
        # The actual values are a view of only the internal cells
        self.values = np.empty((num_cells_x, num_cells_y, state_size))
        init_fun(self)

        # In order to apply BC, we create an array per each side of the domain
        # storing the State values of the ghost cells. This allows to assign
        # to them a view of the internal values array (self.values) by the
        # Periodic BoundaryCondition in such a way that, when internal values
        # are updated by the simulation, also the values of the ghost cells
        # are in sync since they are just memory "references"
        self.left_ghost = np.empty((num_cells_y, state_size))
        self.right_ghost = np.empty_like(self.left_ghost)

        # Left BC: Create the left layer of ghost cells
        self.left_ghost = self.mesh.left.bc(
            self,
            self.mesh.centroids[0, :],  # type: ignore
            self.values[0, :],
        )  # type: ignore

        # Right BC
        self.right_ghost = self.mesh.right.bc(
            self,
            self.mesh.centroids[-1, :],  # type: ignore
            self.values[-1, :],
        )  # type: ignore

        if not (self.mesh.oneD):
            self.btm_ghost = np.empty((num_cells_x, state_size))
            self.top_ghost = np.empty_like(self.btm_ghost)

            # Bottom BC
            self.btm_ghost = self.mesh.bottom.bc(
                self,
                self.mesh.centroids[:, 0],  # type: ignore
                self.values[:, 0],
            )  # type: ignore

            # Top BC
            self.top_ghost = self.mesh.top.bc(
                self,
                self.mesh.centroids[:, -1],  # type: ignore
                self.values[:, -1],
            )  # type: ignore

    def step(self, dt: float, scheme: Scheme):
        """ This method advances one step in time (for the moment using an
        explicit Euler scheme for time integration, but in future we will
        provide a way to give arbitrary time schemes)

        A `scheme` callable gets as input the internal values of the cells, the
        neighbour values, the normals associated to the neighbours and the
        value of the face surfaces. A `scheme` generally has to be coded as a
        1D scheme that operates only on the *right* neighbour. It is then
        called (in 2D) 4 times, one for each set of neighbours (left, bottom,
        right, top).  As an example, when called for the right neighbours, the
        data structures sent to the `scheme` callable for `values` and
        `neigh_values` are:

        values:>
        +---------------+---------------+---------------+
        | `values[0,2]` | `values[1,2]` | `values[2,2]` |
        +---------------+---------------+---------------+
        | `values[0,1]` | `values[1,1]` | `values[2,1]` |
        +---------------+---------------+---------------+
        | `values[0,0]` | `values[1,0]` | `values[2,0]` |
        +---------------+---------------+---------------+

        neighbours:>
        +---------------+---------------+------------------+
        | `values[1,2]` | `values[2,2]` | `right_ghost[2]` |
        +---------------+---------------+------------------+
        | `values[1,1]` | `values[2,1]` | `right_ghost[1]` |
        +---------------+---------------+------------------+
        | `values[1,0]` | `values[2,0]` | `right_ghost[0]` |
        +---------------+---------------+------------------+

        Parameters
        ----------
        dt
            Time increment of the step
        scheme
            A callable representime the space scheme to use

        """
        # We accumulate the delta fluxes for each set of neighbours
        fluxes = np.zeros_like(self.values)

        # Left Neighbours
        neighs = np.concatenate(
            (self.left_ghost[np.newaxis, :, :], self.values[:-1, :])
        )
        fluxes += scheme.convective_flux(
            self.values,
            neighs,
            self.mesh.normals[:, :, 0, :],
            self.mesh.surfaces[:, :, 0],
        )

        # Right Neighbours
        neighs = np.concatenate(
            (self.values[1:, :], self.right_ghost[np.newaxis, :, :])
        )

        fluxes += scheme.convective_flux(
            self.values,
            neighs,
            self.mesh.normals[:, :, 2, :],
            self.mesh.surfaces[:, :, 2],
        )

        if not (self.mesh.oneD):

            # Top Neighbours
            neighs = np.concatenate(
                (self.top_ghost[:, np.newaxis, :], self.values[:, :-1]), axis=1
            )
            fluxes += scheme.convective_flux(
                self.values,
                neighs,
                self.mesh.normals[:, :, 3, :],
                self.mesh.surfaces[:, :, 3],
            )

            # Bottom Neighbours
            neighs = np.concatenate(
                (self.values[:, 1:], self.btm_ghost[:, np.newaxis, :]), axis=1
            )
            fluxes += scheme.convective_flux(
                self.values,
                neighs,
                self.mesh.normals[:, :, 1, :],
                self.mesh.surfaces[:, :, 1],
            )

        self.values -= fluxes * dt / self.mesh.volumes[:, :, np.newaxis]

        # Let's put here an handy post step if needed after the values update
        self.post_step()

    def post_step(self):
        """ This method can be used to post-process the data after that the
        flux update took place.

        For example it is used in the :class:`EulerSolver` in order to retrieve
        the auxiliary variables, from the conservative ones, using the
        :class:`EOS`
        """
        pass

    def save(self, t, filename: os.PathLike):
        """ This method saves the simulation instant in a `xdmf` file
        supporting time series.

        Parameters
        ---------
        t
            The time instant to save
        filename
            The name of the file holding the time series. It's an XDMF file
        """

        if not (hasattr(self, "_writer")):
            io_mesh = self.mesh.export()
            self._writer = XdmfTimeSeriesWriter(filename)
            self._writer.write_points_cells(io_mesh.points, io_mesh.cells)

        cell_data = {}
        for i, field in enumerate(self.Q.fields):
            cell_data[field] = self.values[:, :, i].ravel()

        cell_type_str = self.mesh.cell_type._meshio_cell_type
        self._writer.write_data(t, cell_data={cell_type_str: cell_data})

    def plot(self):
        """ Plot the current state of the simulation in a GUI.

        You can specify which fields to plot

        Parameters
        ---------
        fields
            A list (or a string, if a singular field is needed) of fields
            to plot
        """
        plt = self.mesh.backend
        plt.update(self)

    def animate(self, t):
        """ Animate the simulation. Call :meth:`animate` for each time instant
        you want to provide in the animation.

        Parameters
        ----------
        t
            The time instant to animate
        """
        plt = self.mesh.backend
        plt.append(self, t)

    def show(self, fields: Union[List[str], str]):
        """ Display on screen the given fields
        """

        plt = self.mesh.backend
        plt.show(fields)
