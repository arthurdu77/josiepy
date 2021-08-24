# Architecture of the code

![](img/josiepy-design.svg)

`josiepy` has been thought as a fast prototyping tool that would allow
to experiment on a single aspect of the resolution of a <span
acronym-label="pde" acronym-form="singular+short">pde</span> problem,
without the need of knowing how things are done for the other modules. A
representation of the core modules is shown in
<a href="#fig:josiepy-arch" data-reference-type="ref" data-reference="fig:josiepy-arch">1</a>.
We can see that the code is logically arranged in three macro-areas:

-   Mesh Generation

-   Physics of the problem to be solved

-   Numerics

The three aspects are then combined and used in the `solver` module the
aim of which is to orchestrate the simulation exposing the mesh geometry
to the chosen scheme, advance the simulation in time, and store all the
metadata about the simulation. The `io` module operates on the solver
object in order to serialize simulation results to the disk. All the
underlying data structures for storing mesh-related quantities or the
problem fields evolved in time during the simulation are based on the
NumPy array <span acronym-label="api"
acronym-form="singular+short">api</span>. This allows to define
operations on the entire block of values, exploiting modern processor
optimizations for large float vectors operations (SSEx, AVX2, AVX512)
that NumPy ships internally via the linking to optimized linear algebra
implementations, but also other type of low-level containers that are
not stored locally on a machine, as for the bare NumPy `ndarrays`, but
potentially on distributed machines and GPUs .

## Physics of the Problem

The model template which `josiepy` supports is,
$$\\pdeFullVector$$
in order to take care of most of the problems governed by <span
acronym-label="pde" acronym-form="plural+short">pdes</span> that concern
the underlying scientific context of this work with a logic separation
of the various type of contributions (convective, non-conservative,
diffusive and source terms) together with time integration implemented
with the <span acronym-label="mol"
acronym-form="singular+short">mol</span> semi-discretization technique.
This is indeed an opinionated choice, but it appeared flexible enough
for the purpose of this work. The dynamic nature of the Python language
allows faster refactoring *w.r.t.* C nonetheless.

## The `State` object

The first thing to do is to define the phase space containing the
problem variables. For example, to create a solver for the Euler system
(<a href="#ssub:the_euler_system" data-reference-type="ref" data-reference="ssub:the_euler_system">[ssub:the_euler_system]</a>),
the fields are defined as,

``` python
from josie.fluid.fields import FluidFields

class EulerFields(FluidFields):
    # Conservative fields
    rho = 0
    rhoU = 1
    rhoV = 2
    rhoE = 3

    # auxiliary fields
    U = 4
    V = 5
    rhoe = 6
    p = 7
    c = 8
```

Strictly speaking, the fields that are present in the equations are the
conservative ones, but to compute the other properties of the system,
notably using the <span acronym-label="eos"
acronym-form="singular+short">eos</span>, it might be necessary to
access other auxiliary fields, arbitrarily defined (increasing the
memory footprint of the running simulation and the corresponding size of
the saved file on disk). Once the fields of the problem are defined,
they can be assigned to a `State` object,

``` python
class EulerState(State):
    fields = EulerFields
```

which, under the hood, is a familiar NumPy array. In facts it is
possible to *cast* a NumPy array into a `State` object, and access it
with integer indices or using its fields,

``` python
    rnd_state = np.random.random(len(EulerFields)).view(EulerState)
    fields = rnd_state.fields
    assert rnd_state[0] == rnd_state[fields.rho]
```

The class `EulerFields` is actually an enumeration of integers, that is
fast to access and with small memory footprint, that can be used to
index the `State` array without the need of remembering the index of the
desired field. `State` can also be multidimensional (as it is within the
solvers namespace), and the `Ellipsis` \[`...`\] Python object allows to
retrieve all the values of a set of fields for all the cells of a mesh
irrespective of the dimensionality (1D, 2D or 3D) of the problem.

``` python
    rnd_state = np.random.random((100, 100, len(EulerFields))).view(EulerState)
    U = rnd_state[..., EulerFields.U]
```

There is also a functional <span acronym-label="api"
acronym-form="singular+short">api</span> to define a state class, that
can allow the definition of a state in a slightly less verbose fashion,

``` python
    MyStateClass = StateTemplate("rho", "rhoU", "rhoV")
    zero = np.zeros(10).view(MyStateClass)
```

## The `Problem` object

The `Problem` is the “continuous” representation of the problem the user
is willing to simulate. The `Problem` class implements the corresponding
methods to the terms
$\\pdeConvective, \\pdeDiffusiveMultiplier, \\pdeNonConservativeMultiplier, \\pdeSource$
of the reference
<a href="#eq:pde-convective-system-3" data-reference-type="ref" data-reference="eq:pde-convective-system-3">[eq:pde-convective-system-3]</a>.
As an example, the implementation of the `josie.euler.EulerProblem`
needs to provide the implementation of the convective flux for the Euler
system, that is

``` python
class EulerProblem(Problem):
    def __init__(self, eos: EOS):
        self.eos = eos

    def F(self, cells: CellSet) -> np.ndarray:
        values: Q = cells.values.view(Q)
        fields = values.fields

        num_cells_x, num_cells_y, _ = values.shape

        # Flux tensor
        F = np.empty(
            (num_cells_x, num_cells_y, len(ConsFields), MAX_DIMENSIONALITY)
        )

        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        rhoE = values[..., fields.rhoE]
        U = values[..., fields.U]
        V = values[..., fields.V]
        p = values[..., fields.p]

        rhoUU = np.multiply(rhoU, U)
        rhoUV = np.multiply(rhoU, V)
        rhoVV = np.multiply(rhoV, V)
        rhoVU = rhoUV  # np.multiply(rhoV, U)

        F[..., fields.rho, Direction.X] = rhoU
        F[..., fields.rho, Direction.Y] = rhoV
        F[..., fields.rhoU, Direction.X] = rhoUU + p
        F[..., fields.rhoU, Direction.Y] = rhoUV
        F[..., fields.rhoV, Direction.X] = rhoVU
        F[..., fields.rhoV, Direction.Y] = rhoVV + p
        F[..., fields.rhoE, Direction.X] = np.multiply(rhoE + p, U)
        F[..., fields.rhoE, Direction.Y] = np.multiply(rhoE + p, V)

        return F
```

The `Problem.F` method operates on all the cell meshes (`cells`) as an
entire entity to promote vectorized operations driven by NumPy. To
understand better the data structure for the mesh and the values
contained in each mesh cell, we will now discuss the mesh generation.

## Mesh Generation

In
<a href="#sub:num-mesh-generation" data-reference-type="ref" data-reference="sub:num-mesh-generation">[sub:num-mesh-generation]</a>
we discussed the mathematical background that backs our implementation
of the structured mesher integrated in `josiepy`. Here we will discuss
the choices we made for the <span acronym-label="api"
acronym-form="singular+short">api</span>. The mesh generation starts
defining the boundaries of the domain, directly in Python

``` python
from josie.boundary import Line, CircleArc

left = Line([0, 0], [0, 1])
bottom = CircleArc([0, 0], [1, 0], [0.5, 0.5])
right = Line([1, 0], [1, 1])
top = Line([0, 1], [1, 1])
```

to be sure that the defined domain coincides exactly with what the user
had in mind, without the need of saving the mesh in another format and
then opening in a post-processing tool, it is possible to simply plot
the curves that constitute the domain boundary,

``` python
for curve in [left, bottom, right, top]:
    curve.plot()
```

and the resulting image is shown in
<a href="#fig:josiepy-boundary-plot" data-reference-type="ref" data-reference="fig:josiepy-boundary-plot">2</a>.

<figure>
<img src="./shared/img/josiepy-boundary-plot.png" id="fig:josiepy-boundary-plot" alt="The plotted boundary of the domain" /><figcaption aria-hidden="true">The plotted boundary of the domain</figcaption>
</figure>

A problem governed by a system of <span acronym-label="pde"
acronym-form="plural+short">pdes</span> requires the imposition of <span
acronym-label="bc" acronym-form="plural+short">bcs</span> on all the
domain boundaries. With `josiepy` it is possible to directly assign the
<span acronym-label="bc" acronym-form="plural+short">bcs</span> to the
actual boundaries of the domain. The generic module `josie.bc` provides
the `Dirichlet` and `Neumann` base classes that the user can use
respectively to impose a fixed value for the state on a boundary or a
gradient value. As an example, let us impose a zero value on the left,
bottom, and right boundary and a zero-gradient <span acronym-label="bc"
acronym-form="singular+short">bc</span> on the top boundary:

``` python
from josie.bc import Dirichlet, Neumann

Q_zero = np.zeros(len(EulerState.fields)).view(EulerState)
dQ_zero = Q_zero

left.bc = Dirichlet(Q_zero)
top.bc = Neumann(dQ_zero)
bottom.bc = Dirichlet(Q_zero)
right.bc = Dirichlet(Q_zero)
```

This way of defining the <span acronym-label="bc"
acronym-form="plural+short">bcs</span> is easy and allows to assign to
the entire state the same <span acronym-label="bc"
acronym-form="singular+short">bc</span>. More often it is needed to set
different conditions on the individual fields of a state. This is indeed
also possible with `josiepy`,

``` python
from josie.bc import BoundaryCondition

Q_bc = EulerState(rho=Dirichlet(1), rhoU=Neumann(0), rhoV=Dirichlet(1), 
                  rhoE=Dirichlet(1), rhoe=Dirichlet(3), U=Neumann(0), 
                  V=Neumann(0), p=Dirichlet(1), c=Dirichlet(10))

left.bc = BoundaryCondition(Q_bc)
```

What is happening here is that we can assign per each field of the state
a `Dirichlet` or `Neumann` condition. In facts, the boundary conditions
objects act a bit magically based on the given input. If the given input
value is an entire `State` object, then the <span acronym-label="bc"
acronym-form="singular+short">bc</span> is set on all the fields of the
problem. If the input value is just a `float`, then the returned <span
acronym-label="bc" acronym-form="singular+short">bc</span> object is a
`ScalarBC`, and it just sets the condition on one individual (that is
“scalar”) field. As boundary values, not only constant values are
possible: in order to impose space and time dependent boundary values,
also `Callable` objects can be given as argument to the <span
acronym-label="bc" acronym-form="plural+short">bcs</span>. For
reference, the `josie.ns.bc` module implements some specific <span
acronym-label="bc" acronym-form="plural+short">bcs</span> for the
Navier-Stokes problem that make use of the `Callable` input.

Internally, the are applied assigning a value to the ghost cells of the
mesh. shows a sample mesh with the ghost shell next to each boundary
highlighted. The entire mesh including the ghost cells is allocated in a
contiguous memory region such that much of the NumPy slicing operations
are just memory views and not copies of the same data. The corner values
are unused in the current implementation and `NaNs` are stored in those
locations. The `Dirichlet` <span acronym-label="bc"
acronym-form="singular+short">bc</span> implementation currently imposes
the boundary value as the arithmetical mean of the neighboring points,
$$\\genericField_D = \\frac{\\genericField_i + \\genericField_G}{2}$$
where $\\genericField_D$ is the value to impose on the boundary face for
the generic field $\\genericField$. Therefore this is translated on a
ghost value to be imposed:
$$\\label{eq:dirichlet-bc}
    \\genericField_G = 2 \\genericField_D - \\genericField_i$$
Similarly, the `Neumann` condition is imposed as:
$$\\gradient{\\genericField}\_N \\cdot\\n = \\frac{\\genericField_G - \\genericField_i}{\\Delta\\vx\_{Gi} \\cdot\\n} 
    \\ensuremath{\\triangleq}g_N$$
hence the value of the field on the ghost cell is evaluated (at each
time step) as:
$$\\label{eq:neumann-bc}
    \\genericField_G = g_N \\Delta\\vx\_{Gi} \\cdot\\n + \\genericField_i$$
being *g*<sub>*N*</sub> the value of the gradient to impose on the
boundary for the generic field $\\genericField$ and $\\Delta \\vx\_{Gi}$
is the relative distance vector between the boundary cell *i* and the
corresponding ghost *G*.

<figure>
<embed src="./shared/schemes/mesh-bc.pdf" id="fig:mesh-bc" /><figcaption aria-hidden="true">The mesh data structure including the ghost cells</figcaption>
</figure>

## The Numerics

Again referring to
<a href="#fig:josiepy-arch" data-reference-type="ref" data-reference="fig:josiepy-arch">1</a>,
we talked about the mesh generation module in
<a href="#ssub:mesh_generation" data-reference-type="ref" data-reference="ssub:mesh_generation">1.1.2</a>
and of the physical properties configuration in
<a href="#ssub:physics_of_the_problem" data-reference-type="ref" data-reference="ssub:physics_of_the_problem">1.1.1</a>.
We are now going to introduce the fundamentals for the implementation of
a numerical scheme well suited to a specific problem. The core object of
our discussion is going to be the `josie.scheme.Scheme` class.

## The `Scheme` object

<figure>
<embed src="./shared/schemes/solver-flow.pdf" id="fig:solver-hooks" /><figcaption aria-hidden="true">The flow of operations driven by the <code class="sourceCode python">Solver</code> object</figcaption>
</figure>

<span id="par:the_scheme_object"
label="par:the_scheme_object">\[par:the_scheme_object\]</span> The
`Scheme` object is an <span acronym-label="abc"
acronym-form="singular+short">abc</span>, a sort of interface for those
more comfortable with C-family languages vocabulary, that exposes
methods that act like “hooks” that are called at specific moments during
the simulation lifetime. shows a graphical representation of how the
`josie.solver.Solver` object interacts with the `Scheme`. In order to
modularize even further the code, allowing very precise editing of the
scheme implementation in order to implement exactly what the
mathematical modeler has in mind, the classes `ConvectiveScheme`,
`NonConservativeScheme`, `DiffusiveScheme` and `SourceScheme` are direct
children mixinsof the parent `Scheme` and they expose the interface to
implement the terms explained in
<a href="#chap:numerical" data-reference-type="ref" data-reference="chap:numerical">[chap:numerical]</a>
for the discretization of the reference equation
$$\\pdeFullVector$$
that are:

-   The convective term approximation (see
    <a href="#sub:convective-num" data-reference-type="ref" data-reference="sub:convective-num">[sub:convective-num]</a>),
    $$\\numConvectiveFaces, \\quad \\text{\\texttt{ConvectiveScheme.F}}$$

-   The non-conservative term approximation (see
    <a href="#sub:non-conservative-num" data-reference-type="ref" data-reference="sub:non-conservative-num">[sub:non-conservative-num]</a>),
    $$\\numPreMultipliedNonConservativeFaces, \\quad \\text{\\texttt{NonConservativeScheme.G}}$$

-   The diffusive term approximation (see
    <a href="#sub:diffusive-num" data-reference-type="ref" data-reference="sub:diffusive-num">[sub:diffusive-num]</a>),
    $$\\numDiffusiveFaces, \\quad \\text{\\texttt{DiffusiveScheme.D}}$$

-   The source term approximation (see
    <a href="#sub:source-num" data-reference-type="ref" data-reference="sub:source-num">[sub:source-num]</a>),
    $$\\numSource, \\quad \\text{\\texttt{SourceScheme.s}}$$

In addition the `TimeScheme` class provides the abstract interface to
implement time schemes (without needing to know how the other terms have
been implemented). Some of those terms can be problem specific, notably
the convective term implementation that depends on the eigenstructure of
the problem, others may be reused as-is for most schemes. That is why a
meta-module named `josie.general` contains all the schemes that can be
used “vanilla” for all scheme implementations, notably different
gradient schemes (see
<a href="#sub:diffusive-num" data-reference-type="ref" data-reference="sub:diffusive-num">[sub:diffusive-num]</a>)
stored in `josie.general.schemes.diffusive`, the different time schemes
(see
<a href="#sub:time-num" data-reference-type="ref" data-reference="sub:time-num">[sub:time-num]</a>)
in `josie.general.schemes.time`, and the source schemes (see
<a href="#sub:source-num" data-reference-type="ref" data-reference="sub:source-num">[sub:source-num]</a>)
in `josie.general.schemes.source`. shows graphically this modularized
organization.

<figure>
<embed src="./shared/schemes/josiepy-solvers.pdf" id="fig:josiepy-solvers" /><figcaption aria-hidden="true">Each solver can have its own specific implementation of objects or share the <code>general</code> ones</figcaption>
</figure>

As a final note, let us propose as an example the implementation of the
Rusanov scheme for the Euler system as explained in
<a href="#ssub:euler-rusanov" data-reference-type="ref" data-reference="ssub:euler-rusanov">[ssub:euler-rusanov]</a>.
This implementation is directly extracted from the `josie.euler.schemes`
package.

``` python
class Rusanov(EulerScheme):
    @staticmethod
    def compute_sigma(
        U_L: np.ndarray, U_R: np.ndarray, c_L: np.ndarray, c_R: np.ndarray
    ) -> np.ndarray:
        sigma_L = np.abs(U_L) + c_L[..., np.newaxis]

        sigma_R = np.abs(U_R) + c_R[..., np.newaxis]

        # Concatenate everything in a single array
        sigma_array = np.concatenate((sigma_L, sigma_R), axis=-1)

        # And the we found the max on the last axis (i.e. the maximum value
        # of sigma for each cell)
        sigma = np.max(sigma_array, axis=-1, keepdims=True)

        return sigma

    def F(self, cells: MeshCellSet, neighs: NeighboursCellSet):
        Q_L: EulerState = cells.values.view(EulerState)
        Q_R: EulerState = neighs.values.view(EulerState)

        fields = EulerState.fields

        FS = np.zeros_like(Q_L).view(EulerState)

        # Get normal velocities
        U_L = self.compute_U_norm(Q_L, neighs.normals)
        U_R = self.compute_U_norm(Q_R, neighs.normals)

        # Speed of sound
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]

        sigma = self.compute_sigma(U_L, U_R, c_L, c_R)

        DeltaF = 0.5 * (self.problem.F(cells) + self.problem.F(neighs))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...kl,...l->...k", DeltaF, neighs.normals)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Q_L_cons = Q_L.get_conservative()
        Q_R_cons = Q_R.get_conservative()

        DeltaQ = 0.5 * sigma * (Q_R_cons - Q_L_cons)

        FS.set_conservative(
            neighs.surfaces[..., np.newaxis] * (DeltaF - DeltaQ)
        )

        return FS
```

As it is possible to see, the implementation is very dense and
abstracted. `EulerScheme` is a child of the general mixin
`ConvectiveScheme` that makes easier to set up an Euler problem
(*e.g.* providing <span acronym-label="eos"
acronym-form="singular+short">eos</span> implementation stored in
`josie.euler.eos`). The important aspect is the implementation of the
`ConvectiveScheme.F` method. This method acts on **all** the cells of a
mesh and the corresponding neighbors (as shown graphically in
<a href="#fig:cell-neighbour" data-reference-type="ref" data-reference="fig:cell-neighbour">[fig:cell-neighbour]</a>),
to be treated as a whole vector to benefit of the NumPy acceleration.
Once the “space part” of the scheme implementation is ready, the user
can actually plug it with whatever time scheme they like, exploiting
multi-inheritance, as for example:

``` python
from josie.euler.schemes import Rusanov
from josie.general.schemes.time import RK2

class MyRusanov(Rusanov, RK2):
    pass
    
```

That is all it is needed to do to define a Rusanov scheme integrated in
time with a <span acronym-label="rk"
acronym-form="singular+short">rk</span> 2.

## Wrapping things up, the `Solver` object

Once all the aspects for the definition of the case to simulate are
ready, that is the physical description of the problem
(<a href="#ssub:physics_of_the_problem" data-reference-type="ref" data-reference="ssub:physics_of_the_problem">1.1.1</a>),
the mesh
(<a href="#ssub:mesh_generation" data-reference-type="ref" data-reference="ssub:mesh_generation">1.1.2</a>),
the Numerics
(<a href="#ssub:the_numerics" data-reference-type="ref" data-reference="ssub:the_numerics">1.1.3</a>),
then everything is wrapped into the `Solver` object. As reference, let
us consider
<a href="#fig:solver-hooks" data-reference-type="ref" data-reference="fig:solver-hooks">4</a>,
it shows the interoperability among the different principal objects that
take a role during a simulation, that are the `Solver, Mesh, Scheme`
classes. The `Problem` object is embedded into `Scheme` in a composition
approach. Two main “hooks” are available:

-   `init` method that is called once at the beginning of the simulation
    where the `Mesh` and `Scheme` object can initialize their local data
    structures, storing mesh geometry information and scheme-specific
    data respectively, and the `Solver` objects applies the initial
    condition to the whole domain. For certain objects, the `init`
    method is decomposed in multiple sub-steps that are shown in
    <a href="#fig:solver-hooks" data-reference-type="ref" data-reference="fig:solver-hooks">4</a>
    as dashed circles.

-   `step` method that is called routinely at each time step. This hook
    is decomposed in several different sub-steps within the
    `Scheme object`, notably the `pre_accumulate` method that exposes
    access to all the neighbors of the mesh cells as a whole, in order
    to apply operations that need to access all the neighbors
    simultaneously (as for example the Least Square method to
    approximate fields gradient described in
    <a href="#ssub:least_squares_gradient" data-reference-type="ref" data-reference="ssub:least_squares_gradient">[ssub:least_squares_gradient]</a>).

## I/O Control

The last element we want to discuss in this section is about controlling
the I/O of a test case. The user certainly does not want to save
everything at each time step and, for different simulations, often
different serialization strategies are required. For this reason a
hierarchy of objects are stored in the `josie.io.write` package. Notably
it is possible to choose a `WriteStrategy`, that imposes
(tautologically) which strategy the user wants to use to serialize the
results on disk, *e.g.* every *N* time steps or every *d* seconds. Those
strategies are then taken as input argument by a concrete implementation
of a `Writer` object, that takes care of actually serializing the data
to disk. This structure allows to implement very easily and
independently different serialization strategies and different
serialization drivers. At the date of editing of this manuscript, the
`josie.io.write.writer` module allows to serialize simulation results
into XDMF , memory (but be careful to not fill up your machine RAM!),
and nowhere (that is always a sound option). Other drivers are indeed
very easy to implement at need. As a final note we show how to finally
run a simulation saving data every 0.01 s in a XDMF file,

``` python
from josie.io.write.strategy import TimeStrategy
from josie.io.write.writer import XDMFWriter
writer = XDMFWriter("euler.xdmf", 
    TimeStrategy(dt_save=0.01, animate=True), 
        solver, final_time=1, CFL=0.2)
writer.solve()
```

Finally a complete test case “main” file, from top to bottom, is shown
in
<a href="#sec:josie-euler-jet" data-reference-type="ref" data-reference="sec:josie-euler-jet">3</a>.
The test case is a 2D jet governed by the Euler system of equations as
presented in
<a href="#ssub:the_euler_system" data-reference-type="ref" data-reference="ssub:the_euler_system">[ssub:the_euler_system]</a>.
The code is quite verbose in order to explain each step of the code, but
a complete non-trivial simulation for an Euler jet can be written in
about 100 lines.

