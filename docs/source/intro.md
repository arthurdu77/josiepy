# Motivation

The situation of simulation codes in the framework of physics systems
governed by <span acronym-label="pde"
acronym-form="plural+short">pdes</span> is characterized by the
co-existence of two kinds of software: industrial codes that are
extremely flexible and performant, with enhanced multiphysics
capabilites. Notable examples are closed commercial or enterprise
specific suites like CEDRE from ONERA , products from NUMECA, ANSYS,
ConvergeCFD, COMSOL just to cite few. <span acronym-label="foss"
acronym-form="singular+short">foss</span> examples are also available
like OpenFOAM or FreeFem . These advanced suites are often optimized for
the execution time on CPU-centric virtual <span acronym-label="numa"
acronym-form="singular+short">numa</span> architectures on <span
acronym-label="hpc" acronym-form="singular+short">hpc</span> clusters.
On the other side, academic codes that work on simplified
configurations, sometimes limited to low dimensionalities, that assume
lots of hypotheses on the mesh structure (*e.g.* undeformed, regular
Cartesian meshes with fixed spacing) and can have limited applicability
on real scenarios. For the first category, often the chosen
implementation language is low-level and compiled, often being C, with a
steep learning curve and a real difficulty of integrating localized
modifications to specific part of the code. Testing different numerical
approaches to solve a complex <span acronym-label="pde"
acronym-form="singular+short">pde</span> system can be long, cumbersome
and sometimes frustrating, if the solver you need to embed your schemes
into is written in a complex, compiled, programming language such as C
or you have limited access to the entire code base or documentation.
Despite C being invented to produce pain, and pain being cathartic
sometimes, *when half way through the journey of your life, you find
yourself in a gloomy wood, because the path which led aright was
lost*[1], having an easy framework to test your numerical experiments
without the need of a savvy guide like Virgil, can be useful. Jokes
apart, having C a clear established role in the field, we believe that
there is room for a mid-ground solution that can be optimized on the
global “Time to Market” of a simulation including in the evaluation
metrics also the time a developers needs to understand and get
acquainted with a certain code base in order to set up a reasonably
complex case. In addition to that, nowadays, the Python language is used
in all sort of HPC fields, like Data Analysis and <span
acronym-label="ai" acronym-form="singular+short">ai</span>, its
ecosystem is equipped with powerful compiled extensions like NumPy ,
that allow fast debugging cycles without sacrificing too much
performance in the execution.

Here comes `josiepy` , a Python library that allows to easily solve
multidimensional[2] <span acronym-label="pde"
acronym-form="singular+short">pde</span> systems encoded in the same
spirit as described in
<a href="#sec:basic-numerical" data-reference-type="ref" data-reference="sec:basic-numerical">[sec:basic-numerical]</a>.
It is heavily based on the solid fundamentals of the NumPy —that
incidentally allows to use different backends like `DistArray` , for
distributed arrays allowing parallel execution on CPUs, or `cupy` , for
execution on GPUs, or even for hybrid large scale workflows— and it
basically allows to “program” your test case in Python, without the need
of cryptic and limited configuration files. In addition to that,
`josiepy` also ships a basic structured mesher based on <span
acronym-label="tfi" acronym-form="singular+short">tfi</span> that allows
to “program” your meshes directly in Python, without the need of
interfacing with other meshing tools. The simulation results can be
exported in common files such as `XDMF` and inspected in post-processing
tools such as Paraview. The framework aims at providing a fast
prototyping tool granting the possibility to have all the required tools
for a simulation in one place: a structured mesh generator, a selection
of space and time schemes, high-order extensions of those schemes, a
selection of explicit time integrators, everything easily installable
using Python packaging tools and accessible with a familiar programming
language <span acronym-label="api"
acronym-form="singular+short">api</span>. The numerical schemes can be
implemented directly in Python and they can run as fast as NumPy API
allows, that in most cases is close to what compiled high performance
languages like C and C allow .

