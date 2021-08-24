# Future plans and perspectives

The main focus of `josiepy` as we already mentioned is not to replace or
to compete with established <span acronym-label="cfd"
acronym-form="singular+short">cfd</span> software that can encompass
extremely variegated configurations, mesh types, and industrial-specific
needs. The vision behind the library is to provide an agile framework in
a familiar language that allows to test the implementation of different
aspects of a simulation, notably numerical schemes, , , and so on,
*without sacrificing too much performance*. That is why the envisioned
roadmap is the following:

-   Better integration of the NumPy API that will allow to leverage
    different backends, potentially on GPUs like .

-   Extending the integrated mesh generator to allow the creation of
    block-structured meshes

-   The addition of 3D capability both for mesh generation and
    simulation

-   The addition of more advanced mesh generation algorithms, like the
    one based on elliptic and hyperbolic equations (see
    <a href="#sub:num-mesh-generation" data-reference-type="ref" data-reference="sub:num-mesh-generation">[sub:num-mesh-generation]</a>)

-   The addition of modern turbulence models

-   Improvement of the infrastructure to facilitate interoperability
    with <span acronym-label="hpc"
    acronym-form="singular+short">hpc</span> clusters (notably an agile,
    automatic checkpointing ability)

The code is available with a very permissive license at the link , all
the interested individuals are encouraged to join the GitLab project and
exchange ideas and code to improve the current state of `josiepy`. The
current workable tasks and encountered bugs are readily reported at
<https://gitlab.com/rubendibattista/josiepy/issues> and the official
documentation can be found at [josiepy.rdb.is](josiepy.rdb.is), where
few tutorials can be executed directly on the browser thanks to Jupyter
([jupyter.org](jupyter.org)). The ongoing work of will leverage the
capabilities of `josiepy`. Most of the results presented in
<a href="#chap:numerical" data-reference-type="ref" data-reference="chap:numerical">[chap:numerical]</a>
are promptly available in the repository as Jupyter notebooks or
integration tests that are automatically executed at each commit,
ensuring the correct functioning of the code while it gets improved. At
the current date, the testing suite of the software counts 480 tests
with a code coverage of 93% of the total code base, all executed
automatically at each commit.
