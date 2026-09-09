# Data access and attribution

OpenGray distributes software. It does not bundle original patient-data archives, cached dose matrices, reference plans, or study case exports.

The supported importer reads archives obtained directly from the [OpenKBP-Opt authors](https://github.com/ababier/open-kbp-opt). Follow the upstream access instructions and applicable source-data terms. The [OpenKBP-Opt repository license](https://github.com/ababier/open-kbp-opt/blob/master/LICENSE) and the [OpenKBP repository license](https://github.com/ababier/open-kbp/blob/master/LICENSE) are MIT licenses for their software and associated documentation. OpenGray's Apache license does not replace upstream or source-collection terms.

Ingested files remain local. The importer records source checksums and provenance. Keep downloaded archives, derived patient arrays, and run artifacts outside public commits.

Cite both the underlying resource and its optimization extension when applicable:

- Babier A, et al. OpenKBP. *Medical Physics*. 2021;48:5549–5561. [doi:10.1002/mp.14845](https://doi.org/10.1002/mp.14845).
- Babier A, et al. OpenKBP-Opt. *Physics in Medicine & Biology*. 2022;67:185012. [doi:10.1088/1361-6560/ac8044](https://doi.org/10.1088/1361-6560/ac8044).

Attribution and license notices for the upstream software are retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
