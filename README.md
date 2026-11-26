# minimeshb

minimeshb is a small Rust library to read and write a practical subset of
the meshb/sol file formats.

Its goal is to provide simple, dependency-light I/O for common mesh workflows
while staying consistent with the libMeshb conventions for the supported
keywords.

## Scope

- Supports ASCII and binary files:
	- .mesh / .meshb
	- .sol / .solb
- Supports dimensions 2 and 3.
- Uses 0-based indexing in Rust and converts to/from meshb 1-based indexing
	on file I/O.

## Features Present

### Mesh sections

- Vertices
- Edges
- EdgesP2
- Triangles
- TrianglesP2
- Tetrahedra
- TetrahedraP2

### Solution sections

- SolAtVertices
- SolAtEdges
- SolAtTriangles
- SolAtTetrahedra

### Solution value kinds

- Scalar fields
- Vector fields
- Symmetric matrix fields

## Features Missing

Compared to the full libMeshb keyword space, minimeshb intentionally does not
implement many advanced sections yet.

### Missing element and topology families

- Quadrilaterals and higher-order quadrilateral families
- Prisms and pyramids (including higher-order families)
- Hexahedra (including Q2/Q3/Q4 families)
- Polygon/polyhedron keyword families

### Missing solution families

- SolAtQuadrilaterals
- SolAtPrisms
- SolAtHexahedra
- SolAtPolyhedra and boundary polygon solutions
- Integer and dedicated double solution keyword families

### Missing metadata and advanced keywords

- Geometry-link keywords and many CAD association tables
- Ordering keywords for higher-order node permutations
- Partition, color, grain, and various GID keyword families
- Many auxiliary keywords from the full libMeshb reference

### Current format limitations

- Multi-field solution declarations are not supported (single field type only).
- Big-endian binary marker handling is not implemented.

## Building The libmeshb Integration Feature

The optional integration with libMeshb is enabled with the Cargo feature named
libmeshb-sys.

This feature is primarily intended for unit testing and consistency checks
against the reference C library, not for the default pure-Rust workflow.

### Build libMeshb from tag v7.80

1. Get the reference library source at tag v7.80:

	git clone --branch v7.80 --depth 1 https://github.com/LoicMarechal/libMeshb.git

2. Configure and install it:

	cd libMeshb
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
	cmake --build build -j
	cmake --install build

3. Ensure headers and libraries are visible to the Rust build:

	export CPATH=/usr/local/include:${CPATH}
	export LIBRARY_PATH=/usr/local/lib:${LIBRARY_PATH}
	export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

If your environment requires a system include path, install or link
libmeshb7.h so it is reachable from standard include locations.

### Build and test with the feature

Run the test suite with the integration feature enabled:

	cargo test --features libmeshb-sys

Without this feature, minimeshb uses only its native Rust reader/writer.

## Notes

- This crate targets a minimal, commonly useful subset.
- If full format coverage is required, use libMeshb directly or extend this
	crate section-by-section.
