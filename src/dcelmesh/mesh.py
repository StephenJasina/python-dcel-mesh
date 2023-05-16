"""
Mesh implementation based loosely on OpenMesh.

The homepages of OpenMesh and its Python bindings can be found here:
    https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh
    https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/openmesh-python
"""

import itertools
import typing


class Mesh:
    """Class representing a mesh built of vertices and faces."""

    class Vertex:
        """
        Class representing a single vertex in a mesh.

        The vertex has a pointer to some halfedge incident to it. For
        efficiency, if the vertex is on the boundary, the halfedge it
        points to will also be on the boundary.
        """

        def __init__(self, index: int):
            """
            Create a vertex.

            `index` is a unique key, and `halfedge` is the
            clockwise-most incident halfedge if it exists (ties broken
            arbitrarily).
            """
            self._index: int = index
            self._halfedge: typing.Optional[Mesh.Halfedge] = None

        def index(self) -> int:
            """Get this vertex's key."""
            return self._index

        def is_on_boundary(self) -> bool:
            """
            Return whether this vertex is on the boundary of the mesh.

            By convention, if the vertex has no incident faces, it is
            considered to be on the boundary.
            """
            return self._halfedge is None or self._halfedge.is_on_boundary()

        def halfedges_out(self) -> typing.Iterator['Mesh.Halfedge']:
            """
            Circulate over the outgoing halfedges of this vertex.

            By convention, circulation is counterclockwise.
            """
            halfedge = self._halfedge
            while halfedge is not None:
                yield halfedge
                halfedge = halfedge.counterclockwise()
                if halfedge == self._halfedge:
                    break

        def halfedges_in(self) -> typing.Iterator['Mesh.Halfedge']:
            """
            Circulate over the ingoing halfedges of this vertex.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges_out():
                yield halfedge.previous()

        def vertices(self) -> typing.Iterator['Mesh.Vertex']:
            """
            Circulate over the vertices adjacent to this vertex.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges_out():
                yield halfedge.destination()
            if self.is_on_boundary():
                yield halfedge.previous().origin()

        def edges(self) -> typing.Iterator['Mesh.Edge']:
            """
            Circulate over the edges incident to this vertex.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges_out():
                yield halfedge.edge()
            if self.is_on_boundary():
                yield halfedge.previous().edge()

        def faces(self) -> typing.Iterator['Mesh.Face']:
            """
            Circulate over the incident faces of this vertex.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges_out():
                yield halfedge.face()

    class Halfedge:
        """
        Class representing a halfedge in a mesh.

        The halfedge has pointers to
          * Its origin vertex
          * The next halfedge going counterclockwise around its incident
            face
          * The previous halfedge going counterclockwise around its
            incident face
          * The twin halfedge with the same vertices but opposite
            direction
          * The incident face
        """

        def __init__(self, index: int, origin: 'Mesh.Vertex'):
            """
            Create a halfedge.

            `index` is a unique key, and `origin` is the vertex this
            halfedge points out of.
            """
            self._index: int = index
            self._origin: Mesh.Vertex = origin
            self._next: typing.Optional[Mesh.Halfedge] = None
            self._previous: typing.Optional[Mesh.Halfedge] = None
            self._twin: typing.Optional[Mesh.Halfedge] = None
            self._edge: typing.Optional[Mesh.Edge] = None
            self._face: typing.Optional[Mesh.Face] = None

        def index(self) -> int:
            """Get this halfedge's key."""
            return self._index

        def origin(self) -> 'Mesh.Vertex':
            """Get the origin of this halfedge."""
            return self._origin

        def destination(self) -> 'Mesh.Vertex':
            """Get the destination of this halfedge."""
            return self.next().origin()

        def next(self) -> 'Mesh.Halfedge':
            """
            Get the next halfedge when traversing the same face.

            By convention, traversal is counterclockwise.
            """
            if self._next is None:
                raise Mesh.IllegalMeshException(
                    'Halfedge '
                    f'{(self.origin().index(), self.destination().index())} '
                    'has no next'
                )
            return self._next

        def previous(self) -> 'Mesh.Halfedge':
            """
            Get the previous halfedge when traversing the same face.

            By convention, traversal is counterclockwise (so the
            returned halfedge is found by going clockwise).
            """
            if self._previous is None:
                raise Mesh.IllegalMeshException(
                    'Halfedge '
                    f'{(self.origin().index(), self.destination().index())} '
                    'has no previous'
                )
            return self._previous

        def twin(self) -> typing.Optional['Mesh.Halfedge']:
            """
            Get the halfedge pointing in the opposite direction.

            This function returns `None` if the the halfedge does not
            exist in the mesh. In particular, this happens when the
            halfedge is on the boundary.
            """
            return self._twin

        def counterclockwise(self) -> typing.Optional['Mesh.Halfedge']:
            """
            Get the next halfedge from the same origin vertex.

            This function assumes the faces surrounding the origin
            vertex are connected edge-to-edge. This halfedge is the last
            one around the vertex, `None` is returned.

            By convention, traversal is counterclockwise.
            """
            return self.previous().twin()

        def clockwise(self) -> typing.Optional['Mesh.Halfedge']:
            """
            Get the previous halfedge from the same origin vertex.

            This function assumes the faces surrounding the origin
            vertex are connected edge-to-edge. This halfedge is the
            first one around the vertex, `None` is returned.

            By convention, traversal is counterclockwise (so the
            returned halfedge is found by going clockwise).
            """
            twin = self.twin()
            if twin is None:
                return None
            return twin.next()

        def is_on_boundary(self) -> bool:
            """Get whether this halfedge is on the boundary."""
            return self.twin() is None

        def edge(self) -> 'Mesh.Edge':
            """Get the edge incident to this halfedge."""
            if self._edge is None:
                raise Mesh.IllegalMeshException(
                    'Halfedge '
                    f'{(self.origin().index(), self.destination().index())} '
                    'does not correspond to any edge'
                )
            return self._edge

        def face(self) -> 'Mesh.Face':
            """Get the face incident to this halfedge."""
            if self._face is None:
                raise Mesh.IllegalMeshException(
                    'Halfedge '
                    f'{(self.origin().index(), self.destination().index())} '
                    'is not incident to any face'
                )
            return self._face

    class Edge:
        """
        Class representing an edge in a mesh.

        The edge has a pointer to one of its corresponding halfedges.
        """

        def __init__(self, index: int, halfedge: 'Mesh.Halfedge'):
            """
            Create an edge.

            `index` is a unique key, and `halfedge` is any halfedge
            incident to this edge.
            """
            self._index: int = index
            self._halfedge: Mesh.Halfedge = halfedge

        def index(self) -> int:
            """Get this edge's key."""
            return self._index

        def is_on_boundary(self) -> bool:
            """Return whether this edge is on the boundary of the mesh."""
            return self._halfedge.is_on_boundary()

        def vertices(self) -> typing.Iterator['Mesh.Vertex']:
            """Iterate over the vertices incident to this edge."""
            yield self._halfedge.origin()
            yield self._halfedge.destination()

        def halfedges(self) -> typing.Iterator['Mesh.Halfedge']:
            """Iterate over the halfedges incident to this edge."""
            yield self._halfedge
            twin = self._halfedge.twin()
            if twin is not None:
                yield twin

        def faces(self) -> typing.Iterator['Mesh.Face']:
            """Iterate over the faces incident to this edge."""
            for halfedge in self.halfedges():
                yield halfedge.face()

    class Face:
        """
        Class representing a face in a mesh.

        The face has a pointer to some halfedge incident to it.
        """

        def __init__(self, index: int, halfedge: 'Mesh.Halfedge'):
            """
            Create a face.

            `index` is a unique key, and `halfedge` is any halfedge
            incident to this face.
            """
            self._index: int = index
            self._halfedge: Mesh.Halfedge = halfedge

        def index(self) -> int:
            """Get this face's key."""
            return self._index

        def halfedges(self) -> typing.Iterator['Mesh.Halfedge']:
            """
            Circulate over the halfedges incident to this face.

            By convention, circulation is counterclockwise.
            """
            halfedge = self._halfedge
            while True:
                yield halfedge
                halfedge = halfedge.next()
                if halfedge == self._halfedge:
                    break

        def vertices(self) -> typing.Iterator['Mesh.Vertex']:
            """
            Circulate over the vertices incident to this face.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges():
                yield halfedge.origin()

        def edges(self) -> typing.Iterator['Mesh.Edge']:
            """
            Circulate over the edges incident to this face.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges():
                yield halfedge.edge()

        def faces(self) -> typing.Iterator['Mesh.Face']:
            """
            Circulate over the faces adjacent to this face.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges():
                twin = halfedge.twin()
                if twin is None:
                    continue
                yield twin.face()

    class IllegalMeshException(Exception):
        """Class representing errors due to bad mesh construction."""

        pass

    def __init__(self, n_vertices: int = 0,
                 faces: typing.List[typing.List[int]] = []):
        """
        Create a mesh.

        The mesh will have the given number of vertices faces determined
        by `faces`. `faces` should be a list of triples of vertex
        indices oriented counterclockwise. Each vertex index must be
        less than `n_vertices`.
        """
        self._vertices: typing.List[Mesh.Vertex] = []
        self._halfedges: typing.List[Mesh.Halfedge] = []
        self._edges: typing.List[Mesh.Edge] = []
        self._faces: typing.List[Mesh.Face] = []

        # Keep track of the edges based on their (origin, destination)
        # pairs. This allows for quick lookup to manage twin halfedges,
        # as well as to look for duplicate halfedges.
        self._halfedge_lookup: typing.Dict[typing.Tuple[int, int],
                                           Mesh.Halfedge] = {}

        for _ in range(n_vertices):
            self.add_vertex()

        for vertex_indices in faces:
            self.add_face(vertex_indices)

    def add_vertex(self) -> 'Mesh.Vertex':
        """Add a vertex to this mesh, and return it."""
        vertex = self.Vertex(len(self._vertices))
        self._vertices.append(vertex)

        return vertex

    def get_vertex(self, index: int) -> 'Mesh.Vertex':
        """Return the vertex with the given index."""
        return self._vertices[index]

    def add_face(self, vertex_indices: typing.List[int]) -> 'Mesh.Face':
        """
        Add an (oriented) face to this mesh, and return it.

        The face is defined by the given vertex indices. This function
        assumes that the vertices already exist.
        """
        # Validity checking
        for vertex_index in vertex_indices:
            if vertex_index >= len(self._vertices):
                raise self.IllegalMeshException(
                    f'Vertex {vertex_index} does not exist'
                )

        halfedges: typing.List[Mesh.Halfedge] = []
        for i, vertex_index in enumerate(vertex_indices):
            halfedge = self.Halfedge(i + len(self._halfedges),
                                     self._vertices[vertex_index])
            halfedges.append(halfedge)
        halfedges.append(halfedges[0])

        face = self.Face(len(self._faces), halfedges[0])

        # More validity checking
        for halfedge1, halfedge2 in itertools.pairwise(halfedges):
            key = (halfedge1.origin().index(), halfedge2.origin().index())
            if key in self._halfedge_lookup:
                raise self.IllegalMeshException(
                    f'Halfedge {key} defined twice'
                )

            if not halfedge1.origin().is_on_boundary():
                raise self.IllegalMeshException(
                    f'Vertex {key[0]} cannot have multiple rings of faces'
                )

        for halfedge1, halfedge2 in itertools.pairwise(halfedges):
            # Update halfedge list and halfedge lookup
            self._halfedges.append(halfedge1)
            key = (halfedge1.origin().index(), halfedge2.origin().index())
            if key in self._halfedge_lookup:
                raise self.IllegalMeshException(
                    f'Halfedge {key} defined twice'
                )
            self._halfedge_lookup[key] = halfedge1

            # Update next and previous
            halfedge1._next = halfedge2
            halfedge2._previous = halfedge1

            # Update twin and edge
            twin_key = (halfedge2.origin().index(), halfedge1.origin().index())
            if twin_key in self._halfedge_lookup:
                twin = self._halfedge_lookup[twin_key]
                twin._twin = halfedge1
                halfedge1._twin = twin
                halfedge1._edge = twin._edge
            else:
                edge = Mesh.Edge(len(self._edges), halfedge1)
                halfedge1._edge = edge
                self._edges.append(edge)

            # Update face
            halfedge1._face = face

        for halfedge in halfedges[:-1]:
            # Update vertex
            if halfedge._origin._halfedge is None:
                halfedge._origin._halfedge = halfedge

            # A vertex's halfedge should be as clockwise as possible
            he = halfedge._origin._halfedge
            he_clockwise = he.clockwise()
            while he_clockwise is not None \
                    and he != halfedge:
                he = he_clockwise
                he_clockwise = he.clockwise()
            halfedge._origin._halfedge = he

        self._faces.append(face)
        return face

    def vertices(self) -> typing.Iterator['Mesh.Vertex']:
        """Iterate over the vertices in this mesh."""
        for vertex in self._vertices:
            yield vertex

    def n_vertices(self) -> int:
        """Get the number of vertices in this mesh."""
        return len(self._vertices)

    def halfedges(self) -> typing.Iterator['Mesh.Halfedge']:
        """Iterate over the halfedges in this mesh."""
        for halfedge in self._halfedges:
            yield halfedge

    def n_halfedges(self) -> int:
        """Get the number of halfedges in this mesh."""
        return len(self._halfedges)

    def edges(self) -> typing.Iterator['Mesh.Edge']:
        """Iterate over the edges in this mesh."""
        for edge in self._edges:
            yield edge

    def n_edges(self) -> int:
        """Get the number of edges in this mesh."""
        return len(self._edges)

    def faces(self) -> typing.Iterator['Mesh.Face']:
        """Iterate over the faces in this mesh."""
        for face in self._faces:
            yield face

    def n_faces(self) -> int:
        """Get the number of faces in this mesh."""
        return len(self._faces)
