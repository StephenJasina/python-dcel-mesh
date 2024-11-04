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
            self.index: int = index
            self._halfedge: typing.Optional[Mesh.Halfedge] = None
            self._most_clockwise: typing.List[Mesh.Halfedge] = []

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

            This function assumes that the faces incident to this vertex
            are connected contiguously.

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

            This function assumes that the faces incident to this vertex
            are connected contiguously.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges_out():
                yield halfedge.previous

        def vertices(self) -> typing.Iterator['Mesh.Vertex']:
            """
            Circulate over the vertices adjacent to this vertex.

            This function assumes that the faces incident to this vertex
            are connected contiguously.

            By convention, circulation is counterclockwise.
            """
            halfedge = None
            for halfedge in self.halfedges_out():
                yield halfedge.destination
            if halfedge is not None and self.is_on_boundary():
                yield halfedge.previous.origin

        def edges(self) -> typing.Iterator['Mesh.Edge']:
            """
            Circulate over the edges incident to this vertex.

            This function assumes that the faces incident to this vertex
            are connected contiguously.

            By convention, circulation is counterclockwise.
            """
            halfedge = None
            for halfedge in self.halfedges_out():
                yield halfedge.edge
            if halfedge is not None and self.is_on_boundary():
                yield halfedge.previous.edge

        def faces(self) -> typing.Iterator['Mesh.Face']:
            """
            Circulate over the incident faces of this vertex.

            This function assumes that the faces incident to this vertex
            are connected contiguously.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges_out():
                yield halfedge.face

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
            self.index: int = index
            self.origin: Mesh.Vertex = origin
            self.destination: Mesh.Vertex = None              # type: ignore
            self.next: Mesh.Halfedge = None                   # type: ignore
            self.previous: Mesh.Halfedge = None               # type: ignore
            self.twin: typing.Optional[Mesh.Halfedge] = None  # type: ignore
            self.edge: Mesh.Edge = None                       # type: ignore
            self.face: Mesh.Face = None                       # type: ignore

        def counterclockwise(self) -> typing.Optional['Mesh.Halfedge']:
            """
            Get the next halfedge from the same origin vertex.

            This function assumes the faces surrounding the origin
            vertex are connected edge-to-edge. This halfedge is the last
            one around the vertex, `None` is returned.

            By convention, traversal is counterclockwise.
            """
            return self.previous.twin

        def clockwise(self) -> typing.Optional['Mesh.Halfedge']:
            """
            Get the previous halfedge from the same origin vertex.

            This function assumes the faces surrounding the origin
            vertex are connected edge-to-edge. This halfedge is the
            first one around the vertex, `None` is returned.

            By convention, traversal is counterclockwise (so the
            returned halfedge is found by going clockwise).
            """
            if self.twin is None:
                return None
            return self.twin.next

        def is_on_boundary(self) -> bool:
            """Get whether this halfedge is on the boundary."""
            return self.twin is None

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
            self.index: int = index
            self._halfedge: Mesh.Halfedge = halfedge

        def is_on_boundary(self) -> bool:
            """Return whether this edge is on the boundary of the mesh."""
            return self._halfedge.is_on_boundary()

        def vertices(self) -> typing.Iterator['Mesh.Vertex']:
            """Iterate over the vertices incident to this edge."""
            yield self._halfedge.origin
            yield self._halfedge.destination

        def halfedges(self) -> typing.Iterator['Mesh.Halfedge']:
            """Iterate over the halfedges incident to this edge."""
            yield self._halfedge
            twin = self._halfedge.twin
            if twin is not None:
                yield twin

        def faces(self) -> typing.Iterator['Mesh.Face']:
            """Iterate over the faces incident to this edge."""
            for halfedge in self.halfedges():
                yield halfedge.face

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
            self.index: int = index
            self._halfedge: Mesh.Halfedge = halfedge

        def halfedges(self) -> typing.Iterator['Mesh.Halfedge']:
            """
            Circulate over the halfedges incident to this face.

            By convention, circulation is counterclockwise.
            """
            halfedge = self._halfedge
            while True:
                yield halfedge
                halfedge = halfedge.next
                if halfedge == self._halfedge:
                    break

        def vertices(self) -> typing.Iterator['Mesh.Vertex']:
            """
            Circulate over the vertices incident to this face.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges():
                yield halfedge.origin

        def edges(self) -> typing.Iterator['Mesh.Edge']:
            """
            Circulate over the edges incident to this face.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges():
                yield halfedge.edge

        def faces(self) -> typing.Iterator['Mesh.Face']:
            """
            Circulate over the faces adjacent to this face.

            By convention, circulation is counterclockwise.
            """
            for halfedge in self.halfedges():
                twin = halfedge.twin
                if twin is None:
                    continue
                yield twin.face

    class IllegalMeshException(Exception):
        """Class representing errors due to bad mesh construction."""

    def __init__(
        self,
        n_vertices: int = 0,
        faces: typing.Optional[typing.List[typing.List[int]]] = None
    ):
        """
        Create a mesh.

        The mesh will have the given number of vertices faces determined
        by `faces`. `faces` should be a list of triples of vertex
        indices oriented counterclockwise. Each vertex index must be
        less than `n_vertices`.
        """
        self._vertices: typing.List[typing.Optional[Mesh.Vertex]] = []
        self.n_vertices: int = 0
        self._halfedges: typing.List[typing.Optional[Mesh.Halfedge]] = []
        self.n_halfedges: int = 0
        self._edges: typing.List[typing.Optional[Mesh.Edge]] = []
        self.n_edges: int = 0
        self._faces: typing.List[typing.Optional[Mesh.Face]] = []
        self.n_faces: int = 0

        # Keep track of the edges based on their (origin, destination)
        # pairs. This allows for quick lookup to manage twin halfedges,
        # as well as to look for duplicate halfedges.
        self._halfedge_lookup: typing.Dict[typing.Tuple[int, int],
                                           Mesh.Halfedge] = {}

        for _ in range(n_vertices):
            self.add_vertex()

        if faces is not None:
            for vertex_indices in faces:
                self.add_face(vertex_indices)

    def vertices(self) -> typing.Iterator['Mesh.Vertex']:
        """Iterate over the vertices in this mesh."""
        for vertex in self._vertices:
            if vertex is not None:
                yield vertex

    def get_vertex(self, index: int) -> 'Mesh.Vertex':
        """
        Return the vertex with the given index.

        If the requested vertex does not exist, raise a
        `Mesh.IllegalMeshException`.
        """
        if index >= len(self._vertices):
            raise Mesh.IllegalMeshException(
                f'Vertex {index} does not exist'
            )
        vertex = self._vertices[index]
        if vertex is None:
            raise Mesh.IllegalMeshException(
                f'Vertex {index} does not exist'
            )
        return vertex

    def add_vertex(self) -> 'Mesh.Vertex':
        """Add a vertex to this mesh, and return it."""
        vertex = self.Vertex(len(self._vertices))
        self._vertices.append(vertex)
        self.n_vertices += 1

        return vertex

    def remove_vertex(self, index: int) -> None:
        """
        Remove a vertex from this mesh.

        Also remove incident halfedges, edges, and faces.

        If the requested vertex does not exist, raise a
        `Mesh.IllegalMeshException`.
        """
        vertex = self.get_vertex(index)

        # Remove the faces incident to the vertex. This will
        # automatically also remove the incident halfedges and edges
        while vertex._halfedge is not None:
            self.remove_face(next(vertex.faces()).index)

        # Remove the actual vertex
        self._vertices[index] = None
        self.n_vertices -= 1

    def halfedges(self) -> typing.Iterator['Mesh.Halfedge']:
        """Iterate over the halfedges in this mesh."""
        for halfedge in self._halfedges:
            if halfedge is not None:
                yield halfedge

    def get_halfedge(self, index: int, second: typing.Optional[int] = None) \
            -> 'Mesh.Halfedge':
        """
        Return the halfedge with the given index.

        If two indices are given, return instead the halfedge between
        those two indices.

        If the requested halfedge does not exist, raise a
        `Mesh.IllegalMeshException`.
        """
        if second is None:
            if index >= len(self._halfedges):
                raise Mesh.IllegalMeshException(
                    f'Halfedge {index} does not exist'
                )
            halfedge = self._halfedges[index]
            if halfedge is None:
                raise Mesh.IllegalMeshException(
                    f'Halfedge {index} does not exist'
                )
            return halfedge

        key = (index, second)
        if key in self._halfedge_lookup:
            return self._halfedge_lookup[key]
        raise Mesh.IllegalMeshException(
            f'Halfedge ({index}, {second}) does not exist'
        )

    def edges(self) -> typing.Iterator['Mesh.Edge']:
        """Iterate over the edges in this mesh."""
        for edge in self._edges:
            if edge is not None:
                yield edge

    def get_edge(self, index: int, second: typing.Optional[int] = None) \
            -> 'Mesh.Edge':
        """
        Return the edge with the given index.

        If two indices are given, return instead the edge between those
        two indices.

        If the requested edge does not exist, raise a
        `Mesh.IllegalMeshException`.
        """
        if second is None:
            if index >= len(self._edges):
                raise Mesh.IllegalMeshException(
                    f'Edge {index} does not exist'
                )
            edge = self._edges[index]
            if edge is None:
                raise Mesh.IllegalMeshException(
                    f'Edge {index} does not exist'
                )
            return edge

        for key in ((index, second), (second, index)):
            if key in self._halfedge_lookup:
                return self._halfedge_lookup[key].edge
        raise Mesh.IllegalMeshException(
            f'Edge {{{index}, {second}}} does not exist'
        )

    def faces(self) -> typing.Iterator['Mesh.Face']:
        """Iterate over the faces in this mesh."""
        for face in self._faces:
            if face is not None:
                yield face

    def get_face(self, index: int) -> 'Mesh.Face':
        """
        Return the face with the given index.

        If the requested face does not exist, raise a
        `Mesh.IllegalMeshException`.
        """
        face = self._faces[index]
        if face is None:
            raise Mesh.IllegalMeshException(
                f'Face {index} does not exist'
            )
        return face

    def add_face(self, vertex_indices: typing.List[int]) -> 'Mesh.Face':
        """
        Add an (oriented) face to this mesh, and return it.

        The face is defined by the given vertex indices. This function
        assumes that the vertices already exist.
        """
        # Create the face and its incident halfedges
        halfedges: typing.List[Mesh.Halfedge] = []
        for i, vertex_index in enumerate(vertex_indices):
            # Validity checking
            if vertex_index >= len(self._vertices):
                raise self.IllegalMeshException(
                    f'Vertex {vertex_index} does not exist'
                )
            vertex = self._vertices[vertex_index]
            if vertex is None:
                raise self.IllegalMeshException(
                    f'Vertex {vertex_index} does not exist'
                )

            halfedges.append(self.Halfedge(i + len(self._halfedges), vertex))
        face = self.Face(len(self._faces), halfedges[0])

        # Validity checking
        n_sides = len(vertex_indices)
        for halfedge1, halfedge2 in itertools.islice(
            itertools.pairwise(itertools.cycle(halfedges)), n_sides
        ):
            if not halfedge1.origin.is_on_boundary():
                raise self.IllegalMeshException(
                    f'Vertex {halfedge1.origin.index} '
                    'cannot have multiple rings of faces'
                )

            key = (halfedge1.origin.index, halfedge2.origin.index)
            if key in self._halfedge_lookup:
                raise self.IllegalMeshException(
                    f'Halfedge {key[0]} -> {key[1]} defined twice'
                )

        for halfedge1, halfedge2 in itertools.islice(
            itertools.pairwise(itertools.cycle(halfedges)), n_sides
        ):
            # Update destination, next, and previous
            halfedge1.destination = halfedge2.origin
            halfedge1.next = halfedge2
            halfedge2.previous = halfedge1

            # Update halfedge list and halfedge lookup
            self._halfedges.append(halfedge1)
            self._halfedge_lookup[
                halfedge1.origin.index, halfedge1.destination.index
            ] = halfedge1

            # Update twin and edge
            twin_key = (halfedge1.destination.index, halfedge1.origin.index)
            if twin_key in self._halfedge_lookup:
                twin = self._halfedge_lookup[twin_key]
                twin.twin = halfedge1
                halfedge1.twin = twin
                halfedge1.edge = twin.edge
            else:
                edge = Mesh.Edge(len(self._edges), halfedge1)
                halfedge1.edge = edge
                self._edges.append(edge)
                self.n_edges += 1

            # Update face
            halfedge1.face = face

        for halfedge in halfedges:
            vertex = halfedge.origin
            # Update vertex's information
            if vertex._halfedge is None:
                vertex._halfedge = halfedge
                vertex._most_clockwise.append(halfedge)
                continue

            # A vertex's halfedge should be as clockwise as possible
            halfedge_counterclockwise = halfedge.counterclockwise()
            if halfedge_counterclockwise is not None:
                vertex._most_clockwise = [
                    h
                    for h in vertex._most_clockwise
                    if h.index != halfedge_counterclockwise.index
                ]
            if halfedge.clockwise() is None:
                vertex._most_clockwise.append(halfedge)
            vertex._halfedge = (
                vertex._most_clockwise[0]
                if vertex._most_clockwise
                else halfedge
            )

        self._faces.append(face)
        self.n_halfedges += n_sides
        self.n_faces += 1
        return face

    def remove_face(self, index: int) -> None:
        """
        Remove a face with the given index and its incident half edges.

        Also remove incident edges if they have no remaining half edges.

        If the requested face does not exist, raise a
        `Mesh.IllegalMeshException`.
        """
        face = self.get_face(index)

        for halfedge in face.halfedges():
            # Fix the twin and edge
            if not halfedge.is_on_boundary():
                halfedge.twin.twin = None
                halfedge.edge._halfedge = halfedge.twin
            else:
                self._edges[halfedge.edge.index] = None
                self.n_edges -= 1

            # Fix the vertex
            vertex = halfedge.origin
            vertex._most_clockwise = [
                h
                for h in vertex._most_clockwise
                if h.index != halfedge.index
            ]
            halfedge_counterclockwise = halfedge.counterclockwise()
            if halfedge_counterclockwise is not None:
                vertex._most_clockwise.append(halfedge_counterclockwise)
            vertex._halfedge = (
                vertex._most_clockwise[0]
                if vertex._most_clockwise
                else None
            )

            # Actually remove the halfedge
            del self._halfedge_lookup[
                halfedge.origin.index, halfedge.destination.index
            ]
            self._halfedges[halfedge.index] = None
            self.n_halfedges -= 1

        self._faces[face.index] = None
        self.n_faces -= 1

    def reindex(self) -> typing.List[int]:
        """
        Reindex the indices of the mesh elements so they are contiguous.

        This function is meant to be called after removing faces from
        the mesh. The idea is that deleting faces and vertices may cause
        the set of vertex indices to have gaps. This function relabels
        the vertices so they are 0, 1, 2, ...

        In addition, this function renames halfedges, edges, and faces
        so that they have contiguous numbering.

        This method returns a mapping from new vertex labels to old
        vertex labels (in the form of a list).
        """
        # Fix vertices
        self._vertices = [
            vertex
            for vertex in self._vertices
            if vertex is not None
        ]
        mapping = [
            vertex.index
            for vertex in self._vertices
        ]
        for index, vertex in enumerate(self._vertices):
            vertex.index = index

        # Fix halfedges
        self._halfedges = [
            halfedge
            for halfedge in self._halfedges
            if halfedge is not None
        ]
        for index, halfedge in enumerate(self._halfedges):
            halfedge.index = index
        self._halfedge_lookup = {
            (halfedge.origin.index, halfedge.destination.index): halfedge
            for halfedge in self._halfedges
        }

        # Fix edges
        self._edges = [
            edge
            for edge in self._edges
            if edge is not None
        ]
        for index, edge in enumerate(self._edges):
            edge.index = index

        # Fix faces
        self._faces = [
            face
            for face in self._faces
            if face is not None
        ]
        for index, face in enumerate(self._faces):
            face.index = index

        return mapping
