import logging
from paraview.util.vtkAlgorithm import (
    smproxy,
    VTKPythonAlgorithmBase,
    smproperty,
    smdomain,
    smhint,
)
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.util.vtkConstants import (
    VTK_ID_TYPE,
    VTK_UNSIGNED_CHAR,
    VTK_TRIANGLE,
    VTK_TETRA,
    VTK_POLY_LINE,
)
from vtkmodules.vtkCommonCore import vtkDataArraySelection
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet
from vtk.util.numpy_support import numpy_to_vtk
import logging
import os
import numpy.ctypeslib as npct
import ctypes as ct

try:
    dirname = os.path.join(os.path.dirname(__file__), "..", "target", "release")
    lib = npct.load_library("libmeshbreader", dirname)
except:
    dirname = os.path.join(os.path.dirname(__file__), "..", "target", "debug")
    lib = npct.load_library("libmeshbreader", dirname)

c_uint64p = ct.POINTER(ct.c_uint64)
c_int32p = ct.POINTER(ct.c_int32)
c_doublep = ct.POINTER(ct.c_double)


class MeshbReader:

    def __init__(self, fname):

        self._ptr = lib.minimeshb_reader_create(fname.encode("utf-8"))
        assert self._ptr != 0
        self._dim = lib.minimeshb_dimension(self._ptr)

    def __del__(self):

        lib.minimeshb_reader_delete(self._ptr)

    def mesh_info(self):
        return (
            self._dim,
            lib.minimeshb_num_verts(self._ptr),
            lib.minimeshb_num_edges(self._ptr),
            lib.minimeshb_num_triangles(self._ptr),
            lib.minimeshb_num_tetrahedra(self._ptr),
        )

    def read_verts(self):

        n = lib.minimeshb_num_verts(self._ptr)
        verts = np.zeros((n, self._dim), dtype=np.float64)
        tags = np.zeros(n, dtype=np.int32)
        if n > 0:
            if self._dim == 2:
                lib.minimeshb_read_verts_2d(
                    self._ptr,
                    verts.ctypes.data_as(c_doublep),
                    tags.ctypes.data_as(c_int32p),
                )
            elif self._dim == 3:
                lib.minimeshb_read_verts_3d(
                    self._ptr,
                    verts.ctypes.data_as(c_doublep),
                    tags.ctypes.data_as(c_int32p),
                )
            else:
                raise NotImplementedError

        return verts, tags

    def read_edges(self):

        n = lib.minimeshb_num_edges(self._ptr)
        conn = np.zeros((n, 2), dtype=np.uint64)
        tags = np.zeros(n, dtype=np.int32)
        if n > 0:
            lib.minimeshb_read_edges(
                self._ptr,
                conn.ctypes.data_as(c_uint64p),
                tags.ctypes.data_as(c_int32p),
            )

        return (conn, tags)

    def read_triangles(self):

        n = lib.minimeshb_num_triangles(self._ptr)
        conn = np.zeros((n, 3), dtype=np.uint64)
        tags = np.zeros(n, dtype=np.int32)
        if n > 0:
            lib.minimeshb_read_triangles(
                self._ptr,
                conn.ctypes.data_as(c_uint64p),
                tags.ctypes.data_as(c_int32p),
            )

        return (conn, tags)

    def read_tetrahedra(self):

        n = lib.minimeshb_num_tetrahedra(self._ptr)
        conn = np.zeros((n, 4), dtype=np.uint64)
        tags = np.zeros(n, dtype=np.int32)
        if n > 0:
            lib.minimeshb_read_tetrahedra(
                self._ptr,
                conn.ctypes.data_as(c_uint64p),
                tags.ctypes.data_as(c_int32p),
            )

        return (conn, tags)

    def solution_info(self):
        size = np.zeros(2, dtype=np.uint64)
        lib.minimeshb_solution_size(self._ptr, size.ctypes.data_as(c_uint64p))

        return (self._dim, size[0], size[1])

    def read_solution(self):

        size = np.zeros(2, dtype=np.uint64)
        lib.minimeshb_solution_size(self._ptr, size.ctypes.data_as(c_uint64p))
        data = np.zeros(size, dtype=np.float64)
        if size[0] > 0:
            if size[1] == 1:
                lib.minimeshb_read_scalar_solution(
                    self._ptr, data.ctypes.data_as(c_doublep)
                )
            elif size[1] == self._dim:
                if self._dim == 2:
                    lib.minimeshb_read_vector_solution_2d(
                        self._ptr, data.ctypes.data_as(c_doublep)
                    )
                elif self._dim == 3:
                    lib.minimeshb_read_vector_solution_3d(
                        self._ptr, data.ctypes.data_as(c_doublep)
                    )
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()


@smproxy.reader(
    name="PythonMeshbReader",
    label=".mesh(b)/.sol(b) reader",
    extensions=["mesh", "meshb"],
    file_description=".mesh(b) files",
)
class PythonMeshbReader(VTKPythonAlgorithmBase):
    """A reader that reads .mesh(b) files"""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet"
        )
        self._filename = None

        def createModifiedCallback(anobject):
            import weakref

            weakref_obj = weakref.ref(anobject)
            anobject = None

            def _markmodified(*args, **kwars):
                o = weakref_obj()
                if o is not None:
                    o.Modified()

            return _markmodified

        self._arrayselection = vtkDataArraySelection()
        self._arrayselection.AddObserver("ModifiedEvent", createModifiedCallback(self))
        self._mesh_reader = None
        self._sol_readers = []

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(
        extensions=["mesh", "meshb"],
        file_description=".mesh(b)/.sol(b) files",
    )
    def SetFileName(self, name):
        """Specify filename for the file to read."""
        if self._filename != name and os.path.exists(name):
            self._filename = name
            logging.info("Mesh file: %s" % name)
            self._mesh_reader = MeshbReader(name)
            (dim, n_verts, _, _, _) = self._mesh_reader.mesh_info()

            dirname = os.path.dirname(name)
            prefix = os.path.basename(name).replace(".meshb", "").replace(".mesh", "")
            for f in os.listdir(dirname):
                if f.endswith(".sol") or f.endswith(".solb"):
                    if f.startswith(prefix):
                        name = (
                            f.replace(prefix, "")
                            .replace(".solb", "")
                            .replace(".sol", "")
                        )
                        reader = MeshbReader(os.path.join(dirname, f))
                        (dim1, n_verts_1, _) = reader.solution_info()
                        if dim1 == dim and n_verts_1 == n_verts:
                            logging.info("Solution file: %s, %s" % (name, f))
                            self._sol_readers.append(
                                (name, MeshbReader(os.path.join(dirname, f)))
                            )

            self.Modified()

    def _get_timesteps(self):
        return None

    @smproperty.doublevector(
        name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty"
    )
    def GetTimestepValues(self):
        return self._get_timesteps()

    # Array selection API is typical with readers in VTK
    # This is intended to allow ability for users to choose which arrays to
    # load. To expose that in ParaView, simply use the
    # smproperty.dataarrayselection().
    # This method **must** return a `vtkDataArraySelection` instance.
    @smproperty.dataarrayselection(name="Arrays")
    def GetDataArraySelection(self):

        return self._arrayselection

    def RequestInformation(self, request, inInfoVec, outInfoVec):

        for name, reader in self._sol_readers:
            (_, _, m) = reader.solution_info()
            if m == 1:
                self._arrayselection.AddArray(name)
                self._arrayselection.EnableArray(name)
            else:
                for i in range(m):
                    self._arrayselection.AddArray("%s_%d" % (name, i))
                    self._arrayselection.EnableArray("%s_%d" % (name, i))

        return 1

    def EnableAllVariables(self):

        self._arrayselection.EnableAllArrays()
        self._first_load = False
        self.Modified()

    def _read_grid_coordinates(self):

        logging.debug("reading the coordinates")
        xyz, _ = self._mesh_reader.read_verts()
        logging.info("Read %d vertices" % xyz.shape[0])
        return xyz

    def _read_cells(self):

        logging.debug("reading the cells")

        res = {}

        tets, tet_tags = self._mesh_reader.read_tetrahedra()
        res[VTK_TETRA] = [tets, tet_tags]

        tris, tri_tags = self._mesh_reader.read_triangles()
        res[VTK_TRIANGLE] = [tris, tri_tags]

        edgs, edg_tags = self._mesh_reader.read_edges()
        res[VTK_POLY_LINE] = [edgs, edg_tags]

        return res

    def _read_data(self):

        names = []
        data = []

        for name, reader in self._sol_readers:
            sol = reader.read_solution()
            m = sol.shape[1]
            if m == 1:
                names.append(name)
                data.append(sol[:, 0])
            else:
                for i in range(m):
                    names.append("%s_%d" % (name, i))
                    data.append(sol[:, 0])

        return names, data

    def _get_cell_type(self, dim):

        if dim == 1:
            return VTK_POLY_LINE
        elif dim == 2:
            return VTK_TRIANGLE
        elif dim == 3:
            return VTK_TETRA
        else:
            raise NotImplementedError()

    def _get_tags(self, dim, cells):

        _, tags = cells[self._get_cell_type(dim)]
        return np.unique(tags)

    def _create_unstructured_grid(self, dim, tag, xyz, cells):

        logging.debug("create a vtkUnstructuredGrid")
        flg = np.zeros(xyz.shape[0], dtype=bool)

        cell_type = self._get_cell_type(dim)
        conn, tags = cells[cell_type]
        (iels,) = np.nonzero(tags == tag)
        conn = conn[iels, :]
        ids = conn.ravel()
        flg[ids] = True

        (used_verts,) = np.nonzero(flg)
        if used_verts.size == 0:
            return None, None

        new_idx = np.zeros(xyz.shape[0], dtype=np.int64) - 1
        new_idx[used_verts] = np.arange(used_verts.size)

        ug = vtkUnstructuredGrid()
        pug = dsa.WrapDataObject(ug)
        pug.SetPoints(xyz[used_verts, :])

        n, m = conn.shape
        offsets = m * np.arange(n + 1)
        connectivity = new_idx[conn.ravel()]
        cell_types = cell_type * np.ones(n, dtype=np.int64)
        tags = tag * np.ones(n)

        ca = vtkCellArray()
        ca.SetNumberOfCells(offsets.size - 1)
        ca.SetData(
            numpy_to_vtk(offsets, deep=1, array_type=VTK_ID_TYPE),
            numpy_to_vtk(connectivity, deep=1, array_type=VTK_ID_TYPE),
        )
        ct = numpy_to_vtk(cell_types, deep=1, array_type=VTK_UNSIGNED_CHAR)
        ug.SetCells(ct, ca)

        data = pug.GetCellData()
        data.append(tags, "tag")

        return pug, used_verts

    def RequestData(self, request, inInfoVec, outInfoVec):

        xyz = self._read_grid_coordinates()
        cells = self._read_cells()
        dims = [x for x in range(1, 4) if cells[self._get_cell_type(x)][0].size > 0]
        cell_dim = max(dims)
        data = self._read_data()

        mbds = vtkMultiBlockDataSet.GetData(outInfoVec, 0)
        iz = 0
        for dim in [cell_dim, cell_dim - 1]:
            tags = self._get_tags(dim, cells)
            for tag in tags:
                if dim == cell_dim:
                    name = "Volume_%d" % tag
                else:
                    name = "Boundary_%d" % tag
                (pug, vert_ids) = self._create_unstructured_grid(dim, tag, xyz, cells)
                if pug is not None:
                    logging.info(
                        "%d vertices, %d cells"
                        % (pug.GetNumberOfPoints(), pug.GetNumberOfCells())
                    )
                    mbds.SetBlock(iz, pug.VTKObject)
                    mbds.GetMetaData(iz).Set(mbds.NAME(), name)
                    iz += 1
                    data = pug.GetPointData()
                    added = []
                    for name, arr in zip(*data):
                        if name not in added:
                            data.append(arr[vert_ids], name)

        return 1


def test(fname):
    from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter

    reader = PythonMeshbReader()
    reader.SetFileName(fname)
    reader.EnableAllVariables()
    reader.Update()

    grid = reader.GetOutputDataObject(0)
    nb = grid.GetNumberOfBlocks()
    if nb == 0:
        logging.error("No block created")
        quit()

    for i in range(nb):
        blk = grid.GetBlock(i)

        logging.info(
            "Block %d: %d points, %d cells"
            % (i, blk.GetNumberOfPoints(), blk.GetNumberOfCells())
        )
        if blk.GetNumberOfCells() == 0:
            logging.error("No cell in block %d" % i)
            quit()
        if blk.GetNumberOfPoints() == 0:
            logging.error("No point in block %d" % i)
            quit()

        data = blk.GetCellData()
        na = data.GetNumberOfArrays()
        for i in range(na):
            logging.info("Cell data %s" % data.GetArrayName(i))

    fname = fname.replace(".meshb", ".vtm")
    fname = fname.replace(".mesh", ".vtm")
    logging.info("Writing %s" % fname)
    writer = vtkXMLMultiBlockDataWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetDataModeToAscii()
    writer.SetCompressorTypeToNone()
    # writer.SetDataModeToAppended()
    # writer.EncodeAppendedDataOff()
    writer.Update()


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    test("../data/mesh3d.meshb")
