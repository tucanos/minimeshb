use super::{GmfElementTypes, GmfFieldTypes};
use crate::Result;
use crate::{Error, libmeshb::field_order};
use libmeshb_sys::{
    GmfCloseMesh, GmfKwdCod, GmfOpenMesh, GmfSca, GmfSetKwd, GmfSetLin, GmfSymMat, GmfVec, GmfWrite,
};
use log::debug;
use std::ffi::{CString, c_int};

/// Writer for .mesh(b) / .sol(b) files (interface to libMeshb)
/// file version 2 (int32 and float64) is used
pub struct GmfWriter {
    file: i64,
    dim: u8,
}

impl GmfWriter {
    /// Create a new file
    pub fn new(fname: &str, version: u8, dim: u8) -> Result<Self> {
        debug!("Open {fname} (write)");

        let cfname = CString::new(fname).unwrap();
        let file = unsafe {
            GmfOpenMesh(
                cfname.as_ptr(),
                GmfWrite as c_int,
                c_int::from(version),
                c_int::from(dim),
            )
        };
        if file == 0 {
            return Err(Error::from(&format!("unable to open {fname}")));
        }
        Ok(Self { file, dim })
    }

    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.file != 0
    }

    #[must_use]
    pub const fn is_invalid(&self) -> bool {
        self.file == 0
    }

    pub fn write_vertices<
        const D: usize,
        I1: ExactSizeIterator<Item = [f64; D]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        verts: I1,
        tags: I2,
    ) -> Result<()> {
        assert_eq!(verts.len(), tags.len());
        debug!("Write {} vertices", verts.len());

        unsafe {
            GmfSetKwd(
                self.file,
                GmfKwdCod::GmfVertices as c_int,
                verts.len().try_into().unwrap(),
            );
        }

        for (p, t) in verts.zip(tags) {
            if D == 2 {
                unsafe {
                    GmfSetLin(
                        self.file,
                        GmfKwdCod::GmfVertices as c_int,
                        p[0],
                        p[1],
                        t as c_int,
                    );
                }
            } else if D == 3 {
                unsafe {
                    GmfSetLin(
                        self.file,
                        GmfKwdCod::GmfVertices as c_int,
                        p[0],
                        p[1],
                        p[2],
                        t as c_int,
                    );
                }
            } else {
                unreachable!();
            }
        }

        Ok(())
    }

    fn write_elements<
        const N: usize,
        I1: ExactSizeIterator<Item = [usize; N]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &self,
        etype: GmfElementTypes,
        elems: I1,
        etags: I2,
    ) -> Result<()> {
        assert_eq!(elems.len(), etags.len());
        debug!("Write {} elements ({:?})", elems.len(), etype);

        unsafe {
            GmfSetKwd(self.file, etype as c_int, elems.len().try_into().unwrap());
        }

        for (e, t) in elems.zip(etags) {
            match etype {
                GmfElementTypes::Vertex => unsafe {
                    GmfSetLin(self.file, etype as c_int, e[0] + 1, t as c_int);
                },
                GmfElementTypes::Edge => unsafe {
                    GmfSetLin(self.file, etype as c_int, e[0] + 1, e[1] + 1, t as c_int);
                },
                GmfElementTypes::Triangle => unsafe {
                    GmfSetLin(
                        self.file,
                        etype as c_int,
                        e[0] + 1,
                        e[1] + 1,
                        e[2] + 1,
                        t as c_int,
                    );
                },
                GmfElementTypes::Tetrahedron => unsafe {
                    GmfSetLin(
                        self.file,
                        etype as c_int,
                        e[0] + 1,
                        e[1] + 1,
                        e[2] + 1,
                        e[3] + 1,
                        i64::from(t),
                    );
                },
            };
        }

        Ok(())
    }

    pub fn write_edges<
        I1: ExactSizeIterator<Item = [usize; 2]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        self.write_elements(GmfElementTypes::Edge, elems, tags)
    }

    pub fn write_triangles<
        I1: ExactSizeIterator<Item = [usize; 3]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        self.write_elements(GmfElementTypes::Triangle, elems, tags)
    }

    pub fn write_tetrahedra<
        I1: ExactSizeIterator<Item = [usize; 4]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        self.write_elements(GmfElementTypes::Tetrahedron, elems, tags)
    }

    /// Write a solution defined at the vertices (.sol(b) file)
    pub fn write_solution<const N: usize, I: ExactSizeIterator<Item = [f64; N]>>(
        &mut self,
        sols: I,
    ) -> Result<()> {
        let n_verts = sols.len();
        debug!("Write {n_verts}x{N} values");

        let field_type = match N {
            1 => GmfFieldTypes::Scalar,
            x if x == self.dim as usize => GmfFieldTypes::Vector,
            x if x == (self.dim * (self.dim + 1)) as usize / 2 => GmfFieldTypes::Metric,
            _ => unreachable!(),
        };

        unsafe {
            let val = match field_type {
                GmfFieldTypes::Scalar => GmfSca,
                GmfFieldTypes::Vector => GmfVec,
                GmfFieldTypes::Metric => GmfSymMat,
            } as c_int;

            GmfSetKwd(
                self.file,
                GmfKwdCod::GmfSolAtVertices as c_int,
                n_verts as i64,
                1,
                &val,
            );
        }

        let order = field_order(self.dim as usize, field_type);

        let mut vals = [0.0; N];
        for s in sols {
            for (i, j) in order.iter().copied().enumerate() {
                vals[j] = s[i];
            }

            unsafe {
                GmfSetLin(self.file, GmfKwdCod::GmfSolAtVertices as c_int, &vals);
            }
        }

        Ok(())
    }

    pub fn close(&mut self) {
        if self.is_valid() {
            unsafe {
                GmfCloseMesh(self.file);
            }
            self.file = 0;
        }
    }
}

impl Drop for GmfWriter {
    fn drop(&mut self) {
        self.close();
    }
}
