use crate::libmeshb::field_order;
use crate::{Error, Result};
use libmeshb_sys::{
    GmfCloseMesh, GmfGetLin, GmfGotoKwd, GmfKwdCod, GmfOpenMesh, GmfRead, GmfSca, GmfStatKwd,
    GmfSymMat, GmfVec,
};
use log::debug;
use std::ffi::{c_int, CString};
use std::ptr::addr_of_mut;

use super::{GmfElementTypes, GmfFieldTypes};

/// Reader for .mesh(b) / .sol(b) files (interface to libMeshb)
pub struct GmfReader {
    file: i64,
    dim: c_int,
    version: c_int,
}

impl GmfReader {
    /// Create a new file
    #[must_use]
    pub fn new(fname: &str) -> Result<Self> {
        debug!("Open {} (read)", fname);
        let mut dim: c_int = 0;
        let mut version: c_int = 0;

        let cfname = CString::new(fname).unwrap();
        let file = unsafe {
            GmfOpenMesh(
                cfname.as_ptr(),
                GmfRead as c_int,
                addr_of_mut!(version),
                addr_of_mut!(dim),
            )
        };
        if file == 0 {
            return Err(Error::from(&format!("unable to open {fname}")));
        }
        assert!(version > 0, "Invalid version in {fname}");
        assert!(dim > 0, "Invalid dimension in {fname}");
        Ok(Self { file, dim, version })
    }

    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.file != 0
    }

    #[must_use]
    pub const fn is_invalid(&self) -> bool {
        self.file == 0
    }

    /// Get the dimension (# or components for the coordinates)
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim as usize
    }

    /// Read the vertices
    pub fn read_vertices<const D: usize>(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = ([f64; D], i32)> + '_> {
        assert!(self.is_valid());
        assert_eq!(D, self.dim.try_into().unwrap());

        let n_nodes = unsafe { GmfStatKwd(self.file, GmfKwdCod::GmfVertices as c_int) };
        debug!("Read {} vertices", n_nodes);

        unsafe { GmfGotoKwd(self.file, GmfKwdCod::GmfVertices as c_int) };

        let mut vals = [0.0; D];

        Ok((0..n_nodes as usize).map(move |_| {
            let mut tag: c_int = 0;
            if self.version == 1 {
                let mut x = 0_f32;
                let mut y = 0_f32;
                let mut z = 0_f32;
                if D == 2 {
                    unsafe {
                        GmfGetLin(
                            self.file,
                            GmfKwdCod::GmfVertices as c_int,
                            addr_of_mut!(x),
                            addr_of_mut!(y),
                            addr_of_mut!(tag),
                        )
                    };
                    vals[0] = x.into();
                    vals[1] = y.into();
                } else if D == 3 {
                    unsafe {
                        GmfGetLin(
                            self.file,
                            GmfKwdCod::GmfVertices as c_int,
                            addr_of_mut!(x),
                            addr_of_mut!(y),
                            addr_of_mut!(z),
                            addr_of_mut!(tag),
                        )
                    };
                    vals[0] = x.into();
                    vals[1] = y.into();
                    vals[2] = y.into();
                } else {
                    unreachable!()
                }
            } else {
                {
                    let mut x = 0_f64;
                    let mut y = 0_f64;
                    let mut z = 0_f64;
                    if D == 2 {
                        unsafe {
                            GmfGetLin(
                                self.file,
                                GmfKwdCod::GmfVertices as c_int,
                                addr_of_mut!(x),
                                addr_of_mut!(y),
                                addr_of_mut!(tag),
                            )
                        };
                        vals[0] = x;
                        vals[1] = y;
                    } else if D == 3 {
                        unsafe {
                            GmfGetLin(
                                self.file,
                                GmfKwdCod::GmfVertices as c_int,
                                addr_of_mut!(x),
                                addr_of_mut!(y),
                                addr_of_mut!(z),
                                addr_of_mut!(tag),
                            )
                        };
                        vals[0] = x;
                        vals[1] = y;
                        vals[2] = z;
                    } else {
                        unreachable!()
                    }
                }
            }
            (vals, tag as i32)
        }))
    }

    /// Read the element connectivity and element tags
    #[must_use]
    pub fn read_elements<const N: usize>(
        &self,
        etype: GmfElementTypes,
    ) -> Result<impl ExactSizeIterator<Item = ([usize; N], i32)> + '_> {
        assert!(self.is_valid());

        let m = match etype {
            GmfElementTypes::Vertex => 1,
            GmfElementTypes::Edge => 2,
            GmfElementTypes::Triangle => 3,
            GmfElementTypes::Tetrahedron => 4,
        };
        assert_eq!(N, m);

        let n_elems = unsafe { GmfStatKwd(self.file, etype as c_int) };
        debug!("Read {} elements ({:?})", n_elems, etype);

        unsafe { GmfGotoKwd(self.file, etype as c_int) };

        let mut vals = [0_usize; N];

        Ok((0..n_elems as usize).map(move |_| {
            let mut tag: c_int = 0;
            match N {
                1 => {
                    if self.version < 4 {
                        let mut i0: c_int = 0;
                        for _ in 0..n_elems {
                            unsafe {
                                GmfGetLin(
                                    self.file,
                                    etype as c_int,
                                    addr_of_mut!(i0),
                                    addr_of_mut!(tag),
                                )
                            };
                            vals[0] = i0.try_into().unwrap();
                        }
                    } else {
                        let mut i0 = 0_i64;
                        for _ in 0..n_elems {
                            unsafe {
                                GmfGetLin(
                                    self.file,
                                    etype as c_int,
                                    addr_of_mut!(i0),
                                    addr_of_mut!(tag),
                                )
                            };
                            vals[0] = i0.try_into().unwrap();
                        }
                    }
                }
                2 => {
                    if self.version < 4 {
                        let mut i0: c_int = 0;
                        let mut i1: c_int = 0;
                        unsafe {
                            GmfGetLin(
                                self.file,
                                etype as c_int,
                                addr_of_mut!(i0),
                                addr_of_mut!(i1),
                                addr_of_mut!(tag),
                            )
                        };
                        vals[0] = i0.try_into().unwrap();
                        vals[1] = i1.try_into().unwrap();
                    } else {
                        let mut i0 = 0_i64;
                        let mut i1 = 0_i64;
                        unsafe {
                            GmfGetLin(
                                self.file,
                                etype as c_int,
                                addr_of_mut!(i0),
                                addr_of_mut!(i1),
                                addr_of_mut!(tag),
                            )
                        };
                        vals[0] = i0.try_into().unwrap();
                        vals[1] = i1.try_into().unwrap();
                    }
                }
                3 => {
                    if self.version < 4 {
                        let mut i0: c_int = 0;
                        let mut i1: c_int = 0;
                        let mut i2: c_int = 0;
                        unsafe {
                            GmfGetLin(
                                self.file,
                                etype as c_int,
                                addr_of_mut!(i0),
                                addr_of_mut!(i1),
                                addr_of_mut!(i2),
                                addr_of_mut!(tag),
                            )
                        };
                        vals[0] = i0.try_into().unwrap();
                        vals[1] = i1.try_into().unwrap();
                        vals[2] = i2.try_into().unwrap();
                    } else {
                        let mut i0 = 0_i64;
                        let mut i1 = 0_i64;
                        let mut i2 = 0_i64;
                        unsafe {
                            GmfGetLin(
                                self.file,
                                etype as c_int,
                                addr_of_mut!(i0),
                                addr_of_mut!(i1),
                                addr_of_mut!(i2),
                                addr_of_mut!(tag),
                            )
                        };
                        vals[0] = i0.try_into().unwrap();
                        vals[1] = i1.try_into().unwrap();
                        vals[2] = i2.try_into().unwrap();
                    }
                }
                4 => {
                    if self.version < 4 {
                        let mut i0: c_int = 0;
                        let mut i1: c_int = 0;
                        let mut i2: c_int = 0;
                        let mut i3: c_int = 0;
                        unsafe {
                            GmfGetLin(
                                self.file,
                                etype as c_int,
                                addr_of_mut!(i0),
                                addr_of_mut!(i1),
                                addr_of_mut!(i2),
                                addr_of_mut!(i3),
                                addr_of_mut!(tag),
                            )
                        };
                        vals[0] = i0.try_into().unwrap();
                        vals[1] = i1.try_into().unwrap();
                        vals[2] = i2.try_into().unwrap();
                        vals[3] = i3.try_into().unwrap();
                    } else {
                        let mut i0 = 0_i64;
                        let mut i1 = 0_i64;
                        let mut i2 = 0_i64;
                        let mut i3 = 0_i64;
                        unsafe {
                            GmfGetLin(
                                self.file,
                                etype as c_int,
                                addr_of_mut!(i0),
                                addr_of_mut!(i1),
                                addr_of_mut!(i2),
                                addr_of_mut!(i3),
                                addr_of_mut!(tag),
                            )
                        };
                        vals[0] = i0.try_into().unwrap();
                        vals[1] = i1.try_into().unwrap();
                        vals[2] = i2.try_into().unwrap();
                        vals[3] = i3.try_into().unwrap();
                    }
                }
                _ => unimplemented!(),
            }
            for v in &mut vals {
                *v -= 1;
            }
            (vals, tag as i32)
        }))
    }

    pub fn read_edges(&mut self) -> Result<impl ExactSizeIterator<Item = ([usize; 2], i32)> + '_> {
        self.read_elements(GmfElementTypes::Edge)
    }

    pub fn read_triangles(
        &mut self,
    ) -> Result<impl ExactSizeIterator<Item = ([usize; 3], i32)> + '_> {
        self.read_elements(GmfElementTypes::Triangle)
    }

    pub fn read_tetrahedra(
        &mut self,
    ) -> Result<impl ExactSizeIterator<Item = ([usize; 4], i32)> + '_> {
        self.read_elements(GmfElementTypes::Tetrahedron)
    }

    /// Read the field defined at the vertices (for .sol(b) files)
    #[must_use]
    pub fn get_solution_size(&mut self) -> Result<usize> {
        let mut field_type: c_int = 0;
        let mut n_types: c_int = 0;
        let mut sol_size: c_int = 0;
        let _n_verts = unsafe {
            GmfStatKwd(
                self.file,
                GmfKwdCod::GmfSolAtVertices as c_int,
                addr_of_mut!(n_types),
                addr_of_mut!(sol_size),
                addr_of_mut!(field_type),
            )
        };
        assert_eq!(n_types, 1);

        let n_comp = match field_type as u32 {
            x if x == GmfSca => 1,
            x if x == GmfVec => self.dim,
            x if x == GmfSymMat => (self.dim * (self.dim + 1)) / 2,
            _ => unreachable!("Field type {field_type} unknown: {GmfSca} {GmfVec} {GmfSymMat}"),
        };

        Ok(n_comp.try_into().unwrap())
    }

    pub fn read_solution<const N: usize>(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = [f64; N]> + '_> {
        let mut field_type: c_int = 0;
        let mut n_types: c_int = 0;
        let mut sol_size: c_int = 0;
        let n_verts = unsafe {
            GmfStatKwd(
                self.file,
                GmfKwdCod::GmfSolAtVertices as c_int,
                addr_of_mut!(n_types),
                addr_of_mut!(sol_size),
                addr_of_mut!(field_type),
            )
        };
        assert_eq!(n_types, 1);

        let (field_type, n_comp) = match field_type as u32 {
            x if x == GmfSca => (GmfFieldTypes::Scalar, 1),
            x if x == GmfVec => (GmfFieldTypes::Vector, self.dim),
            x if x == GmfSymMat => (GmfFieldTypes::Metric, (self.dim * (self.dim + 1)) / 2),
            _ => unreachable!("Field type {field_type} unknown: {GmfSca} {GmfVec} {GmfSymMat}"),
        };
        assert_eq!(sol_size, n_comp as c_int);

        debug!("Read {}x{} values", n_verts, n_comp);

        let mut val = [0.0; N];

        unsafe { GmfGotoKwd(self.file, GmfKwdCod::GmfSolAtVertices as c_int) };

        let order = field_order(self.dim as usize, field_type);

        Ok((0..n_verts as usize).map(move |_| {
            if self.version == 1 {
                let mut s = [0_f32; N];
                unsafe {
                    GmfGetLin(
                        self.file,
                        GmfKwdCod::GmfSolAtVertices as c_int,
                        addr_of_mut!(s),
                    );
                }
                for (i, j) in order.iter().copied().enumerate() {
                    val[i] = s[j].try_into().unwrap();
                }
            } else {
                let mut s = [0_f64; N];
                unsafe {
                    GmfGetLin(
                        self.file,
                        GmfKwdCod::GmfSolAtVertices as c_int,
                        addr_of_mut!(s),
                    );
                }
                for (i, j) in order.iter().copied().enumerate() {
                    val[i] = s[j];
                }
            }
            val
        }))
    }

    fn close(&mut self) {
        if self.is_valid() {
            unsafe {
                GmfCloseMesh(self.file);
            }
            self.file = 0;
        }
    }
}

impl Drop for GmfReader {
    fn drop(&mut self) {
        self.close()
    }
}
