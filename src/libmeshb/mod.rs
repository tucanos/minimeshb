use libmeshb_sys::{GmfKwdCod, GmfSca, GmfSymMat, GmfVec};

mod reader;
mod writer;

pub use reader::GmfReader;
pub use writer::GmfWriter;

#[derive(Clone, Copy, Debug)]
pub enum GmfElementTypes {
    Vertex = GmfKwdCod::GmfVertices as isize,
    Edge = GmfKwdCod::GmfEdges as isize,
    Triangle = GmfKwdCod::GmfTriangles as isize,
    Tetrahedron = GmfKwdCod::GmfTetrahedra as isize,
}

#[derive(Clone, Copy)]
enum GmfFieldTypes {
    Scalar = GmfSca as isize,
    Vector = GmfVec as isize,
    Metric = GmfSymMat as isize,
}

/// Reorder the entries (actually used only for symmetric tensors) to ensure consistency between
/// with the conventions used the meshb format
fn field_order(dim: usize, field_type: GmfFieldTypes) -> Vec<usize> {
    match dim {
        2 => match field_type {
            GmfFieldTypes::Scalar => vec![0],
            GmfFieldTypes::Vector => vec![0, 1],
            GmfFieldTypes::Metric => vec![0, 2, 1],
        },
        3 => match field_type {
            GmfFieldTypes::Scalar => vec![0],
            GmfFieldTypes::Vector => vec![0, 1, 2],
            GmfFieldTypes::Metric => vec![0, 2, 5, 1, 4, 3],
        },
        _ => unreachable!(),
    }
}
