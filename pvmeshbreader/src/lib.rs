#![allow(clippy::missing_safety_doc)]
#![allow(non_camel_case_types)]
use minimeshb::reader::MeshbReader;
use std::ffi::{c_char, CStr};

pub struct minimeshb_reader_t {
    implem: MeshbReader,
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_reader_create(fname: *mut c_char) -> *mut minimeshb_reader_t {
    let fname = CStr::from_ptr(fname);
    println!("{fname:?}");
    MeshbReader::new(fname.to_str().unwrap()).map_or(std::ptr::null_mut(), |implem| {
        Box::into_raw(Box::new(minimeshb_reader_t { implem }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_reader_delete(reader: *mut minimeshb_reader_t) {
    let _ = Box::from_raw(reader);
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_dimension(reader: *mut minimeshb_reader_t) -> u8 {
    (*reader).implem.dimension()
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_num_verts(reader: *mut minimeshb_reader_t) -> usize {
    (*reader).implem.n_verts().unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_read_verts_2d(
    reader: *mut minimeshb_reader_t,
    verts: *mut f64,
    tags: *mut i32,
) {
    for (i, (v, t)) in (*reader).implem.read_vertices::<2>().unwrap().enumerate() {
        v.iter()
            .enumerate()
            .for_each(|(j, v)| *verts.add(2 * i + j) = *v);
        *tags.add(i) = t;
    }
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_read_verts_3d(
    reader: *mut minimeshb_reader_t,
    verts: *mut f64,
    tags: *mut i32,
) {
    for (i, (v, t)) in (*reader).implem.read_vertices::<3>().unwrap().enumerate() {
        v.iter()
            .enumerate()
            .for_each(|(j, v)| *verts.add(3 * i + j) = *v);
        *tags.add(i) = t;
    }
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_num_edges(reader: *mut minimeshb_reader_t) -> usize {
    (*reader).implem.n_elements("Edges").unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_read_edges(
    reader: *mut minimeshb_reader_t,
    conn: *mut u64,
    tags: *mut i32,
) {
    for (i, (v, t)) in (*reader).implem.read_edges().unwrap().enumerate() {
        v.iter()
            .enumerate()
            .for_each(|(j, v)| *conn.add(2 * i + j) = *v);
        *tags.add(i) = t;
    }
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_num_triangles(reader: *mut minimeshb_reader_t) -> usize {
    (*reader).implem.n_elements("Triangles").unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_read_triangles(
    reader: *mut minimeshb_reader_t,
    conn: *mut u64,
    tags: *mut i32,
) {
    for (i, (v, t)) in (*reader).implem.read_triangles().unwrap().enumerate() {
        v.iter()
            .enumerate()
            .for_each(|(j, v)| *conn.add(3 * i + j) = *v);
        *tags.add(i) = t;
    }
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_num_tetrahedra(reader: *mut minimeshb_reader_t) -> usize {
    (*reader).implem.n_elements("Tetrahedra").unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_read_tetrahedra(
    reader: *mut minimeshb_reader_t,
    conn: *mut u64,
    tags: *mut i32,
) {
    for (i, (v, t)) in (*reader).implem.read_tetrahedra().unwrap().enumerate() {
        v.iter()
            .enumerate()
            .for_each(|(j, v)| *conn.add(4 * i + j) = *v);
        *tags.add(i) = t;
    }
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_solution_size(
    reader: *mut minimeshb_reader_t,
    size: *mut usize,
) {
    let (n_verts, m) = (*reader).implem.get_solution_size().unwrap_or((0, 0));
    *size.add(0) = n_verts;
    *size.add(1) = m;
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_read_scalar_solution(
    reader: *mut minimeshb_reader_t,
    vals: *mut f64,
) {
    for (i, v) in (*reader).implem.read_solution::<1>().unwrap().enumerate() {
        *vals.add(i) = v[0];
    }
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_read_vector_solution_2d(
    reader: *mut minimeshb_reader_t,
    vals: *mut f64,
) {
    for (i, v) in (*reader).implem.read_solution::<2>().unwrap().enumerate() {
        v.iter()
            .enumerate()
            .for_each(|(j, v)| *vals.add(2 * i + j) = *v);
    }
}

#[no_mangle]
pub unsafe extern "C" fn minimeshb_read_vector_solution_3d(
    reader: *mut minimeshb_reader_t,
    vals: *mut f64,
) {
    for (i, v) in (*reader).implem.read_solution::<3>().unwrap().enumerate() {
        v.iter()
            .enumerate()
            .for_each(|(j, v)| *vals.add(3 * i + j) = *v);
    }
}
