use log::debug;

use crate::{Element, Error, Result, Solution, Tag, Vertex};
use std::{
    fs::File,
    io::{BufWriter, Seek, Write},
};

#[derive(Debug)]
pub struct MeshbWriter {
    is_binary: bool,
    version: u8,
    dimension: u8,
    writer: BufWriter<File>,
}

impl MeshbWriter {
    pub fn new(fname: &str, version: u8, dimension: u8) -> Result<Self> {
        if !fname.ends_with(".mesh")
            && !fname.ends_with(".meshb")
            && !fname.ends_with(".sol")
            && !fname.ends_with(".solb")
        {
            return Err(Error::from(&format!("Invalid file extension for {fname}")));
        }

        if fname.ends_with(".meshb") || fname.ends_with(".solb") {
            Self::new_binary(fname, version, dimension)
        } else {
            Self::new_ascii(fname, version, dimension)
        }
    }

    fn new_ascii(fname: &str, version: u8, dimension: u8) -> Result<Self> {
        debug!("create {fname} (version = {version}, dimension = {dimension}, ascii)");

        let file = File::create(fname)?;

        let mut res = Self {
            is_binary: false,
            version,
            dimension,
            writer: BufWriter::new(file),
        };

        writeln!(res.writer, "MeshVersionFormatted 1")?;
        writeln!(res.writer, "Dimension {dimension}")?;

        Ok(res)
    }

    fn write_kwd(&mut self, kwd: i32) {
        self.writer.write_all(&kwd.to_le_bytes()).unwrap();
    }

    fn write_float(&mut self, v: f64) {
        if self.version == 1 {
            self.writer.write_all(&(v as f32).to_le_bytes()).unwrap();
        } else {
            self.writer.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    fn write_index(&mut self, v: u64) {
        if self.version < 4 {
            self.writer.write_all(&(v as u32).to_le_bytes()).unwrap();
        } else {
            self.writer.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    fn new_binary(fname: &str, version: u8, dimension: u8) -> Result<Self> {
        debug!("create {fname} (version = {version}, dimension = {dimension}, binary)");

        let file = File::create(fname)?;

        let mut res = Self {
            is_binary: true,
            version,
            dimension,
            writer: BufWriter::new(file),
        };

        res.write_kwd(1);
        res.write_kwd(version as i32);

        res.write_kwd(3); // Dimension
        let mut next = res.writer.stream_position().unwrap();
        next += if res.version >= 3 {
            std::mem::size_of::<u64>() as u64
        } else {
            std::mem::size_of::<u32>() as u64
        }; // Next
        next += std::mem::size_of::<i32>() as u64; // Dimension
        res.write_index(next);
        res.write_kwd(dimension as i32);
        Ok(res)
    }

    pub fn close(mut self) {
        if self.is_binary {
            self.write_kwd(54);
            self.write_index(0);
        } else {
            writeln!(self.writer, "End").unwrap();
        }
    }

    pub fn write_vertices<V: Vertex, T: Tag>(&mut self, verts: &[V], tags: &[T]) -> Result<()> {
        assert_eq!(verts.len(), tags.len());
        debug!("write {} vertices", verts.len());

        if self.is_binary {
            self.write_vertices_binary(verts, tags)
        } else {
            self.write_vertices_ascii(verts, tags)
        }
    }

    fn write_vertices_ascii<V: Vertex, T: Tag>(&mut self, verts: &[V], tags: &[T]) -> Result<()> {
        writeln!(self.writer, "Vertices")?;
        writeln!(self.writer, "{}", verts.len())?;

        let mut line = String::new();

        for (v, t) in verts.iter().zip(tags.iter()) {
            line.clear();
            for i in 0..self.dimension as usize {
                line += &format!("{} ", v.get(i));
            }
            line += &format!("{}", t.get());
            writeln!(self.writer, "{}", &line)?;
        }

        Ok(())
    }

    fn write_vertices_binary<V: Vertex, T: Tag>(&mut self, verts: &[V], tags: &[T]) -> Result<()> {
        let mut next = self.writer.stream_position().unwrap();
        next += std::mem::size_of::<i32>() as u64; // Keyword
        next += 2 * if self.version >= 3 {
            std::mem::size_of::<u64>() as u64
        } else {
            std::mem::size_of::<u32>() as u64
        }; // Next + Size
        next += self.dimension as u64
            * verts.len() as u64
            * if self.version == 1 {
                std::mem::size_of::<f32>() as u64
            } else {
                std::mem::size_of::<f64>() as u64
            }; // Coordinates
        next += verts.len() as u64 * std::mem::size_of::<i32>() as u64; // Tags

        self.write_kwd(4);
        self.write_index(next);
        self.write_index(verts.len() as u64);

        for (v, t) in verts.iter().zip(tags.iter()) {
            for i in 0..self.dimension as usize {
                self.write_float(v.get(i));
            }
            self.write_kwd(t.get());
        }

        Ok(())
    }

    fn write_elements<E: Element, T: Tag>(
        &mut self,
        kwd: &str,
        elems: &[E],
        tags: &[T],
    ) -> Result<()> {
        assert_eq!(elems.len(), tags.len());
        debug!("write {} elements", elems.len());

        if self.is_binary {
            self.write_elements_binary(kwd, elems, tags)
        } else {
            self.write_elements_ascii(kwd, elems, tags)
        }
    }

    fn write_elements_ascii<E: Element, T: Tag>(
        &mut self,
        kwd: &str,
        elems: &[E],
        tags: &[T],
    ) -> Result<()> {
        let m = match kwd {
            "Edges" => 2,
            "Triangles" => 3,
            "Tetrahedra" => 4,
            _ => unreachable!(),
        };

        writeln!(self.writer, "{}", kwd)?;
        writeln!(self.writer, "{}", elems.len())?;

        let mut line = String::new();

        for (v, t) in elems.iter().zip(tags.iter()) {
            line.clear();
            for i in 0..m {
                line += &format!("{} ", v.get(i) + 1);
            }
            line += &format!("{}", t.get());
            writeln!(self.writer, "{}", &line)?;
        }

        Ok(())
    }

    fn write_elements_binary<E: Element, T: Tag>(
        &mut self,
        kwd: &str,
        elems: &[E],
        tags: &[T],
    ) -> Result<()> {
        let (m, kwd) = match kwd {
            "Edges" => (2, 5),
            "Triangles" => (3, 6),
            "Tetrahedra" => (4, 8),
            _ => unreachable!(),
        };

        let mut next = self.writer.stream_position().unwrap();
        next += std::mem::size_of::<i32>() as u64; // Keyword
        next += 2 * if self.version >= 3 {
            std::mem::size_of::<u64>() as u64
        } else {
            std::mem::size_of::<u32>() as u64
        }; // Next + Size
        next += m
            * elems.len() as u64
            * if self.version >= 3 {
                std::mem::size_of::<u64>() as u64
            } else {
                std::mem::size_of::<u32>() as u64
            }; // Coordinates
        next += elems.len() as u64 * std::mem::size_of::<i32>() as u64; // Tags

        self.write_kwd(kwd);
        self.write_index(next);
        self.write_index(elems.len() as u64);

        for (v, t) in elems.iter().zip(tags.iter()) {
            for i in 0..m {
                self.write_index(v.get(i as usize) + 1);
            }
            self.write_kwd(t.get());
        }

        Ok(())
    }

    pub fn write_edges<E: Element, T: Tag>(&mut self, elems: &[E], tags: &[T]) -> Result<()> {
        self.write_elements("Edges", elems, tags)
    }

    pub fn write_triangles<E: Element, T: Tag>(&mut self, elems: &[E], tags: &[T]) -> Result<()> {
        self.write_elements("Triangles", elems, tags)
    }

    pub fn write_tetrahedra<E: Element, T: Tag>(&mut self, elems: &[E], tags: &[T]) -> Result<()> {
        self.write_elements("Tetrahedra", elems, tags)
    }

    pub fn write_solution<S: Solution>(&mut self, sols: &[S]) -> Result<()> {
        debug!("write field");

        if self.is_binary {
            self.write_solution_binary(sols)
        } else {
            self.write_solution_ascii(sols)
        }
    }

    fn get_solution_type<S: Solution>(&self) -> Result<u8> {
        if S::m() == 1 {
            Ok(1)
        } else if S::m() == self.dimension as usize {
            Ok(2)
        } else if S::m() == (self.dimension * (self.dimension + 1) / 2) as usize {
            Ok(3)
        } else {
            Err(Error::from(&format!("Unvalid field size {}", S::m())))
        }
    }

    fn write_solution_ascii<S: Solution>(&mut self, sols: &[S]) -> Result<()> {
        writeln!(self.writer, "SolAtVertices")?;
        writeln!(self.writer, "{}", sols.len())?;
        writeln!(self.writer, "1 {}", self.get_solution_type::<S>()?)?;

        let mut line = String::new();

        for s in sols.iter() {
            line.clear();
            for i in 0..S::m() {
                line += &format!("{} ", s.get(i));
            }
            writeln!(self.writer, "{}", &line)?;
        }

        Ok(())
    }

    fn write_solution_binary<S: Solution>(&mut self, sols: &[S]) -> Result<()> {
        let mut next = self.writer.stream_position().unwrap();
        next += std::mem::size_of::<i32>() as u64; // Keyword
        next += 2 * if self.version >= 3 {
            std::mem::size_of::<u64>() as u64
        } else {
            std::mem::size_of::<u32>() as u64
        }; // Next + Size
        next += 2 * std::mem::size_of::<u32>() as u64; // field type
        next += S::m() as u64
            * sols.len() as u64
            * if self.version == 1 {
                std::mem::size_of::<f32>() as u64
            } else {
                std::mem::size_of::<f64>() as u64
            }; // Values

        self.write_kwd(62);
        self.write_index(next);
        self.write_index(sols.len() as u64);
        self.write_kwd(1);
        self.write_kwd(self.get_solution_type::<S>()? as i32);

        for s in sols.iter() {
            for i in 0..S::m() {
                self.write_float(s.get(i));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MeshbWriter;
    use crate::reader::MeshbReader;
    use tempfile::NamedTempFile;

    #[test]
    fn test_write_ascii_3d() {
        let mut reader = MeshbReader::new("./data/mesh3d.mesh").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".mesh";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let (verts, vtags) = reader.read_vertices::<[f64; 3], ()>().unwrap();
        writer.write_vertices(&verts, &vtags).unwrap();

        let (tris, tritags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        writer.write_triangles(&tris, &tritags).unwrap();

        let (tets, tettags) = reader.read_tetrahedra::<[u32; 4], i16>().unwrap();
        writer.write_tetrahedra(&tets, &tettags).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let (verts2, vtags2) = reader2.read_vertices::<[f64; 3], ()>().unwrap();
        let (tris2, tritags2) = reader2.read_triangles::<[u32; 3], i16>().unwrap();
        let (tets2, tettags2) = reader2.read_tetrahedra::<[u32; 4], i16>().unwrap();

        for (vert0, vert1) in verts.iter().zip(verts2.iter()) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        for (tag0, tag1) in vtags.iter().zip(vtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tri0, tri1) in tris.iter().zip(tris2.iter()) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tritags.iter().zip(tritags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tet0, tet1) in tets.iter().zip(tets2.iter()) {
            for (v0, v1) in tet0.iter().zip(tet1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tettags.iter().zip(tettags2.iter()) {
            assert_eq!(tag0, tag1);
        }
    }

    #[test]
    fn test_write_binary_3d() {
        let mut reader = MeshbReader::new("./data/mesh3d.mesh").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".meshb";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let (verts, vtags) = reader.read_vertices::<[f64; 3], ()>().unwrap();
        writer.write_vertices(&verts, &vtags).unwrap();

        let (tris, tritags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        writer.write_triangles(&tris, &tritags).unwrap();

        let (tets, tettags) = reader.read_tetrahedra::<[u32; 4], i16>().unwrap();
        writer.write_tetrahedra(&tets, &tettags).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let (verts2, vtags2) = reader2.read_vertices::<[f64; 3], ()>().unwrap();
        let (tris2, tritags2) = reader2.read_triangles::<[u32; 3], i16>().unwrap();
        let (tets2, tettags2) = reader2.read_tetrahedra::<[u32; 4], i16>().unwrap();

        for (vert0, vert1) in verts.iter().zip(verts2.iter()) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        for (tag0, tag1) in vtags.iter().zip(vtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tri0, tri1) in tris.iter().zip(tris2.iter()) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tritags.iter().zip(tritags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tet0, tet1) in tets.iter().zip(tets2.iter()) {
            for (v0, v1) in tet0.iter().zip(tet1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tettags.iter().zip(tettags2.iter()) {
            assert_eq!(tag0, tag1);
        }
    }

    #[test]
    fn test_write_binary_3d_2() {
        let mut reader = MeshbReader::new("./data/mesh3d.mesh").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".meshb";

        let mut writer = MeshbWriter::new(&fname, 4, 3).unwrap();

        let (verts, vtags) = reader.read_vertices::<[f64; 3], ()>().unwrap();
        writer.write_vertices(&verts, &vtags).unwrap();

        let (tris, tritags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        writer.write_triangles(&tris, &tritags).unwrap();

        let (tets, tettags) = reader.read_tetrahedra::<[u32; 4], i16>().unwrap();
        writer.write_tetrahedra(&tets, &tettags).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        assert_eq!(reader2.version(), 4);

        let (verts2, vtags2) = reader2.read_vertices::<[f64; 3], ()>().unwrap();
        let (tris2, tritags2) = reader2.read_triangles::<[u32; 3], i16>().unwrap();
        let (tets2, tettags2) = reader2.read_tetrahedra::<[u32; 4], i16>().unwrap();

        for (vert0, vert1) in verts.iter().zip(verts2.iter()) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        for (tag0, tag1) in vtags.iter().zip(vtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tri0, tri1) in tris.iter().zip(tris2.iter()) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tritags.iter().zip(tritags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tet0, tet1) in tets.iter().zip(tets2.iter()) {
            for (v0, v1) in tet0.iter().zip(tet1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tettags.iter().zip(tettags2.iter()) {
            assert_eq!(tag0, tag1);
        }
    }

    #[test]
    fn test_write_ascii_sol_3d() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".sol";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let sols = reader.read_solution::<f32>().unwrap();
        writer.write_solution(&sols).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let sols2 = reader2.read_solution::<f32>().unwrap();

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            assert!((sol0 - sol1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_ascii_sol_3d_vec() {
        let mut reader = MeshbReader::new("./data/sol3d_vec.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".sol";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let sols = reader.read_solution::<[f32; 3]>().unwrap();
        writer.write_solution(&sols).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let sols2 = reader2.read_solution::<[f32; 3]>().unwrap();

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            for (s0, s1) in sol0.iter().zip(sol1.iter()) {
                assert!((s0 - s1).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_write_binary_sol_3d() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".solb";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let sols = reader.read_solution::<f32>().unwrap();
        writer.write_solution(&sols).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let sols2 = reader2.read_solution::<f32>().unwrap();

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            assert!((sol0 - sol1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_binary_sol_3d_vec() {
        let mut reader = MeshbReader::new("./data/sol3d_vec.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".solb";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let sols = reader.read_solution::<[f32; 3]>().unwrap();
        writer.write_solution(&sols).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let sols2 = reader2.read_solution::<[f32; 3]>().unwrap();

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            for (s0, s1) in sol0.iter().zip(sol1.iter()) {
                assert!((s0 - s1).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_write_ascii_2d() {
        let mut reader = MeshbReader::new("./data/mesh2d.mesh").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".mesh";

        let mut writer = MeshbWriter::new(&fname, 1, 2).unwrap();

        let (verts, vtags) = reader.read_vertices::<[f64; 2], ()>().unwrap();
        writer.write_vertices(&verts, &vtags).unwrap();

        let (edgs, edgtags) = reader.read_edges::<[u32; 2], i16>().unwrap();
        writer.write_edges(&edgs, &edgtags).unwrap();

        let (tris, tritags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        writer.write_triangles(&tris, &tritags).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let (verts2, vtags2) = reader2.read_vertices::<[f64; 2], ()>().unwrap();
        let (edgs2, edgtags2) = reader2.read_edges::<[u32; 2], i16>().unwrap();
        let (tris2, tritags2) = reader2.read_triangles::<[u32; 3], i16>().unwrap();

        for (vert0, vert1) in verts.iter().zip(verts2.iter()) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        for (tag0, tag1) in vtags.iter().zip(vtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (edg0, edg1) in edgs.iter().zip(edgs2.iter()) {
            for (v0, v1) in edg0.iter().zip(edg1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in edgtags.iter().zip(edgtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tri0, tri1) in tris.iter().zip(tris2.iter()) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tritags.iter().zip(tritags2.iter()) {
            assert_eq!(tag0, tag1);
        }
    }

    #[test]
    fn test_write_binary_2d() {
        let mut reader = MeshbReader::new("./data/mesh2d.mesh").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".meshb";

        let mut writer = MeshbWriter::new(&fname, 1, 2).unwrap();

        let (verts, vtags) = reader.read_vertices::<[f64; 2], ()>().unwrap();
        writer.write_vertices(&verts, &vtags).unwrap();

        let (edgs, edgtags) = reader.read_edges::<[u32; 2], i16>().unwrap();
        writer.write_edges(&edgs, &edgtags).unwrap();

        let (tris, tritags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        writer.write_triangles(&tris, &tritags).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let (verts2, vtags2) = reader2.read_vertices::<[f64; 2], ()>().unwrap();
        let (edgs2, edgtags2) = reader2.read_edges::<[u32; 2], i16>().unwrap();
        let (tris2, tritags2) = reader2.read_triangles::<[u32; 3], i16>().unwrap();

        for (vert0, vert1) in verts.iter().zip(verts2.iter()) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        for (tag0, tag1) in vtags.iter().zip(vtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (edg0, edg1) in edgs.iter().zip(edgs2.iter()) {
            for (v0, v1) in edg0.iter().zip(edg1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in edgtags.iter().zip(edgtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tri0, tri1) in tris.iter().zip(tris2.iter()) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tritags.iter().zip(tritags2.iter()) {
            assert_eq!(tag0, tag1);
        }
    }

    #[test]
    fn test_write_binary_2d_2() {
        let mut reader = MeshbReader::new("./data/mesh2d.mesh").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".meshb";

        let mut writer = MeshbWriter::new(&fname, 4, 2).unwrap();

        let (verts, vtags) = reader.read_vertices::<[f64; 2], ()>().unwrap();
        writer.write_vertices(&verts, &vtags).unwrap();

        let (edgs, edgtags) = reader.read_edges::<[u32; 2], i16>().unwrap();
        writer.write_edges(&edgs, &edgtags).unwrap();

        let (tris, tritags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        writer.write_triangles(&tris, &tritags).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let (verts2, vtags2) = reader2.read_vertices::<[f64; 2], ()>().unwrap();
        let (edgs2, edgtags2) = reader2.read_edges::<[u32; 2], i16>().unwrap();
        let (tris2, tritags2) = reader2.read_triangles::<[u32; 3], i16>().unwrap();

        for (vert0, vert1) in verts.iter().zip(verts2.iter()) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        for (tag0, tag1) in vtags.iter().zip(vtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (edg0, edg1) in edgs.iter().zip(edgs2.iter()) {
            for (v0, v1) in edg0.iter().zip(edg1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in edgtags.iter().zip(edgtags2.iter()) {
            assert_eq!(tag0, tag1);
        }

        for (tri0, tri1) in tris.iter().zip(tris2.iter()) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in tritags.iter().zip(tritags2.iter()) {
            assert_eq!(tag0, tag1);
        }
    }

    #[test]
    fn test_write_ascii_sol_2d() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".sol";

        let mut writer = MeshbWriter::new(&fname, 1, 2).unwrap();

        let sols = reader.read_solution::<f32>().unwrap();
        writer.write_solution(&sols).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let sols2 = reader2.read_solution::<f32>().unwrap();

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            assert!((sol0 - sol1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_ascii_sol_2d_vec() {
        let mut reader = MeshbReader::new("./data/sol2d_vec.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".sol";

        let mut writer = MeshbWriter::new(&fname, 1, 2).unwrap();

        let sols = reader.read_solution::<[f32; 2]>().unwrap();
        writer.write_solution(&sols).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let sols2 = reader2.read_solution::<[f32; 2]>().unwrap();

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            for (s0, s1) in sol0.iter().zip(sol1.iter()) {
                assert!((s0 - s1).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_write_binary_sol_2d() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".solb";

        let mut writer = MeshbWriter::new(&fname, 1, 2).unwrap();

        let sols = reader.read_solution::<f32>().unwrap();
        writer.write_solution(&sols).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let sols2 = reader2.read_solution::<f32>().unwrap();

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            assert!((sol0 - sol1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_binary_sol_2d_vec() {
        let mut reader = MeshbReader::new("./data/sol2d_vec.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".solb";

        let mut writer = MeshbWriter::new(&fname, 1, 2).unwrap();

        let sols = reader.read_solution::<[f32; 2]>().unwrap();
        writer.write_solution(&sols).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let sols2 = reader2.read_solution::<[f32; 2]>().unwrap();

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            for (s0, s1) in sol0.iter().zip(sol1.iter()) {
                assert!((s0 - s1).abs() < 1e-6);
            }
        }
    }
}
