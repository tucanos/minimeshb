use log::debug;

use crate::{Error, Result};
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

        writeln!(res.writer, "MeshVersionFormatted {version}")?;
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

    fn write_pos(&mut self, v: i64) {
        if self.version >= 3 {
            self.writer.write_all(&v.to_le_bytes()).unwrap();
        } else {
            self.writer.write_all(&(v as i32).to_le_bytes()).unwrap();
        }
    }

    fn write_index(&mut self, v: i64) {
        if self.version == 4 {
            self.writer.write_all(&v.to_le_bytes()).unwrap();
        } else {
            self.writer.write_all(&(v as i32).to_le_bytes()).unwrap();
        }
    }

    const fn size_of_float(&self) -> u64 {
        if self.version == 1 {
            size_of::<f32>() as u64
        } else {
            size_of::<f64>() as u64
        }
    }

    const fn size_of_pos(&self) -> u64 {
        if self.version >= 3 {
            size_of::<i64>() as u64
        } else {
            size_of::<i32>() as u64
        }
    }

    const fn size_of_index(&self) -> u64 {
        if self.version == 4 {
            size_of::<u64>() as u64
        } else {
            size_of::<i32>() as u64
        }
    }

    const fn size_of_kwd(&self) -> u64 {
        size_of::<i32>() as u64
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
        res.write_kwd(i32::from(version));

        res.write_kwd(3); // Dimension
        let mut next = res.writer.stream_position().unwrap();
        next += res.size_of_pos(); // Next
        next += res.size_of_kwd(); // Dimension
        res.write_pos(next as i64);
        res.write_kwd(i32::from(dimension));
        Ok(res)
    }

    pub fn close(mut self) {
        if self.is_binary {
            self.write_kwd(54);
            self.write_pos(0);
        } else {
            writeln!(self.writer, "End").unwrap();
        }
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
        debug!("write {} vertices", verts.len());

        if self.is_binary {
            self.write_vertices_binary(verts, tags)
        } else {
            self.write_vertices_ascii(verts, tags)
        }
    }

    fn write_vertices_ascii<
        const D: usize,
        I1: ExactSizeIterator<Item = [f64; D]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        verts: I1,
        tags: I2,
    ) -> Result<()> {
        writeln!(self.writer, "Vertices")?;
        writeln!(self.writer, "{}", verts.len())?;

        let mut line = String::new();

        for (v, t) in verts.zip(tags) {
            line.clear();
            for x in v {
                line += &format!("{x} ");
            }
            line += &format!("{t}");
            writeln!(self.writer, "{}", &line)?;
        }

        Ok(())
    }

    fn write_vertices_binary<
        const D: usize,
        I1: ExactSizeIterator<Item = [f64; D]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        verts: I1,
        tags: I2,
    ) -> Result<()> {
        let mut next = self.writer.stream_position().unwrap();
        next += self.size_of_kwd(); // Keyword
        next += self.size_of_pos();
        next += self.size_of_index();
        next += u64::from(self.dimension) * verts.len() as u64 * self.size_of_float(); // Coordinates
        next += verts.len() as u64 * self.size_of_index(); // Tags

        self.write_kwd(4);
        self.write_pos(next as i64);
        self.write_index(verts.len() as i64);

        for (v, t) in verts.zip(tags) {
            for x in v {
                self.write_float(x);
            }
            self.write_index(i64::from(t));
        }

        Ok(())
    }

    fn write_elements<
        const N: usize,
        I1: ExactSizeIterator<Item = [usize; N]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        kwd: &str,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        assert_eq!(elems.len(), tags.len());
        debug!("write {} elements", elems.len());

        if self.is_binary {
            self.write_elements_binary(kwd, elems, tags)
        } else {
            self.write_elements_ascii(kwd, elems, tags)
        }
    }

    fn write_elements_ascii<
        const N: usize,
        I1: ExactSizeIterator<Item = [usize; N]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        kwd: &str,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        let m = match kwd {
            "Edges" => 2,
            "EdgesP2" => 3,
            "Triangles" => 3,
            "TrianglesP2" => 6,
            "Tetrahedra" => 4,
            "TetrahedraP2" => 10,
            _ => unreachable!(),
        };
        assert_eq!(N, m);

        writeln!(self.writer, "{kwd}")?;
        writeln!(self.writer, "{}", elems.len())?;

        let mut line = String::new();

        for (v, t) in elems.zip(tags) {
            line.clear();
            for x in v {
                line += &format!("{} ", x + 1);
            }
            line += &format!("{t}");
            writeln!(self.writer, "{}", &line)?;
        }

        Ok(())
    }

    fn write_elements_binary<
        const N: usize,
        I1: ExactSizeIterator<Item = [usize; N]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        kwd: &str,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        let (m, kwd) = match kwd {
            "Edges" => (2, 5),
            "EdgesP2" => (3, 25),
            "Triangles" => (3, 6),
            "TrianglesP2" => (6, 24),
            "Tetrahedra" => (4, 8),
            "TetrahedraP2" => (10, 30),
            _ => unreachable!(),
        };
        assert_eq!(N, m);

        let mut next = self.writer.stream_position().unwrap();
        next += self.size_of_kwd(); // Keyword
        next += self.size_of_pos();
        next += self.size_of_index();
        next += (m * elems.len()) as u64 * self.size_of_index(); // Coordinates
        next += elems.len() as u64 * self.size_of_index(); // Tags

        self.write_kwd(kwd);
        self.write_pos(next as i64);
        self.write_index(elems.len() as i64);

        for (v, t) in elems.zip(tags) {
            for x in v {
                self.write_index(x as i64 + 1);
            }
            self.write_index(i64::from(t));
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
        self.write_elements("Edges", elems, tags)
    }

    pub fn write_quadratic_edges<
        I1: ExactSizeIterator<Item = [usize; 3]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        self.write_elements("EdgesP2", elems, tags)
    }

    pub fn write_triangles<
        I1: ExactSizeIterator<Item = [usize; 3]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        self.write_elements("Triangles", elems, tags)
    }

    pub fn write_quadratic_triangles<
        I1: ExactSizeIterator<Item = [usize; 6]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        self.write_elements("TrianglesP2", elems, tags)
    }

    pub fn write_quadratic_tetrahedra<
        I1: ExactSizeIterator<Item = [usize; 10]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        self.write_elements("TetrahedraP2", elems, tags)
    }

    pub fn write_tetrahedra<
        I1: ExactSizeIterator<Item = [usize; 4]>,
        I2: ExactSizeIterator<Item = i32>,
    >(
        &mut self,
        elems: I1,
        tags: I2,
    ) -> Result<()> {
        self.write_elements("Tetrahedra", elems, tags)
    }

    pub fn write_solution<const N: usize, I: ExactSizeIterator<Item = [f64; N]>>(
        &mut self,
        sols: I,
    ) -> Result<()> {
        debug!("write field");

        if self.is_binary {
            self.write_solution_binary(sols)
        } else {
            self.write_solution_ascii(sols)
        }
    }

    fn get_solution_type<const N: usize>(&self) -> Result<u8> {
        if N == 1 {
            Ok(1)
        } else if N == self.dimension as usize {
            Ok(2)
        } else if N == (self.dimension * (self.dimension + 1) / 2) as usize {
            Ok(3)
        } else {
            Err(Error::from(&format!("Unvalid field size {N}")))
        }
    }

    fn write_solution_ascii<const N: usize, I: ExactSizeIterator<Item = [f64; N]>>(
        &mut self,
        sols: I,
    ) -> Result<()> {
        writeln!(self.writer, "SolAtVertices")?;
        writeln!(self.writer, "{}", sols.len())?;
        writeln!(self.writer, "1 {}", self.get_solution_type::<N>()?)?;

        let mut line = String::new();

        for s in sols {
            line.clear();
            for x in s {
                line += &format!("{x} ");
            }
            writeln!(self.writer, "{}", &line)?;
        }

        Ok(())
    }

    fn write_solution_binary<const N: usize, I: ExactSizeIterator<Item = [f64; N]>>(
        &mut self,
        sols: I,
    ) -> Result<()> {
        let mut next = self.writer.stream_position().unwrap();
        next += self.size_of_kwd(); // Keyword
        next += self.size_of_pos();
        next += self.size_of_index();
        next += 2 * self.size_of_kwd(); // field type
        next += N as u64 * sols.len() as u64 * self.size_of_float(); // Values

        self.write_kwd(62);
        self.write_pos(next as i64);
        self.write_index(sols.len() as i64);
        self.write_kwd(1);
        self.write_kwd(i32::from(self.get_solution_type::<N>()?));

        for s in sols {
            for x in s {
                self.write_float(x);
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

    #[cfg(feature = "libmeshb-sys")]
    use crate::libmeshb::GmfReader;
    #[cfg(feature = "libmeshb-sys")]
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn to_vecs<const N: usize, T, I: ExactSizeIterator<Item = ([T; N], i32)>>(
        it: I,
    ) -> (Vec<[T; N]>, Vec<i32>) {
        let mut a = Vec::with_capacity(it.len());
        let mut b = Vec::with_capacity(it.len());

        for (x, y) in it {
            a.push(x);
            b.push(y);
        }
        (a, b)
    }

    fn to_vec<const N: usize, T, I: ExactSizeIterator<Item = [T; N]>>(it: I) -> Vec<[T; N]> {
        it.collect()
    }

    #[test]
    fn test_write_ascii_3d() {
        let mut reader = MeshbReader::new("./data/mesh3d.mesh").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".mesh";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let (verts, vtags) = to_vecs(reader.read_vertices::<3>().unwrap());
        writer
            .write_vertices(verts.iter().copied(), vtags.iter().copied())
            .unwrap();

        let (tris, tritags) = to_vecs(reader.read_triangles().unwrap());
        writer
            .write_triangles(tris.iter().copied(), tritags.iter().copied())
            .unwrap();

        let (tets, tettags) = to_vecs(reader.read_tetrahedra().unwrap());
        writer
            .write_tetrahedra(tets.iter().copied(), tettags.iter().copied())
            .unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let (verts2, vtags2) = to_vecs(reader2.read_vertices::<3>().unwrap());
        let (tris2, tritags2) = to_vecs(reader2.read_triangles().unwrap());
        let (tets2, tettags2) = to_vecs(reader2.read_tetrahedra().unwrap());

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

        let (verts, vtags) = to_vecs(reader.read_vertices::<3>().unwrap());
        writer
            .write_vertices(verts.iter().copied(), vtags.iter().copied())
            .unwrap();

        let (tris, tritags) = to_vecs(reader.read_triangles().unwrap());
        writer
            .write_triangles(tris.iter().copied(), tritags.iter().copied())
            .unwrap();

        let (tets, tettags) = to_vecs(reader.read_tetrahedra().unwrap());
        writer
            .write_tetrahedra(tets.iter().copied(), tettags.iter().copied())
            .unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let (verts2, vtags2) = to_vecs(reader2.read_vertices::<3>().unwrap());
        let (tris2, tritags2) = to_vecs(reader2.read_triangles().unwrap());
        let (tets2, tettags2) = to_vecs(reader2.read_tetrahedra().unwrap());

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

        let (verts, vtags) = to_vecs(reader.read_vertices::<3>().unwrap());
        writer
            .write_vertices(verts.iter().copied(), vtags.iter().copied())
            .unwrap();

        let (tris, tritags) = to_vecs(reader.read_triangles().unwrap());
        writer
            .write_triangles(tris.iter().copied(), tritags.iter().copied())
            .unwrap();

        let (tets, tettags) = to_vecs(reader.read_tetrahedra().unwrap());
        writer
            .write_tetrahedra(tets.iter().copied(), tettags.iter().copied())
            .unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        assert_eq!(reader2.version(), 4);

        let (verts2, vtags2) = to_vecs(reader2.read_vertices::<3>().unwrap());
        let (tris2, tritags2) = to_vecs(reader2.read_triangles().unwrap());
        let (tets2, tettags2) = to_vecs(reader2.read_tetrahedra().unwrap());

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

        let sols = to_vec(reader.read_solution::<1>().unwrap());
        writer.write_solution(sols.iter().copied()).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let sols2 = to_vec(reader2.read_solution::<1>().unwrap());

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            assert!((sol0[0] - sol1[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_ascii_sol_3d_vec() {
        let mut reader = MeshbReader::new("./data/sol3d_vec.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".sol";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let sols = to_vec(reader.read_solution::<3>().unwrap());
        writer.write_solution(sols.iter().copied()).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let sols2 = to_vec(reader2.read_solution::<3>().unwrap());

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

        let sols = to_vec(reader.read_solution::<1>().unwrap());
        writer.write_solution(sols.iter().copied()).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let sols2 = to_vec(reader2.read_solution::<1>().unwrap());

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            assert!((sol0[0] - sol1[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_binary_sol_3d_vec() {
        let mut reader = MeshbReader::new("./data/sol3d_vec.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".solb";

        let mut writer = MeshbWriter::new(&fname, 1, 3).unwrap();

        let sols = to_vec(reader.read_solution::<3>().unwrap());
        writer.write_solution(sols.iter().copied()).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let sols2 = to_vec(reader2.read_solution::<3>().unwrap());

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

        let (verts, vtags) = to_vecs(reader.read_vertices::<2>().unwrap());
        writer
            .write_vertices(verts.iter().copied(), vtags.iter().copied())
            .unwrap();

        let (edgs, edgtags) = to_vecs(reader.read_edges().unwrap());
        writer
            .write_edges(edgs.iter().copied(), edgtags.iter().copied())
            .unwrap();

        let (tris, tritags) = to_vecs(reader.read_triangles().unwrap());
        writer
            .write_triangles(tris.iter().copied(), tritags.iter().copied())
            .unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let (verts2, vtags2) = to_vecs(reader2.read_vertices::<2>().unwrap());
        let (edgs2, edgtags2) = to_vecs(reader2.read_edges().unwrap());
        let (tris2, tritags2) = to_vecs(reader2.read_triangles().unwrap());

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

        let (verts, vtags) = to_vecs(reader.read_vertices::<2>().unwrap());
        writer
            .write_vertices(verts.iter().copied(), vtags.iter().copied())
            .unwrap();

        let (edgs, edgtags) = to_vecs(reader.read_edges().unwrap());
        writer
            .write_edges(edgs.iter().copied(), edgtags.iter().copied())
            .unwrap();

        let (tris, tritags) = to_vecs(reader.read_triangles().unwrap());
        writer
            .write_triangles(tris.iter().copied(), tritags.iter().copied())
            .unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let (verts2, vtags2) = to_vecs(reader2.read_vertices::<2>().unwrap());
        let (edgs2, edgtags2) = to_vecs(reader2.read_edges().unwrap());
        let (tris2, tritags2) = to_vecs(reader2.read_triangles().unwrap());

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

        let (verts, vtags) = to_vecs(reader.read_vertices::<2>().unwrap());
        writer
            .write_vertices(verts.iter().copied(), vtags.iter().copied())
            .unwrap();

        let (edgs, edgtags) = to_vecs(reader.read_edges().unwrap());
        writer
            .write_edges(edgs.iter().copied(), edgtags.iter().copied())
            .unwrap();

        let (tris, tritags) = to_vecs(reader.read_triangles().unwrap());
        writer
            .write_triangles(tris.iter().copied(), tritags.iter().copied())
            .unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let (verts2, vtags2) = to_vecs(reader2.read_vertices::<2>().unwrap());
        let (edgs2, edgtags2) = to_vecs(reader2.read_edges().unwrap());
        let (tris2, tritags2) = to_vecs(reader2.read_triangles().unwrap());

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

        let sols = to_vec(reader.read_solution::<1>().unwrap());
        writer.write_solution(sols.iter().copied()).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let sols2 = to_vec(reader2.read_solution::<1>().unwrap());

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            assert!((sol0[0] - sol1[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_ascii_sol_2d_vec() {
        let mut reader = MeshbReader::new("./data/sol2d_vec.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".sol";

        let mut writer = MeshbWriter::new(&fname, 1, 2).unwrap();

        let sols = to_vec(reader.read_solution::<2>().unwrap());
        writer.write_solution(sols.iter().copied()).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(!reader2.is_binary());
        let sols2 = to_vec(reader2.read_solution::<2>().unwrap());

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

        let sols = to_vec(reader.read_solution::<1>().unwrap());
        writer.write_solution(sols.iter().copied()).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let sols2 = to_vec(reader2.read_solution::<1>().unwrap());

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            assert!((sol0[0] - sol1[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_binary_sol_2d_vec() {
        let mut reader = MeshbReader::new("./data/sol2d_vec.sol").unwrap();
        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".solb";

        let mut writer = MeshbWriter::new(&fname, 1, 2).unwrap();

        let sols = to_vec(reader.read_solution::<2>().unwrap());
        writer.write_solution(sols.iter().copied()).unwrap();

        writer.close();

        let mut reader2 = MeshbReader::new(&fname).unwrap();
        assert!(reader2.is_binary());
        let sols2 = to_vec(reader2.read_solution::<2>().unwrap());

        for (sol0, sol1) in sols.iter().zip(sols2.iter()) {
            for (s0, s1) in sol0.iter().zip(sol1.iter()) {
                assert!((s0 - s1).abs() < 1e-6);
            }
        }
    }

    #[cfg(feature = "libmeshb-sys")]
    fn test_libmeshb_3d(version: u8, binary: bool) {
        let mut rng = StdRng::seed_from_u64(1234);

        let n_verts = 100;
        let n_elems = 200;
        let n_faces = 50;

        let verts = (0..n_verts)
            .map(|_| {
                [
                    rng.random::<f64>() - 0.5,
                    rng.random::<f64>() - 0.5,
                    rng.random::<f64>() - 0.5,
                ]
            })
            .collect::<Vec<_>>();

        let elems = (0..n_elems)
            .map(|_| {
                [
                    rng.random_range(0..n_verts) as usize,
                    rng.random_range(0..n_verts) as usize,
                    rng.random_range(0..n_verts) as usize,
                    rng.random_range(0..n_verts) as usize,
                ]
            })
            .collect::<Vec<_>>();
        let etags = (0..n_elems)
            .map(|_| rng.random_range(0..10))
            .collect::<Vec<_>>();

        let faces = (0..n_faces)
            .map(|_| {
                [
                    rng.random_range(0..n_verts) as usize,
                    rng.random_range(0..n_verts) as usize,
                    rng.random_range(0..n_verts) as usize,
                ]
            })
            .collect::<Vec<_>>();
        let ftags: Vec<i32> = (0..n_faces)
            .map(|_| rng.random_range(0..10))
            .collect::<Vec<_>>();

        let file = NamedTempFile::new().unwrap();
        let fname = if binary {
            file.path().to_str().unwrap().to_owned() + ".meshb"
        } else {
            file.path().to_str().unwrap().to_owned() + ".mesh"
        };

        let mut writer = MeshbWriter::new(&fname, version, 3).unwrap();
        writer
            .write_vertices(verts.iter().copied(), verts.iter().map(|_| 1))
            .unwrap();
        writer
            .write_tetrahedra(elems.iter().copied(), etags.iter().copied())
            .unwrap();
        writer
            .write_triangles(faces.iter().copied(), ftags.iter().copied())
            .unwrap();
        writer.close();

        let mut reader = MeshbReader::new(&fname).unwrap();
        for (i, (vert, tag)) in reader.read_vertices::<3>().unwrap().enumerate() {
            assert_eq!(tag, 1);
            for j in 0..3 {
                assert!((vert[j] - verts[i][j]).abs() < 1e-5);
            }
        }

        for (i, (elem, tag)) in reader.read_tetrahedra().unwrap().enumerate() {
            assert_eq!(tag, etags[i]);
            for j in 0..4 {
                assert_eq!(elem[j], elems[i][j]);
            }
        }

        for (i, (face, tag)) in reader.read_triangles().unwrap().enumerate() {
            assert_eq!(tag, ftags[i]);
            for j in 0..3 {
                assert_eq!(face[j], faces[i][j]);
            }
        }
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_3d_ascii_1() {
        test_libmeshb_3d(1, false);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_3d_ascii_2() {
        test_libmeshb_3d(2, false);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_3d_ascii_3() {
        test_libmeshb_3d(3, false);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_3d_ascii_4() {
        test_libmeshb_3d(4, false);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_3d_binary_1() {
        test_libmeshb_3d(1, true);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_3d_binary_2() {
        test_libmeshb_3d(2, true);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_3d_binary_3() {
        test_libmeshb_3d(3, true);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_3d_binary_4() {
        test_libmeshb_3d(4, true);
    }

    #[cfg(feature = "libmeshb-sys")]
    fn test_libmeshb_2d(version: u8, binary: bool) {
        let mut rng = StdRng::seed_from_u64(1234);

        let n_verts = 100;
        let n_elems = 200;
        let n_faces = 50;

        let verts = (0..n_verts)
            .map(|_| [rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5])
            .collect::<Vec<_>>();

        let elems = (0..n_elems)
            .map(|_| {
                [
                    rng.random_range(0..n_verts) as usize,
                    rng.random_range(0..n_verts) as usize,
                    rng.random_range(0..n_verts) as usize,
                ]
            })
            .collect::<Vec<_>>();
        let etags = (0..n_elems)
            .map(|_| rng.random_range(0..10))
            .collect::<Vec<_>>();

        let faces = (0..n_faces)
            .map(|_| {
                [
                    rng.random_range(0..n_verts) as usize,
                    rng.random_range(0..n_verts) as usize,
                ]
            })
            .collect::<Vec<_>>();
        let ftags: Vec<i32> = (0..n_faces)
            .map(|_| rng.random_range(0..10))
            .collect::<Vec<_>>();

        let file = NamedTempFile::new().unwrap();
        let fname = if binary {
            file.path().to_str().unwrap().to_owned() + ".meshb"
        } else {
            file.path().to_str().unwrap().to_owned() + ".mesh"
        };

        let mut writer = MeshbWriter::new(&fname, version, 2).unwrap();
        writer
            .write_vertices(verts.iter().copied(), verts.iter().map(|_| 1))
            .unwrap();
        writer
            .write_triangles(elems.iter().copied(), etags.iter().copied())
            .unwrap();
        writer
            .write_edges(faces.iter().copied(), ftags.iter().copied())
            .unwrap();
        writer.close();

        let mut reader = GmfReader::new(&fname).unwrap();
        for (i, (vert, tag)) in reader.read_vertices::<2>().unwrap().enumerate() {
            assert_eq!(tag, 1);
            for j in 0..2 {
                assert!((vert[j] - verts[i][j]).abs() < 1e-5);
            }
        }

        for (i, (elem, tag)) in reader.read_triangles().unwrap().enumerate() {
            assert_eq!(tag, etags[i]);
            for j in 0..3 {
                assert_eq!(elem[j], elems[i][j], "{i} {elem:?} {:?}", elems[i]);
            }
        }

        for (i, (face, tag)) in reader.read_edges().unwrap().enumerate() {
            assert_eq!(tag, ftags[i]);
            for j in 0..2 {
                assert_eq!(face[j], faces[i][j], "{i} {face:?} {:?}", faces[i]);
            }
        }
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_2d_ascii_1() {
        test_libmeshb_2d(1, false);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_2d_ascii_2() {
        test_libmeshb_2d(2, false);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_2d_ascii_3() {
        test_libmeshb_2d(3, false);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_2d_ascii_4() {
        test_libmeshb_2d(4, false);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_2d_binary_1() {
        test_libmeshb_2d(1, true);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_2d_binary_2() {
        test_libmeshb_2d(2, true);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_2d_binary_3() {
        test_libmeshb_2d(3, true);
    }

    #[cfg(feature = "libmeshb-sys")]
    #[test]
    fn test_libmeshb_2d_binary_4() {
        test_libmeshb_2d(4, true);
    }
}
