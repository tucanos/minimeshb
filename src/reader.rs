use crate::{Error, Result};
use log::debug;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
};

#[derive(Debug)]
pub struct MeshbReader {
    is_binary: bool,
    version: u8,
    dimension: u8,
    offsets: HashMap<String, u64>,
    reader: BufReader<File>,
}

impl MeshbReader {
    pub fn new(fname: &str) -> Result<Self> {
        if !fname.ends_with(".mesh")
            && !fname.ends_with(".meshb")
            && !fname.ends_with(".sol")
            && !fname.ends_with(".solb")
        {
            return Err(Error::from(&format!("Invalid file extension for {fname}")));
        }

        if fname.ends_with(".meshb") || fname.ends_with(".solb") {
            Self::new_binary(fname)
        } else {
            Self::new_ascii(fname)
        }
    }

    pub fn dimension(&self) -> u8 {
        self.dimension
    }

    pub fn version(&self) -> u8 {
        self.version
    }

    pub fn is_binary(&self) -> bool {
        self.is_binary
    }

    fn new_ascii(fname: &str) -> Result<Self> {
        debug!("parse {fname} (ascii)");
        let is_binary = false;
        let mut version = 0;
        let mut dimension = 0;
        let mut offsets = HashMap::new();

        let file = File::open(fname)?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            let trimmed_line = line.trim();
            if trimmed_line.starts_with("MeshVersionFormatted") {
                let line = trimmed_line
                    .strip_prefix("MeshVersionFormatted")
                    .unwrap()
                    .trim();
                version = line.parse().unwrap();
                debug!("version = {version}");
            } else if trimmed_line.starts_with("Dimension") {
                let line = trimmed_line.strip_prefix("Dimension").unwrap().trim();
                dimension = line.parse().unwrap();
                debug!("dimension = {dimension}");
            } else if trimmed_line == "Vertices"
                || trimmed_line == "Edges"
                || trimmed_line == "Triangles"
                || trimmed_line == "Tetrahedra"
                || trimmed_line == "SolAtVertices"
            {
                debug!("found entry {trimmed_line}");
                offsets.insert(
                    String::from(trimmed_line),
                    reader.stream_position().unwrap(),
                );
            }
            line.clear();
        }

        reader.rewind()?;

        Ok(Self {
            is_binary,
            version,
            dimension,
            offsets,
            reader,
        })
    }

    fn read_kwd(&mut self) -> i32 {
        let mut buffer = [0u8; std::mem::size_of::<i32>()];
        self.reader.read_exact(&mut buffer).unwrap();
        i32::from_le_bytes(buffer)
    }

    fn read_float(&mut self) -> f64 {
        if self.version == 1 {
            let mut buffer = [0u8; std::mem::size_of::<f32>()];
            self.reader.read_exact(&mut buffer).unwrap();
            f32::from_le_bytes(buffer) as f64
        } else {
            let mut buffer = [0u8; std::mem::size_of::<f64>()];
            self.reader.read_exact(&mut buffer).unwrap();
            f64::from_le_bytes(buffer)
        }
    }

    fn read_index(&mut self) -> u64 {
        if self.version >= 3 {
            let mut buffer = [0u8; std::mem::size_of::<u64>()];
            self.reader.read_exact(&mut buffer).unwrap();
            u64::from_le_bytes(buffer)
        } else {
            let mut buffer = [0u8; std::mem::size_of::<u32>()];
            self.reader.read_exact(&mut buffer).unwrap();
            u32::from_le_bytes(buffer) as u64
        }
    }

    fn new_binary(fname: &str) -> Result<Self> {
        debug!("parse {fname} (binary)");
        let is_binary = true;
        let offsets = HashMap::new();

        let file = File::open(fname)?;
        let reader = BufReader::new(file);

        let mut res = Self {
            is_binary,
            version: 0,
            dimension: 0,
            offsets,
            reader,
        };

        let cod = res.read_kwd();
        assert!(cod == 1 || cod == 16777216);
        assert!(cod == 1, "for now");
        res.version = res.read_kwd() as u8;
        debug!("version = {}", res.version);

        let kwd = res.read_kwd() as u8;
        assert_eq!(kwd, 3); // Dimension
        let mut next_offset = res.read_index();
        res.dimension = res.read_kwd() as u8;
        debug!("dimension = {}", res.dimension);

        loop {
            debug!("next_offset {next_offset}");
            res.reader.seek(SeekFrom::Start(next_offset))?;
            let kwd = res.read_kwd() as u8;
            let name = match kwd {
                4 => "Vertices",
                5 => "Edges",
                6 => "Triangles",
                8 => "Tetrahedra",
                62 => "SolAtVertices",
                54 => "End",
                _ => return Err(Error::from(&format!("Unknown keyword {kwd}"))),
            };
            debug!("found entry {name}");
            if kwd == 54 {
                break;
            }
            res.offsets
                .insert(String::from(name), res.reader.stream_position().unwrap());
            next_offset = res.read_index();
        }

        Ok(res)
    }

    fn goto_section(&mut self, kwd: &str) -> Result<usize> {
        if let Some(offset) = self.offsets.get(kwd) {
            self.reader.seek(SeekFrom::Start(*offset))?;

            if self.is_binary {
                let _ = self.read_index();
                let n = self.read_index();
                return Ok(n as usize);
            } else {
                let mut line = String::new();
                while self.reader.read_line(&mut line)? > 0 {
                    let trimmed_line = line.trim();
                    if trimmed_line.is_empty() {
                        continue;
                    }
                    return Ok(trimmed_line.parse().unwrap());
                }
            }
        }
        Err(Error::from(&format!("Unable to get section {kwd}")))
    }

    pub fn read_vertices<const D: usize>(
        &mut self,
    ) -> Result<impl ExactSizeIterator<Item = ([f64; D], i32)> + '_> {
        assert_eq!(D, self.dimension as usize);

        let n_verts = self.goto_section("Vertices")?;
        debug!("read {n_verts} vertices");

        let mut vals = [0.0; D];
        let mut tag = -1;

        let mut line = String::new();

        Ok((0..n_verts).map(move |_| {
            if self.is_binary {
                for v in vals.iter_mut() {
                    *v = self.read_float();
                }
                tag = self.read_kwd();
            } else {
                let len = self.reader.read_line(&mut line).unwrap();
                assert_ne!(len, 0);
                let trimmed_line = line.trim();
                assert!(!trimmed_line.is_empty());
                for (i, v) in trimmed_line.split(' ').enumerate() {
                    if i < D {
                        vals[i] = v.parse().unwrap();
                    }
                    if i == D {
                        tag = v.parse().unwrap();
                    }
                }
                line.clear();
            }
            (vals, tag)
        }))
    }

    fn read_elements<const N: usize>(
        &mut self,
        kwd: &str,
    ) -> Result<impl ExactSizeIterator<Item = ([u64; N], i32)> + '_> {
        let m = match kwd {
            "Edges" => 2,
            "Triangles" => 3,
            "Tetrahedra" => 4,
            _ => unreachable!(),
        };
        assert_eq!(N, m);

        let n_elems = self.goto_section(kwd)?;
        debug!("read {n_elems} elements");

        let mut vals = [0_u64; N];
        let mut tag = -1;
        let mut line = String::new();

        Ok((0..n_elems).map(move |_| {
            if self.is_binary {
                for v in vals.iter_mut() {
                    *v = self.read_index() - 1;
                }
                tag = self.read_kwd();
            } else {
                let len = self.reader.read_line(&mut line).unwrap();
                assert_ne!(len, 0);
                let trimmed_line = line.trim();
                assert!(!trimmed_line.is_empty());
                for (i, v) in trimmed_line.split(' ').enumerate() {
                    if i < N {
                        vals[i] = v.parse::<u64>().unwrap() - 1;
                    }
                    if i == N {
                        tag = v.parse().unwrap();
                    }
                }
                line.clear();
            }
            (vals, tag)
        }))
    }

    pub fn read_edges(&mut self) -> Result<impl ExactSizeIterator<Item = ([u64; 2], i32)> + '_> {
        self.read_elements("Edges")
    }

    pub fn read_triangles(
        &mut self,
    ) -> Result<impl ExactSizeIterator<Item = ([u64; 3], i32)> + '_> {
        self.read_elements("Triangles")
    }

    pub fn read_tetrahedra(
        &mut self,
    ) -> Result<impl ExactSizeIterator<Item = ([u64; 4], i32)> + '_> {
        self.read_elements("Tetrahedra")
    }

    pub fn get_solution_size(&mut self) -> Result<usize> {
        let _ = self.goto_section("SolAtVertices")?;
        let m: i16;
        if self.is_binary {
            let n_fields = self.read_kwd();
            assert_eq!(n_fields, 1);
            m = self.read_kwd() as i16;
        } else {
            let mut line = String::new();
            loop {
                let len = self.reader.read_line(&mut line)?;
                assert_ne!(len, 0);
                let trimmed_line = line.trim();
                if trimmed_line.is_empty() {
                    continue;
                }
                let mut split = trimmed_line.split(' ');
                let n_fields: usize = split.next().unwrap().parse().unwrap();
                assert_eq!(n_fields, 1);
                m = split.next().unwrap().parse().unwrap();
                break;
            }
        }
        match m {
            1 => Ok(1),
            2 => Ok(self.dimension as usize),
            3 => Ok((self.dimension * (self.dimension + 1) / 2) as usize),
            _ => Err(Error::from(&format!("Unvalid field type {m}"))),
        }
    }

    pub fn read_solution<const N: usize>(
        &mut self,
    ) -> Result<impl ExactSizeIterator<Item = [f64; N]> + '_> {
        let n_verts = self.goto_section("SolAtVertices")?;
        let m = self.get_solution_size()?;
        assert_eq!(N, m);

        debug!("read field");

        let mut sol = Vec::with_capacity(n_verts);

        let mut vals = [0.0; N];
        let mut line = String::new();
        let mut idx = 0;
        let mut tmp = Vec::new();

        Ok((0..n_verts).map(move |_| {
            if self.is_binary {
                for v in vals.iter_mut() {
                    *v = self.read_float();
                }
                sol.push(vals);
            } else {
                for v in vals.iter_mut() {
                    if idx == tmp.len() {
                        line.clear();
                        let len = self.reader.read_line(&mut line).unwrap();
                        assert_ne!(len, 0);
                        let trimmed_line = line.trim();
                        assert!(!trimmed_line.is_empty());
                        tmp.clear();
                        for x in trimmed_line.split(' ') {
                            tmp.push(x.parse().unwrap())
                        }
                        idx = 0;
                    }
                    *v = tmp[idx];
                    idx += 1;
                }
            }
            vals
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::MeshbReader;

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

    #[test]
    fn test_read_ascii_3d() {
        let mut reader = MeshbReader::new("./data/mesh3d.mesh").unwrap();
        assert_eq!(reader.dimension(), 3);

        let (verts, _) = to_vecs(reader.read_vertices::<3>().unwrap());
        assert_eq!(verts.len(), 26);
        let (tris, _) = to_vecs(reader.read_triangles().unwrap());
        assert_eq!(tris.len(), 48);
        let (tets, _) = to_vecs(reader.read_tetrahedra().unwrap());
        assert_eq!(tets.len(), 40);
    }

    #[test]
    #[should_panic]
    fn test_read_ascii_3d_sol() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.sol").unwrap();
        let _ = reader.read_vertices::<3>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_read_ascii_3d_sol_2() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.sol").unwrap();
        let _ = reader.read_solution::<3>().unwrap();
    }

    #[test]
    fn test_read_ascii_3d_sol_3() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.sol").unwrap();
        let sol = reader.read_solution::<1>().unwrap();
        assert_eq!(sol.len(), 26);
    }

    #[test]
    fn test_read_ascii_3d_sol_4() {
        let mut reader = MeshbReader::new("./data/sol3d_vec.sol").unwrap();
        let sol = reader.read_solution::<3>().unwrap();
        assert_eq!(sol.len(), 26);
    }

    #[test]
    fn test_read_binary_3d() {
        let mut reader = MeshbReader::new("./data/mesh3d.meshb").unwrap();

        assert_eq!(reader.dimension(), 3);

        let (verts, _) = to_vecs(reader.read_vertices::<3>().unwrap());
        assert_eq!(verts.len(), 26);
        let (tris, _) = to_vecs(reader.read_triangles().unwrap());
        assert_eq!(tris.len(), 48);
        let (tets, _) = to_vecs(reader.read_tetrahedra().unwrap());
        assert_eq!(tets.len(), 40);
    }

    #[test]
    #[should_panic]
    fn test_read_binary_3d_sol() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.solb").unwrap();
        let _ = reader.read_vertices::<3>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_read_binary_3d_sol_2() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.solb").unwrap();
        let _ = reader.read_solution::<3>().unwrap();
    }

    #[test]
    fn test_read_binary_3d_sol_3() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.solb").unwrap();
        let sol = reader.read_solution::<1>().unwrap();
        assert_eq!(sol.len(), 26);
    }

    #[test]
    fn test_read_binary_3d_sol_4() {
        let mut reader = MeshbReader::new("./data/sol3d_vec.solb").unwrap();
        let sol = reader.read_solution::<3>().unwrap();
        assert_eq!(sol.len(), 26);
    }

    #[test]
    fn test_compare_3d() {
        let mut areader = MeshbReader::new("./data/mesh3d.mesh").unwrap();
        let mut breader = MeshbReader::new("./data/mesh3d.meshb").unwrap();

        let averts = areader.read_vertices::<3>().unwrap();
        let bverts = breader.read_vertices::<3>().unwrap();

        for ((vert0, _), (vert1, _)) in averts.zip(bverts) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        let atris = areader.read_triangles().unwrap();
        let btris = breader.read_triangles().unwrap();

        for ((tri0, tag0), (tri1, tag1)) in atris.zip(btris) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
            assert_eq!(tag0, tag1);
        }

        let atets = areader.read_tetrahedra().unwrap();
        let btets = breader.read_tetrahedra().unwrap();

        for ((tet0, tag0), (tet1, tag1)) in atets.zip(btets) {
            for (v0, v1) in tet0.iter().zip(tet1.iter()) {
                assert_eq!(v0, v1);
            }
            assert_eq!(tag0, tag1);
        }
    }

    #[test]
    fn test_read_ascii_2d() {
        let mut reader = MeshbReader::new("./data/mesh2d.mesh").unwrap();
        assert_eq!(reader.dimension(), 2);

        let (verts, _) = to_vecs(reader.read_vertices::<2>().unwrap());
        assert_eq!(verts.len(), 9);
        let (edgs, tags) = to_vecs(reader.read_edges().unwrap());
        assert_eq!(edgs.len(), 10);
        assert_eq!(tags.len(), 10);
        assert_eq!(tags, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);

        let (tris, tags) = to_vecs(reader.read_triangles().unwrap());
        assert_eq!(tris.len(), 8);
        assert_eq!(tags.len(), 8);
        assert_eq!(tags, [1, 1, 1, 1, 2, 2, 2, 2]);
    }

    #[test]
    #[should_panic]
    fn test_read_ascii_2d_sol() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.sol").unwrap();
        let _ = reader.read_vertices::<2>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_read_ascii_2d_sol_2() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.sol").unwrap();
        let _ = reader.read_solution::<2>().unwrap();
    }

    #[test]
    fn test_read_ascii_2d_sol_3() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.sol").unwrap();
        let sol = reader.read_solution::<1>().unwrap();
        assert_eq!(sol.len(), 9);
    }

    #[test]
    fn test_read_ascii_2d_sol_4() {
        let mut reader = MeshbReader::new("./data/sol2d_vec.sol").unwrap();
        let sol = reader.read_solution::<2>().unwrap();
        assert_eq!(sol.len(), 9);
    }

    #[test]
    fn test_read_binary_2d() {
        let mut reader = MeshbReader::new("./data/mesh2d.meshb").unwrap();
        assert_eq!(reader.dimension(), 2);

        let (verts, _) = to_vecs(reader.read_vertices::<2>().unwrap());
        assert_eq!(verts.len(), 9);
        let (edgs, tags) = to_vecs(reader.read_edges().unwrap());
        assert_eq!(edgs.len(), 10);
        assert_eq!(tags.len(), 10);
        assert_eq!(tags, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);

        let (tris, tags) = to_vecs(reader.read_triangles().unwrap());
        assert_eq!(tris.len(), 8);
        assert_eq!(tags.len(), 8);
        assert_eq!(tags, [1, 1, 1, 1, 2, 2, 2, 2]);
    }

    #[test]
    #[should_panic]
    fn test_read_binary_2d_sol() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.solb").unwrap();
        let _ = reader.read_vertices::<3>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_read_binary_2d_sol_2() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.solb").unwrap();
        let _ = reader.read_solution::<3>().unwrap();
    }

    #[test]
    fn test_read_binary_2d_sol_3() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.solb").unwrap();
        let sol = reader.read_solution::<1>().unwrap();
        assert_eq!(sol.len(), 9);
    }

    #[test]
    fn test_read_binary_2d_sol_4() {
        let mut reader = MeshbReader::new("./data/sol2d_vec.solb").unwrap();
        let sol = reader.read_solution::<2>().unwrap();
        assert_eq!(sol.len(), 9);
    }

    #[test]
    fn test_compare_2d() {
        let mut areader = MeshbReader::new("./data/mesh2d.mesh").unwrap();
        let mut breader = MeshbReader::new("./data/mesh2d.meshb").unwrap();

        let averts = areader.read_vertices::<2>().unwrap();
        let bverts = breader.read_vertices::<2>().unwrap();

        for ((vert0, _), (vert1, _)) in averts.zip(bverts) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        let aedgs = areader.read_edges().unwrap();
        let bedgs = breader.read_edges().unwrap();

        for ((edg0, tag0), (edg1, tag1)) in aedgs.zip(bedgs) {
            for (v0, v1) in edg0.iter().zip(edg1.iter()) {
                assert_eq!(v0, v1);
            }
            assert_eq!(tag0, tag1);
        }

        let atris = areader.read_triangles().unwrap();
        let btris = breader.read_triangles().unwrap();

        for ((tri0, tag0), (tri1, tag1)) in atris.zip(btris) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
            assert_eq!(tag0, tag1);
        }
    }
}
