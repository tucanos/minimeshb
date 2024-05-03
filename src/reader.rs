use log::debug;

use crate::{Element, Error, Result, Solution, Tag, Vertex};
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

    pub fn read_vertices<V: Vertex, T: Tag>(&mut self) -> Result<(Vec<V>, Vec<T>)> {
        let n_verts = self.goto_section("Vertices")?;
        debug!("read {n_verts} vertices");

        let mut verts = Vec::with_capacity(n_verts);
        let mut tags = Vec::with_capacity(n_verts);

        let mut vals = vec![0.0; self.dimension as usize];
        let mut tag = -1;

        let mut line = String::new();
        while verts.len() < n_verts {
            if self.is_binary {
                for v in vals.iter_mut() {
                    *v = self.read_float();
                }
                tag = self.read_kwd();
            } else {
                let len = self.reader.read_line(&mut line)?;
                assert_ne!(len, 0);
                let trimmed_line = line.trim();
                if trimmed_line.is_empty() {
                    continue;
                }
                for (i, v) in trimmed_line.split(' ').enumerate() {
                    if i < self.dimension as usize {
                        vals[i] = v.parse().unwrap();
                    }
                    if i == self.dimension as usize {
                        tag = v.parse().unwrap();
                    }
                }
                line.clear();
            }
            verts.push(V::from(&vals));
            tags.push(T::from(tag));
        }

        Ok((verts, tags))
    }

    fn read_elements<E: Element, T: Tag>(&mut self, kwd: &str) -> Result<(Vec<E>, Vec<T>)> {
        let n_elems = self.goto_section(kwd)?;
        debug!("read {n_elems} elements");

        let m = match kwd {
            "Edges" => 2,
            "Triangles" => 3,
            "Tetrahedra" => 4,
            _ => unreachable!(),
        };

        let mut elems = Vec::with_capacity(n_elems);
        let mut tags = Vec::with_capacity(n_elems);

        let mut vals = vec![0_u64; m];
        let mut tag = -1;
        let mut line = String::new();

        while elems.len() < n_elems {
            if self.is_binary {
                for v in vals.iter_mut() {
                    *v = self.read_index() - 1;
                }
                tag = self.read_kwd();
            } else {
                let len = self.reader.read_line(&mut line)?;
                assert_ne!(len, 0);
                let trimmed_line = line.trim();
                if trimmed_line.is_empty() {
                    continue;
                }
                for (i, v) in trimmed_line.split(' ').enumerate() {
                    if i < m {
                        vals[i] = v.parse::<u64>().unwrap() - 1;
                    }
                    if i == m {
                        tag = v.parse().unwrap();
                    }
                }
                line.clear();
            }
            elems.push(E::from(&vals));
            tags.push(T::from(tag));
        }

        Ok((elems, tags))
    }

    pub fn read_edges<E: Element, T: Tag>(&mut self) -> Result<(Vec<E>, Vec<T>)> {
        self.read_elements("Edges")
    }

    pub fn read_triangles<E: Element, T: Tag>(&mut self) -> Result<(Vec<E>, Vec<T>)> {
        self.read_elements("Triangles")
    }

    pub fn read_tetrahedra<E: Element, T: Tag>(&mut self) -> Result<(Vec<E>, Vec<T>)> {
        self.read_elements("Tetrahedra")
    }

    pub fn get_solution_size(&mut self) -> Result<usize> {
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

    pub fn read_solution<S: Solution>(&mut self) -> Result<Vec<S>> {
        let n_verts = self.goto_section("SolAtVertices")?;

        let m = self.get_solution_size()?;

        let mut sol = Vec::with_capacity(n_verts);

        let mut vals = vec![0.0; m];
        if !self.is_binary {
            vals.clear();
        }
        let mut line = String::new();
        while sol.len() < n_verts {
            if self.is_binary {
                for v in vals.iter_mut() {
                    *v = self.read_float();
                }
                sol.push(S::from(&vals));
            } else {
                let len = self.reader.read_line(&mut line)?;
                assert_ne!(len, 0);
                let trimmed_line = line.trim();
                if trimmed_line.is_empty() {
                    continue;
                }
                for v in trimmed_line.split(' ') {
                    vals.push(v.parse().unwrap());
                    if vals.len() == m {
                        sol.push(S::from(&vals));
                        vals.clear();
                    }
                }
                line.clear();
            }
        }

        Ok(sol)
    }
}

#[cfg(test)]
mod tests {
    use super::MeshbReader;

    #[test]
    fn test_read_ascii_3d() {
        let mut reader = MeshbReader::new("./data/mesh3d.mesh").unwrap();
        assert_eq!(reader.dimension(), 3);

        let (verts, _) = reader.read_vertices::<[f64; 3], ()>().unwrap();
        assert_eq!(verts.len(), 26);
        let (tris, tags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        assert_eq!(tris.len(), 48);
        assert_eq!(tags.len(), 48);
        let (tets, tags) = reader.read_tetrahedra::<[u32; 4], i16>().unwrap();
        assert_eq!(tets.len(), 40);
        assert_eq!(tags.len(), 40);
    }

    #[test]
    #[should_panic]
    fn test_read_ascii_3d_sol() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.sol").unwrap();
        let _ = reader.read_vertices::<[f64; 3], ()>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_read_ascii_3d_sol_2() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.sol").unwrap();
        let _ = reader.read_solution::<[f32; 3]>().unwrap();
    }

    #[test]
    fn test_read_ascii_3d_sol_3() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.sol").unwrap();
        let sol = reader.read_solution::<f32>().unwrap();
        assert_eq!(sol.len(), 26);
    }

    #[test]
    fn test_read_ascii_3d_sol_4() {
        let mut reader = MeshbReader::new("./data/sol3d_vec.sol").unwrap();
        let sol = reader.read_solution::<[f32; 3]>().unwrap();
        assert_eq!(sol.len(), 26);
    }

    #[test]
    fn test_read_binary_3d() {
        let mut reader = MeshbReader::new("./data/mesh3d.meshb").unwrap();

        assert_eq!(reader.dimension(), 3);

        let (verts, _) = reader.read_vertices::<[f64; 3], ()>().unwrap();
        assert_eq!(verts.len(), 26);
        let (tris, tags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        assert_eq!(tris.len(), 48);
        assert_eq!(tags.len(), 48);
        let (tets, tags) = reader.read_tetrahedra::<[u32; 4], i16>().unwrap();
        assert_eq!(tets.len(), 40);
        assert_eq!(tags.len(), 40);
    }

    #[test]
    #[should_panic]
    fn test_read_binary_3d_sol() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.solb").unwrap();
        let _ = reader.read_vertices::<[f64; 3], ()>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_read_binary_3d_sol_2() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.solb").unwrap();
        let _ = reader.read_solution::<[f32; 3]>().unwrap();
    }

    #[test]
    fn test_read_binary_3d_sol_3() {
        let mut reader = MeshbReader::new("./data/sol3d_scalar.solb").unwrap();
        let sol = reader.read_solution::<f32>().unwrap();
        assert_eq!(sol.len(), 26);
    }

    #[test]
    fn test_read_binary_3d_sol_4() {
        let mut reader = MeshbReader::new("./data/sol3d_vec.solb").unwrap();
        let sol = reader.read_solution::<[f32; 3]>().unwrap();
        assert_eq!(sol.len(), 26);
    }

    #[test]
    fn test_compare_3d() {
        let mut areader = MeshbReader::new("./data/mesh3d.mesh").unwrap();
        let mut breader = MeshbReader::new("./data/mesh3d.meshb").unwrap();

        let (averts, _) = areader.read_vertices::<[f64; 3], ()>().unwrap();
        let (bverts, _) = breader.read_vertices::<[f64; 3], ()>().unwrap();

        for (vert0, vert1) in averts.iter().zip(bverts.iter()) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        let (atris, atags) = areader.read_triangles::<[u32; 3], i16>().unwrap();
        let (btris, btags) = breader.read_triangles::<[u32; 3], i16>().unwrap();

        for (tri0, tri1) in atris.iter().zip(btris.iter()) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in atags.iter().zip(btags.iter()) {
            assert_eq!(tag0, tag1);
        }

        let (atets, atags) = areader.read_tetrahedra::<[u32; 4], i16>().unwrap();
        let (btets, btags) = breader.read_tetrahedra::<[u32; 4], i16>().unwrap();

        for (tet0, tet1) in atets.iter().zip(btets.iter()) {
            for (v0, v1) in tet0.iter().zip(tet1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in atags.iter().zip(btags.iter()) {
            assert_eq!(tag0, tag1);
        }
    }

    #[test]
    fn test_read_ascii_2d() {
        let mut reader = MeshbReader::new("./data/mesh2d.mesh").unwrap();
        assert_eq!(reader.dimension(), 2);

        let (verts, _) = reader.read_vertices::<[f64; 2], ()>().unwrap();
        assert_eq!(verts.len(), 9);
        let (edgs, tags) = reader.read_edges::<[u32; 2], i16>().unwrap();
        assert_eq!(edgs.len(), 10);
        assert_eq!(tags.len(), 10);
        assert_eq!(tags, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);

        let (tris, tags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        assert_eq!(tris.len(), 8);
        assert_eq!(tags.len(), 8);
        assert_eq!(tags, [1, 1, 1, 1, 2, 2, 2, 2]);
    }

    #[test]
    #[should_panic]
    fn test_read_ascii_2d_sol() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.sol").unwrap();
        let _ = reader.read_vertices::<[f64; 2], ()>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_read_ascii_2d_sol_2() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.sol").unwrap();
        let _ = reader.read_solution::<[f32; 2]>().unwrap();
    }

    #[test]
    fn test_read_ascii_2d_sol_3() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.sol").unwrap();
        let sol = reader.read_solution::<f32>().unwrap();
        assert_eq!(sol.len(), 9);
    }

    #[test]
    fn test_read_ascii_2d_sol_4() {
        let mut reader = MeshbReader::new("./data/sol2d_vec.sol").unwrap();
        let sol = reader.read_solution::<[f32; 2]>().unwrap();
        assert_eq!(sol.len(), 9);
    }

    #[test]
    fn test_read_binary_2d() {
        let mut reader = MeshbReader::new("./data/mesh2d.meshb").unwrap();
        assert_eq!(reader.dimension(), 2);

        let (verts, _) = reader.read_vertices::<[f64; 2], ()>().unwrap();
        assert_eq!(verts.len(), 9);
        let (edgs, tags) = reader.read_edges::<[u32; 2], i16>().unwrap();
        assert_eq!(edgs.len(), 10);
        assert_eq!(tags.len(), 10);
        assert_eq!(tags, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);

        let (tris, tags) = reader.read_triangles::<[u32; 3], i16>().unwrap();
        assert_eq!(tris.len(), 8);
        assert_eq!(tags.len(), 8);
        assert_eq!(tags, [1, 1, 1, 1, 2, 2, 2, 2]);
    }

    #[test]
    #[should_panic]
    fn test_read_binary_2d_sol() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.solb").unwrap();
        let _ = reader.read_vertices::<[f64; 3], ()>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_read_binary_2d_sol_2() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.solb").unwrap();
        let _ = reader.read_solution::<[f32; 3]>().unwrap();
    }

    #[test]
    fn test_read_binary_2d_sol_3() {
        let mut reader = MeshbReader::new("./data/sol2d_scalar.solb").unwrap();
        let sol = reader.read_solution::<f32>().unwrap();
        assert_eq!(sol.len(), 9);
    }

    #[test]
    fn test_read_binary_2d_sol_4() {
        let mut reader = MeshbReader::new("./data/sol2d_vec.solb").unwrap();
        let sol = reader.read_solution::<[f32; 2]>().unwrap();
        assert_eq!(sol.len(), 9);
    }

    #[test]
    fn test_compare_2d() {
        let mut areader = MeshbReader::new("./data/mesh2d.mesh").unwrap();
        let mut breader = MeshbReader::new("./data/mesh2d.meshb").unwrap();

        let (averts, _) = areader.read_vertices::<[f64; 2], ()>().unwrap();
        let (bverts, _) = breader.read_vertices::<[f64; 2], ()>().unwrap();

        for (vert0, vert1) in averts.iter().zip(bverts.iter()) {
            for (v0, v1) in vert0.iter().zip(vert1.iter()) {
                assert!((v0 - v1).abs() < 1e-6);
            }
        }

        let (aedgs, atags) = areader.read_edges::<[u32; 2], i16>().unwrap();
        let (bedgs, btags) = breader.read_edges::<[u32; 2], i16>().unwrap();

        for (edg0, edg1) in aedgs.iter().zip(bedgs.iter()) {
            for (v0, v1) in edg0.iter().zip(edg1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in atags.iter().zip(btags.iter()) {
            assert_eq!(tag0, tag1);
        }

        let (atris, atags) = areader.read_triangles::<[u32; 3], i16>().unwrap();
        let (btris, btags) = breader.read_triangles::<[u32; 3], i16>().unwrap();

        for (tri0, tri1) in atris.iter().zip(btris.iter()) {
            for (v0, v1) in tri0.iter().zip(tri1.iter()) {
                assert_eq!(v0, v1);
            }
        }

        for (tag0, tag1) in atags.iter().zip(btags.iter()) {
            assert_eq!(tag0, tag1);
        }
    }
}
