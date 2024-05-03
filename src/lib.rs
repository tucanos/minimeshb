pub mod reader;
pub mod writer;

use core::fmt;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}

impl std::error::Error for Error {}

impl Error {
    #[must_use]
    pub fn from(msg: &str) -> Box<Self> {
        Box::new(Self(msg.into()))
    }
}

pub trait Vertex {
    fn from(vals: &[f64]) -> Self;
    fn get(&self, i: usize) -> f64;
}

pub trait Solution {
    fn from(vals: &[f64]) -> Self;
    fn get(&self, i: usize) -> f64;
}

pub trait Element {
    fn from(vals: &[u64]) -> Self;
    fn get(&self, i: usize) -> u64;
}

pub trait Tag {
    fn from(tag: i32) -> Self;
    fn get(&self) -> i32;
}

// Set the log level for tests
#[allow(dead_code)]
fn init_log(level: &str) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

// impl for tests
impl Vertex for [f64; 3] {
    fn from(vals: &[f64]) -> Self {
        assert!(vals.len() == 3);
        [vals[0], vals[1], vals[2]]
    }
    fn get(&self, i: usize) -> f64 {
        self[i]
    }
}

impl Vertex for [f64; 2] {
    fn from(vals: &[f64]) -> Self {
        assert!(vals.len() == 2);
        [vals[0], vals[1]]
    }
    fn get(&self, i: usize) -> f64 {
        self[i]
    }
}

impl Element for [u32; 2] {
    fn from(vals: &[u64]) -> Self {
        assert!(vals.len() == 2);
        [vals[0] as u32, vals[1] as u32]
    }
    fn get(&self, i: usize) -> u64 {
        self[i] as u64
    }
}

impl Element for [u32; 3] {
    fn from(vals: &[u64]) -> Self {
        assert!(vals.len() == 3);
        [vals[0] as u32, vals[1] as u32, vals[2] as u32]
    }
    fn get(&self, i: usize) -> u64 {
        self[i] as u64
    }
}

impl Element for [u32; 4] {
    fn from(vals: &[u64]) -> Self {
        assert!(vals.len() == 4);
        [
            vals[0] as u32,
            vals[1] as u32,
            vals[2] as u32,
            vals[3] as u32,
        ]
    }
    fn get(&self, i: usize) -> u64 {
        self[i] as u64
    }
}

impl Tag for () {
    fn from(_tag: i32) -> Self {}
    fn get(&self) -> i32 {
        0
    }
}

impl Tag for i16 {
    fn from(tag: i32) -> Self {
        tag as i16
    }
    fn get(&self) -> i32 {
        *self as i32
    }
}

impl Solution for f32 {
    fn from(s: &[f64]) -> Self {
        assert!(s.len() == 1);
        s[0] as f32
    }
    fn get(&self, i: usize) -> f64 {
        assert_eq!(i, 0);
        *self as f64
    }
}

impl Solution for [f32; 2] {
    fn from(s: &[f64]) -> Self {
        assert!(s.len() == 2);
        [s[0] as f32, s[1] as f32]
    }
    fn get(&self, i: usize) -> f64 {
        self[i] as f64
    }
}

impl Solution for [f32; 3] {
    fn from(s: &[f64]) -> Self {
        assert!(s.len() == 3);
        [s[0] as f32, s[1] as f32, s[2] as f32]
    }
    fn get(&self, i: usize) -> f64 {
        self[i] as f64
    }
}

impl Solution for [f32; 6] {
    fn from(s: &[f64]) -> Self {
        assert!(s.len() == 6);
        [
            s[0] as f32,
            s[1] as f32,
            s[2] as f32,
            s[3] as f32,
            s[4] as f32,
            s[5] as f32,
        ]
    }
    fn get(&self, i: usize) -> f64 {
        self[i] as f64
    }
}
