use std::{cell::UnsafeCell, collections::HashMap, sync::RwLock};

use bytes::Bytes;
use memmap2::Mmap;
use once_cell::sync::OnceCell;
use safetensors::SafeTensors;

pub struct MemCache {
    map: UnsafeCell<HashMap<String, Bytes>>,
    mu: RwLock<()>,
}

impl Default for MemCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MemCache {
    pub fn new() -> Self {
        Self {
            map: UnsafeCell::new(HashMap::new()),
            mu: RwLock::new(()),
        }
    }

    pub fn contains_key(&self, key: &str) -> bool {
        let _lg = self.mu.read().unwrap();
        let m = unsafe { &(*self.map.get()) };
        m.contains_key(key)
    }

    pub fn get_ref<'a>(&'a self, key: &str) -> Option<&'a [u8]> {
        let _lg = self.mu.read().unwrap();
        let m = unsafe { &(*self.map.get()) };
        Some(m.get(key)?.as_ref())
    }

    pub fn insert(&self, key: &str, value: Bytes) {
        let _wg = self.mu.write().unwrap();
        let m = unsafe { &mut (*self.map.get()) };
        m.insert(key.to_owned(), value);
    }
}

unsafe impl Send for MemCache {}
unsafe impl Sync for MemCache {}

pub struct SafetensorCache<'a> {
    map: UnsafeCell<HashMap<String, SafeTensorWithData<'a>>>,
    lk: RwLock<()>,
}

impl Default for SafetensorCache<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'data> SafetensorCache<'data> {
    pub fn new() -> Self {
        Self {
            map: UnsafeCell::new(HashMap::new()),
            lk: RwLock::new(()),
        }
    }

    pub fn insert(&self, key: &str, value: SafeTensorWithData<'data>) {
        let _lg = self.lk.write().unwrap();
        let m = unsafe { &mut (*self.map.get()) };
        if m.contains_key(key) {
            return;
        }
        m.insert(key.to_string(), value);
    }
    pub fn get(&self, key: &str) -> Option<&SafeTensors<'data>> {
        let _lg = self.lk.read().unwrap();
        let m = unsafe { &(*self.map.get()) };
        let v = m.get(key);
        let k = v.map(|x| x.safetensors());
        k
    }

    pub fn contains_key(&self, key: &str) -> bool {
        let _lg = self.lk.read().unwrap();
        unsafe { (*self.map.get()).contains_key(key) }
    }
}

unsafe impl Send for SafetensorCache<'_> {}
unsafe impl Sync for SafetensorCache<'_> {}

#[derive(Debug)]
pub struct SafeTensorWithData<'data> {
    st: OnceCell<SafeTensors<'data>>,
    mmap: Mmap,
}

impl<'data> SafeTensorWithData<'data> {
    pub fn new(mmap: Mmap) -> Self {
        Self {
            st: OnceCell::new(),
            mmap,
        }
    }
    pub fn safetensors(&'data self) -> &'data SafeTensors<'data> {
        let st = self.st.get_or_init(|| {
            let st = safetensors::SafeTensors::deserialize(&self.mmap).unwrap();
            st
        });
        st
    }
}
impl Drop for SafeTensorWithData<'_> {
    fn drop(&mut self) {
        // panic!("SafeTensorWithData dropped");
    }
}
