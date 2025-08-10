use std::{ffi::c_void, num::NonZero, ptr::NonNull};

use nix::{
    fcntl::OFlag,
    sys::{
        mman,
        stat::{self, Mode},
    },
    unistd,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShmQueueError {
    Full,
    Empty,
}

impl std::error::Error for ShmQueueError {}

impl std::fmt::Display for ShmQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShmQueueError::Full => write!(f, "Queue is full"),
            ShmQueueError::Empty => write!(f, "Queue is empty"),
        }
    }
}

pub struct ShmQueueMeta {
    capacity: usize,
    head: usize,
    tail: usize,
    data_offset: usize,
    ready: bool,
}

pub struct ShmQueue<'a, T> {
    owned: bool,
    name: String,
    mmap: (NonNull<c_void>, usize),
    meta: &'a mut ShmQueueMeta,
    data: &'a mut [u8],
    _phantom: std::marker::PhantomData<T>,
}

pub trait ShmBytes {
    const SIZE: usize;

    fn as_bytes(&self) -> impl Iterator<Item = u8> + '_;

    fn from_bytes(bytes: &[u8]) -> Self;
}

impl<'a, T: ShmBytes> ShmQueue<'a, T> {
    pub fn new(name: &str, capacity: usize) -> Self {
        let meta_vs_data = std::mem::size_of::<ShmQueueMeta>() / T::SIZE;
        let len = (capacity + 1 + meta_vs_data) * T::SIZE;
        let shm = mman::shm_open(
            name,
            OFlag::O_CREAT | OFlag::O_RDWR | OFlag::O_EXCL,
            Mode::S_IRUSR | Mode::S_IWUSR,
        )
        .unwrap();

        unistd::ftruncate(&shm, len as _).unwrap();
        let mmap = unsafe {
            mman::mmap(
                None,
                NonZero::new(len).unwrap(),
                mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE,
                mman::MapFlags::MAP_SHARED,
                &shm,
                0,
            )
            .unwrap()
            .cast::<u8>()
        };

        let meta = unsafe { &mut *(mmap.as_ptr() as *mut ShmQueueMeta) };
        meta.capacity = capacity;
        meta.head = 0;
        meta.tail = 0;
        meta.data_offset = 1 + meta_vs_data;
        meta.ready = true;

        let data = unsafe {
            std::slice::from_raw_parts_mut(
                mmap.as_ptr().add(meta.data_offset * T::SIZE),
                capacity * T::SIZE,
            )
        };

        Self {
            owned: true,
            name: name.to_string(),
            mmap: (mmap.cast(), len),
            meta,
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn open(name: &str) -> Option<Self> {
        let Ok(shm) = mman::shm_open(name, OFlag::O_RDWR, Mode::S_IRUSR | Mode::S_IWUSR) else {
            return None;
        };

        let len = stat::fstat(&shm).unwrap().st_size as usize;
        if len < std::mem::size_of::<ShmQueueMeta>() {
            return None;
        }

        let mmap = unsafe {
            mman::mmap(
                None,
                NonZero::new(len).unwrap(),
                mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE,
                mman::MapFlags::MAP_SHARED,
                &shm,
                0,
            )
            .unwrap()
            .cast::<u8>()
        };

        let meta = unsafe { &mut *(mmap.as_ptr() as *mut ShmQueueMeta) };
        while !unsafe { std::ptr::read_volatile(&meta.ready) } {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        let data = unsafe {
            std::slice::from_raw_parts_mut(
                mmap.as_ptr().add(meta.data_offset * T::SIZE),
                meta.capacity * T::SIZE,
            )
        };

        Some(Self {
            owned: false,
            name: name.to_string(),
            mmap: (mmap.cast(), len),
            meta,
            data,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn capacity(&self) -> usize {
        self.meta.capacity
    }

    pub fn send(&mut self, item: &T) -> Result<(), ShmQueueError> {
        let meta = unsafe { std::ptr::read_volatile(self.meta) };
        if meta.head == (meta.tail + 1) % meta.capacity {
            return Err(ShmQueueError::Full);
        }

        for (i, byte) in item.as_bytes().enumerate() {
            self.data[meta.tail * T::SIZE + i] = byte;
        }

        self.meta.tail = (self.meta.tail + 1) % self.meta.capacity;
        Ok(())
    }

    pub fn recv(&mut self) -> Result<T, ShmQueueError> {
        let meta = unsafe { std::ptr::read_volatile(self.meta) };
        if meta.head == meta.tail {
            return Err(ShmQueueError::Empty);
        }

        let item = T::from_bytes(&self.data[meta.head * T::SIZE..(meta.head + 1) * T::SIZE]);
        self.meta.head = (self.meta.head + 1) % self.meta.capacity;
        Ok(item)
    }
}

impl<'a, T> Drop for ShmQueue<'a, T> {
    fn drop(&mut self) {
        let (addr, len) = self.mmap;
        unsafe { mman::munmap(addr, len).unwrap() };
        if self.owned {
            let _ = mman::shm_unlink(self.name.as_str());
        }
    }
}

unsafe impl<'a, T> Send for ShmQueue<'a, T> {}

#[cfg(test)]
mod test {
    use super::*;

    impl ShmBytes for i32 {
        const SIZE: usize = std::mem::size_of::<i32>();

        fn as_bytes(&self) -> impl Iterator<Item = u8> + '_ {
            self.to_le_bytes().into_iter()
        }

        fn from_bytes(bytes: &[u8]) -> Self {
            i32::from_le_bytes(bytes.try_into().unwrap())
        }
    }

    #[test]
    fn test_queue() {
        let mut sender = ShmQueue::new("test_queue", 10);
        assert_eq!(sender.capacity(), 10);

        assert!(sender.send(&1).is_ok());
        assert!(sender.send(&2).is_ok());
        assert_eq!(sender.recv(), Ok(1));
        assert_eq!(sender.recv(), Ok(2));
        assert_eq!(sender.recv(), Err(ShmQueueError::Empty));

        let mut receiver = ShmQueue::open("test_queue").unwrap();
        assert_eq!(receiver.capacity(), 10);

        assert!(receiver.send(&3).is_ok());
        assert_eq!(receiver.recv(), Ok(3));

        assert!(sender.send(&4).is_ok());
        assert_eq!(receiver.recv(), Ok(4));

        let mut counter = 0;
        while sender.send(&5).is_ok() {
            counter += 1;
        }
        assert_eq!(sender.send(&5), Err(ShmQueueError::Full));

        while let Ok(item) = receiver.recv() {
            counter -= 1;
            assert_eq!(item, 5);
        }
        assert_eq!(counter, 0);
        assert_eq!(receiver.recv(), Err(ShmQueueError::Empty));
    }

    #[test]
    fn test_raii() {
        let owner = ShmQueue::<i32>::new("test_queue", 10);
        assert!(ShmQueue::<i32>::open("test_queue").is_some());
        drop(owner);
        assert!(ShmQueue::<i32>::open("test_queue").is_none());
    }
}
