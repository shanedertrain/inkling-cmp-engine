// Ported from engine/server/src/sched/scheduler.cpp @ 4171493 (free_pages_, refcount_, release_pages).
//! Refcounted KV page pool. A page with refcount above 1 is shared and never written.

/// Pages `0..total`, each with a refcount; free pages are handed out from the back of the free list.
#[derive(Debug)]
pub(crate) struct PagePool {
    free: Vec<u32>,
    refcount: Vec<u32>,
}

impl PagePool {
    /// All pages free; page 0 is handed out first.
    pub(crate) fn new(pages: u32) -> Self {
        Self {
            free: (0..pages).rev().collect(),
            refcount: vec![0; pages as usize],
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.refcount.len()
    }

    pub(crate) fn free_count(&self) -> usize {
        self.free.len()
    }

    pub(crate) fn refcount(&self, page: u32) -> u32 {
        self.refcount[page as usize]
    }

    pub(crate) fn refcounts(&self) -> &[u32] {
        &self.refcount
    }

    pub(crate) fn free_list(&self) -> &[u32] {
        &self.free
    }

    /// Takes the next free page with refcount 1.
    ///
    /// # Panics
    ///
    /// When no page is free; callers make room first.
    pub(crate) fn take(&mut self) -> u32 {
        let page = self.free.pop().expect("caller made room before taking a page");
        self.refcount[page as usize] = 1;
        page
    }

    /// Adds an owner to a page in use.
    pub(crate) fn share(&mut self, page: u32) {
        self.refcount[page as usize] += 1;
    }

    /// Drops `pages` from the back down to `keep`, freeing pages whose last owner this was.
    pub(crate) fn release(&mut self, pages: &mut Vec<u32>, keep: usize) {
        while pages.len() > keep {
            let page = pages.pop().expect("length checked above");
            let rc = &mut self.refcount[page as usize];
            *rc -= 1;
            if *rc == 0 {
                self.free.push(page);
            }
        }
    }
}
