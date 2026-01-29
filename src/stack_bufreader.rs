use std::io::{BufRead, Error, ErrorKind, IoSliceMut, Read, Seek, SeekFrom};
use std::mem::MaybeUninit;
use std::{cmp, fmt, io, ptr};

/// The `StackBufReader<R, N>` struct adds buffering to any reader.
///
/// See [`BufReader`][std::io::BufReader] for more details.
///
/// # Examples
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::fs::File;
/// use stack_buf_io::StackBufReader;
///
/// fn main() -> std::io::Result<()> {
///     let f = File::open("log.txt")?;
///     let mut reader = StackBufReader::<_, 4096>::new(f);
///
///     let mut line = String::new();
///     let len = reader.read_line(&mut line)?;
///     println!("First line is {} bytes long", len);
///     Ok(())
/// }
/// ```
pub struct StackBufReader<R: ?Sized + Read, const N: usize = 4096> {
    buf: [MaybeUninit<u8>; N],
    pos: usize,
    filled: usize,
    initialized: bool,
    reader: R
}

impl<R: Read, const N: usize> StackBufReader<R, N> {
    /// Creates a new `StackBufReader<R, N>`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use stack_buf_io::StackBufReader;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("log.txt")?;
    ///     let reader = StackBufReader::<_, 4096>::new(f);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn new(reader: R) -> StackBufReader<R, N> {
        StackBufReader {
            buf: [MaybeUninit::uninit(); N],
            pos: 0,
            filled: 0,
            initialized: false,
            reader
        }
    }
}

impl<R: ?Sized + Read, const N: usize> StackBufReader<R, N> {
    /// Gets a reference to the underlying reader.
    ///
    /// It is inadvisable to directly read from the underlying reader.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use stack_buf_io::StackBufReader;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f1 = File::open("log.txt")?;
    ///     let reader = StackBufReader::new(f1);
    ///
    ///     let f2 = reader.get_ref();
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// It is inadvisable to directly read from the underlying reader.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use stack_buf_io::StackBufReader;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f1 = File::open("log.txt")?;
    ///     let mut reader = StackBufReader::new(f1);
    ///
    ///     let f2 = reader.get_mut();
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Returns a reference to the internally buffered data.
    ///
    /// Unlike [`fill_buf`], this will not attempt to fill the buffer if it is empty.
    ///
    /// [`fill_buf`]: BufRead::fill_buf
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::{BufRead};
    /// use std::fs::File;
    /// use stack_buf_io::StackBufReader;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("log.txt")?;
    ///     let mut reader = StackBufReader::new(f);
    ///     assert!(reader.buffer().is_empty());
    ///
    ///     if reader.fill_buf()?.len() > 0 {
    ///         assert!(!reader.buffer().is_empty());
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn buffer(&self) -> &[u8] {
        // SAFETY: self.pos and self.filled are valid, and self.filled >= self.pos, and
        // that region is initialized because those are all invariants of this type.
        unsafe { self.buf.get_unchecked(self.pos..self.filled).assume_init_ref() }
    }

    /// Returns the number of bytes the internal buffer can hold at once.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::{BufRead};
    /// use std::fs::File;
    /// use stack_buf_io::StackBufReader;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("log.txt")?;
    ///     let mut reader = StackBufReader::new(f);
    ///
    ///     let capacity = reader.capacity();
    ///     let buffer = reader.fill_buf()?;
    ///     assert!(buffer.len() <= capacity);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Unwraps this `StackBufReader<R>`, returning the underlying reader.
    ///
    /// Note that any leftover data in the internal buffer is lost. Therefore,
    /// a following read from the underlying reader may lead to data loss.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use stack_buf_io::StackBufReader;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f1 = File::open("log.txt")?;
    ///     let reader = StackBufReader::new(f1);
    ///
    ///     let f2 = reader.into_inner();
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn into_inner(self) -> R
    where
        R: Sized,
    {
        self.reader
    }

    /// Invalidates all data in the internal buffer.
    #[inline]
    fn discard_buffer(&mut self) {
        self.pos = 0;
        self.filled = 0;
    }

    #[inline]
    pub fn consume(&mut self, amt: usize) {
        self.pos = cmp::min(self.pos + amt, self.filled);
    }

    #[inline]
    pub fn unconsume(&mut self, amt: usize) {
        self.pos = self.pos.saturating_sub(amt);
    }

    #[inline]
    fn consume_with<V>(&mut self, amt: usize, mut visitor: V) -> bool
    where V: FnMut(&[u8]),
    {
        if let Some(claimed) = self.buffer().get(..amt) {
            visitor(claimed);
            // If the indexing into self.buffer() succeeds, amt must be a valid increment.
            self.pos += amt;
            true
        } else {
            false
        }
    }
}

impl<R: ?Sized + Read, const N: usize> StackBufReader<R, N> {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        // If we've reached the end of our internal buffer then we need to fetch
        // some more data from the reader.
        // Branch using `>=` instead of the more correct `==`
        // to tell the compiler that the pos..cap slice is always valid.
        if self.pos >= self.filled {
            debug_assert!(self.pos == self.filled);

            if !self.initialized {
                // SAFETY: 0 is a valid value for MaybeUninit<u8> and the length matches the allocation
                // since it is comes from a slice reference.
                unsafe { ptr::write_bytes::<u8>(&mut self.buf as *mut _ as *mut u8, 0, self.buf.len()) };
                self.initialized = true;
            }

            let result = self.reader.read(unsafe { self.buf.assume_init_mut() });

            self.pos = 0;
            self.filled = 0;
            self.filled = result?;
        }
        Ok(self.buffer())
    }
}

impl<R: ?Sized + Read, const N: usize> Read for StackBufReader<R, N> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.pos == self.filled && buf.len() >= self.capacity() {
            self.discard_buffer();
            return self.reader.read(buf);
        }
        let mut rem = self.fill_buf()?;
        let nread = rem.read(buf)?;
        self.consume(nread);
        Ok(nread)
    }

    #[inline]
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let total_len = bufs.iter().map(|b| b.len()).sum::<usize>();
        if self.pos == self.filled && total_len >= self.capacity() {
            self.discard_buffer();
            return self.reader.read_vectored(bufs);
        }
        let mut rem = self.fill_buf()?;
        let nread = rem.read_vectored(bufs)?;

        self.consume(nread);
        Ok(nread)
    }

    // The inner reader might have an optimized `read_to_end`. Drain our buffer and then
    // delegate to the inner implementation.
    #[inline]
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let inner_buf = self.buffer();
        buf.try_reserve(inner_buf.len())?;
        buf.extend_from_slice(inner_buf);
        let nread = inner_buf.len();
        self.discard_buffer();
        Ok(nread + self.reader.read_to_end(buf)?)
    }

    // The inner reader might have an optimized `read_to_end`. Drain our buffer and then
    // delegate to the inner implementation.
    #[inline]
    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        // In the general `else` case below we must read bytes into a side buffer, check
        // that they are valid UTF-8, and then append them to `buf`. This requires a
        // potentially large memcpy.
        //
        // If `buf` is empty--the most common case--we can leverage `append_to_string`
        // to read directly into `buf`'s internal byte buffer, saving an allocation and
        // a memcpy.
        if buf.is_empty() {
            // `append_to_string`'s safety relies on the buffer only being appended to since
            // it only checks the UTF-8 validity of new data. If there were existing content in
            // `buf` then an untrustworthy reader (i.e. `self.inner`) could not only append
            // bytes but also modify existing bytes and render them invalid. On the other hand,
            // if `buf` is empty then by definition any writes must be appends and
            // `append_to_string` will validate all of the new bytes.
            unsafe { append_to_string(buf, |b| self.read_to_end(b)) }
        } else {
            // We cannot append our byte buffer directly onto the `buf` String as there could
            // be an incomplete UTF-8 sequence that has only been partially read. We must read
            // everything into a side buffer first and then call `from_utf8` on the complete
            // buffer.
            let mut bytes = Vec::new();
            self.read_to_end(&mut bytes)?;
            let string = core::str::from_utf8(&bytes).map_err(|_| Error::new(ErrorKind::InvalidData, "stream did not contain valid UTF-8"))?;
            *buf += string;
            Ok(string.len())
        }
    }

    // Small read_exacts from a BufReader are extremely common when used with a deserializer.
    // The default implementation calls read in a loop, which results in surprisingly poor code
    // generation for the common path where the buffer has enough bytes to fill the passed-in
    // buffer.
    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        if self.consume_with(buf.len(), |claimed| buf.copy_from_slice(claimed)) {
            return Ok(());
        }

        default_read_exact(self, buf)
    }
}

impl<R: ?Sized + Read, const N: usize> BufRead for StackBufReader<R, N> {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.fill_buf()
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.consume(amt);
    }
}

impl<R: ?Sized + Read + fmt::Debug, const N: usize> fmt::Debug for StackBufReader<R, N>
{
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("BufReader")
            .field("reader", &&self.reader)
            .field(
                "buffer",
                &format_args!("{}/{}", self.filled - self.pos, self.capacity()),
            )
            .finish()
    }
}

impl<R: ?Sized + Read + Seek, const N: usize> Seek for StackBufReader<R, N> {
    /// Seek to an offset, in bytes, in the underlying reader.
    ///
    /// The position used for seeking with <code>[SeekFrom::Current]\(_)</code> is the
    /// position the underlying reader would be at if the `BufReader<R>` had no
    /// internal buffer.
    ///
    /// Seeking always discards the internal buffer, even if the seek position
    /// would otherwise fall within it. This guarantees that calling
    /// [`StackBufReader::into_inner()`] immediately after a seek yields the underlying reader
    /// at the same position.
    ///
    /// To seek without discarding the internal buffer, use [`StackBufReader::seek_relative`].
    ///
    /// See [`std::io::Seek`] for more details.
    ///
    /// Note: In the edge case where you're seeking with <code>[SeekFrom::Current]\(n)</code>
    /// where `n` minus the internal buffer length overflows an `i64`, two
    /// seeks will be performed instead of one. If the second seek returns
    /// [`Err`], the underlying reader will be left at the same position it would
    /// have if you called `seek` with <code>[SeekFrom::Current]\(0)</code>.
    ///
    /// [`std::io::Seek`]: Seek
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let result: u64;
        if let SeekFrom::Current(n) = pos {
            let remainder = (self.filled - self.pos) as i64;
            // it should be safe to assume that remainder fits within an i64 as the alternative
            // means we managed to allocate 8 exbibytes and that's absurd.
            // But it's not out of the realm of possibility for some weird underlying reader to
            // support seeking by i64::MIN so we need to handle underflow when subtracting
            // remainder.
            if let Some(offset) = n.checked_sub(remainder) {
                result = self.reader.seek(SeekFrom::Current(offset))?;
            } else {
                // seek backwards by our remainder, and then by the offset
                self.reader.seek(SeekFrom::Current(-remainder))?;
                self.discard_buffer();
                result = self.reader.seek(SeekFrom::Current(n))?;
            }
        } else {
            // Seeking with Start/End doesn't care about our buffer length.
            result = self.reader.seek(pos)?;
        }
        self.discard_buffer();
        Ok(result)
    }

    /// Returns the current seek position from the start of the stream.
    ///
    /// The value returned is equivalent to `self.seek(SeekFrom::Current(0))`
    /// but does not flush the internal buffer. Due to this optimization the
    /// function does not guarantee that calling `.into_inner()` immediately
    /// afterwards will yield the underlying reader at the same position. Use
    /// [`StackBufReader::seek`] instead if you require that guarantee.
    ///
    /// # Panics
    ///
    /// This function will panic if the position of the inner reader is smaller
    /// than the amount of buffered data. That can happen if the inner reader
    /// has an incorrect implementation of [`Seek::stream_position`], or if the
    /// position has gone out of sync due to calling [`Seek::seek`] directly on
    /// the underlying reader.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::{
    ///     io::{self, BufRead, Seek},
    ///     fs::File,
    /// };
    /// use stack_buf_io::StackBufReader;
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = StackBufReader::new(File::open("foo.txt")?);
    ///
    ///     let before = f.stream_position()?;
    ///     f.read_line(&mut String::new())?;
    ///     let after = f.stream_position()?;
    ///
    ///     println!("The first line was {} bytes long", after - before);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    fn stream_position(&mut self) -> io::Result<u64> {
        let remainder = (self.filled - self.pos) as u64;
        self.reader.stream_position().map(|pos| {
            pos.checked_sub(remainder).expect(
                "overflow when subtracting remaining buffer size from inner stream position",
            )
        })
    }

    /// Seeks relative to the current position.
    ///
    /// If the new position lies within the buffer, the buffer will not be
    /// flushed, allowing for more efficient seeks. This method does not return
    /// the location of the underlying reader, so the caller must track this
    /// information themselves if it is required.
    #[inline]
    fn seek_relative(&mut self, offset: i64) -> io::Result<()> {
        let pos = self.pos as u64;
        if offset < 0 {
            if let Some(_) = pos.checked_sub((-offset) as u64) {
                self.unconsume((-offset) as usize);
                return Ok(());
            }
        } else if let Some(new_pos) = pos.checked_add(offset as u64) {
            if new_pos <= self.filled as u64 {
                self.consume(offset as usize);
                return Ok(());
            }
        }

        self.seek(SeekFrom::Current(offset)).map(drop)
    }
}

// Several `read_to_string` and `read_line` methods in the standard library will
// append data into a `String` buffer, but we need to be pretty careful when
// doing this. The implementation will just call `.as_mut_vec()` and then
// delegate to a byte-oriented reading method, but we must ensure that when
// returning we never leave `buf` in a state such that it contains invalid UTF-8
// in its bounds.
//
// To this end, we use an RAII guard (to protect against panics) which updates
// the length of the string when it is dropped. This guard initially truncates
// the string to the prior length and only after we've validated that the
// new contents are valid UTF-8 do we allow it to set a longer length.
//
// The unsafety in this function is twofold:
//
// 1. We're looking at the raw bytes of `buf`, so we take on the burden of UTF-8
//    checks.
// 2. We're passing a raw buffer to the function `f`, and it is expected that
//    the function only *appends* bytes to the buffer. We'll get undefined
//    behavior if existing bytes are overwritten to have non-UTF-8 data.
unsafe fn append_to_string<F>(buf: &mut String, f: F) -> std::io::Result<usize>
where
    F: FnOnce(&mut Vec<u8>) -> io::Result<usize>,
{
    struct Guard<'a> {
        buf: &'a mut Vec<u8>,
        len: usize,
    }

    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            unsafe {
                self.buf.set_len(self.len);
            }
        }
    }

    let mut g = Guard { len: buf.len(), buf: unsafe { buf.as_mut_vec() } };
    let ret = f(g.buf);

    // SAFETY: the caller promises to only append data to `buf`
    let appended = unsafe { g.buf.get_unchecked(g.len..) };
    if str::from_utf8(appended).is_err() {
        ret.and_then(|_| Err(Error::new(ErrorKind::InvalidData, "stream did not contain valid UTF-8")))
    } else {
        g.len = g.buf.len();
        ret
    }
}

fn default_read_exact<R: Read + ?Sized>(this: &mut R, mut buf: &mut [u8]) -> std::io::Result<()> {
    while !buf.is_empty() {
        match this.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                buf = &mut buf[n..];
            }
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    if !buf.is_empty() { Err(Error::new(ErrorKind::UnexpectedEof, "failed to fill whole buffer")) } else { Ok(()) }
}