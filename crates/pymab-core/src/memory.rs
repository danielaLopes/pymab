//! Capacity-aware helpers for reporting owned Rust state memory.

use std::collections::VecDeque;
use std::mem::size_of;

/// Bytes reserved by a vector's heap allocation.
#[must_use]
pub fn vec_heap_bytes<T>(value: &Vec<T>) -> usize {
    value.capacity().saturating_mul(size_of::<T>())
}

/// Bytes reserved by a deque's heap allocation.
#[must_use]
pub fn vec_deque_heap_bytes<T>(value: &VecDeque<T>) -> usize {
    value.capacity().saturating_mul(size_of::<T>())
}

/// Bytes occupied by an owned boxed slice allocation.
#[must_use]
pub fn boxed_slice_heap_bytes<T>(value: &[T]) -> usize {
    value.len().saturating_mul(size_of::<T>())
}

/// Bytes required by a dense matrix stored as scalar elements.
#[must_use]
pub fn dense_matrix_heap_bytes<T>(rows: usize, columns: usize) -> Option<usize> {
    rows.checked_mul(columns)?.checked_mul(size_of::<T>())
}

#[cfg(test)]
mod tests {
    use super::{
        boxed_slice_heap_bytes, dense_matrix_heap_bytes, vec_deque_heap_bytes, vec_heap_bytes,
    };
    use std::collections::VecDeque;
    use std::mem::size_of;

    #[test]
    fn helpers_account_for_reserved_capacity() {
        let values = Vec::<u64>::with_capacity(13);
        let history = VecDeque::<(u64, usize, f64)>::with_capacity(7);
        assert!(vec_heap_bytes(&values) >= 13 * size_of::<u64>());
        assert!(vec_deque_heap_bytes(&history) >= 7 * size_of::<(u64, usize, f64)>());
        assert_eq!(boxed_slice_heap_bytes(&[0_u32; 5]), 5 * size_of::<u32>());
        assert_eq!(
            dense_matrix_heap_bytes::<f64>(3, 4),
            Some(12 * size_of::<f64>())
        );
    }

    #[test]
    fn matrix_accounting_detects_overflow() {
        assert_eq!(dense_matrix_heap_bytes::<u64>(usize::MAX, 2), None);
    }
}
