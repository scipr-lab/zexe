#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unused_import_braces, trivial_casts, trivial_numeric_casts)]
#![deny(unused_qualifications, variant_size_differences)]
#![deny(
    non_shorthand_field_patterns,
    unused_attributes,
    unused_imports,
    unused_extern_crates
)]
#![deny(
    renamed_and_removed_lints,
    unused_allocation,
    unused_comparisons,
    bare_trait_objects
)]
#![deny(const_err, unused_must_use, unused_mut, unused_unsafe)]
#![forbid(unsafe_code)]

#[cfg(all(test, not(feature = "std")))]
#[macro_use]
extern crate std;

/// this crate needs to be public, cause we expose `to_bytes!` macro
/// see similar issue in [`smallvec#198`]
///
/// [`smallvec#198`]: https://github.com/servo/rust-smallvec/pull/198
#[cfg(not(feature = "std"))]
#[macro_use]
#[doc(hidden)]
pub extern crate alloc;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
#[doc(hidden)]
pub use alloc::{boxed::Box, format, vec, vec::Vec};

#[cfg(feature = "std")]
#[allow(unused_imports)]
#[doc(hidden)]
pub use std::{boxed::Box, format, vec, vec::Vec};

pub use algebra_core::*;

#[cfg(feature = "bls12_377")]
pub mod bls12_377;
#[cfg(feature = "bls12_377")]
pub use bls12_377::Bls12_377;

#[cfg(feature = "bls12_381")]
pub mod bls12_381;
#[cfg(feature = "bls12_381")]
pub use bls12_381::Bls12_381;

#[cfg(feature = "bls12_377")]
pub mod edwards_bls12;

#[cfg(feature = "bls12_381")]
pub mod jubjub;

#[cfg(feature = "mnt6")]
pub mod mnt6;
#[cfg(feature = "mnt6")]
pub use mnt6::MNT6;

#[cfg(feature = "sw6")]
pub mod sw6;
#[cfg(feature = "sw6")]
pub use  sw6::SW6;

#[cfg(feature = "sw6")]
pub mod edwards_sw6;


#[cfg(test)]
pub(crate) mod tests;
