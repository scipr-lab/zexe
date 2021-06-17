//! Core interface for working with various relations that are useful in
//! zkSNARKs. At the moment, we only implement APIs for working with Rank-1
//! Constraint Systems (R1CS).

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    missing_docs
)]
#![deny(unsafe_code)]

#[macro_use]
extern crate ark_std;

pub mod r1cs;

/// An *indexed relation* is a set of triples of the form `(index, instance, witness)`.
pub trait Relation {
    /// The index is a "large" but static part of the triple. Examples include
    /// the circuit in the circuit satisfiability relation, and constraint
    /// matrices in the R1CS relation.
    type Index: Eq;
    /// The instance is a "small" part of the triple. Like the index, it is publicly known.
    type Instance: Eq;
    /// The witness is a "large" but private part of the triple.
    type Witness: Eq;
}

/// An *indexed NP relation* is a relation with an efficient membership check.
pub trait NPRelation: Relation {
    /// Checks whether the triple `(index, instance, witness)` is a member of the relation.
    fn check_membership(
        index: &Self::Index,
        instance: &Self::Instance,
        witness: &Self::Witness,
    ) -> bool;
}
