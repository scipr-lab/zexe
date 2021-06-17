//! This crate contains traits that define the basic behaviour of SNARKs.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    missing_docs
)]
#![forbid(unsafe_code)]

use ark_ff::ToBytes;
use ark_relations::NPRelation;
use ark_std::rand::{CryptoRng, RngCore};
use core::fmt::Debug;

/// Specialized interface for R1CS-based SNARKs.
pub mod r1cs;

/// The basic functionality for a SNARK.
pub trait SNARK<R: NPRelation> {
    /// The information required by the prover to produce a proof for a specific
    /// circuit *C*.
    type ProvingKey: Clone;

    /// The information required by the verifier to check a proof for a specific
    /// circuit *C*.
    type VerifyingKey: Clone + ToBytes;

    /// The proof output by the prover.
    type Proof: Clone;

    /// This contains the verification key, but preprocessed to enable faster
    /// verification.
    type ProcessedVerifyingKey: Clone;

    /// Errors encountered during setup, proving, or verification.
    type Error: 'static + ark_std::error::Error;

    /// Generates a proof of satisfaction of the arithmetic circuit C (specified
    /// as R1CS constraints).
    fn prove<Rng: RngCore + CryptoRng>(
        pk: &Self::ProvingKey,
        index: &Option<R::Index>,
        instance: &R::Instance,
        witness: &R::Witness,
        rng: &mut Rng,
    ) -> Result<Self::Proof, Self::Error>;

    /// Checks that `proof` is a valid proof of the satisfaction of circuit
    /// encoded in `vk`, with respect to the public input `public_input`,
    /// specified as R1CS constraints.
    fn verify(
        vk: &Self::VerifyingKey,
        instance: &R::Instance,
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error> {
        let pvk = Self::process_vk(vk)?;
        Self::verify_with_processed_vk(&pvk, instance, proof)
    }

    /// Preprocesses `vk` to enable faster verification.
    fn process_vk(vk: &Self::VerifyingKey) -> Result<Self::ProcessedVerifyingKey, Self::Error>;

    /// Checks that `proof` is a valid proof of the satisfaction of circuit
    /// encoded in `pvk`, with respect to the public input `public_input`,
    /// specified as R1CS constraints.
    fn verify_with_processed_vk(
        pvk: &Self::ProcessedVerifyingKey,
        instance: &R::Instance,
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error>;
}

/// A SNARK with (only) circuit-specific setup.
pub trait CircuitSpecificSetupSNARK<R: NPRelation>: SNARK<R> {
    /// The setup algorithm for circuit-specific SNARKs.
    fn circuit_specific_setup<Rng: RngCore + CryptoRng>(
        index: &R::Index,
        rng: &mut Rng,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error>;
}

/// A helper type for universal-setup SNARKs, which must infer their computation
/// size bounds.
pub enum IndexingError<Bound, E> {
    /// The provided universal public parameters were insufficient to encode
    /// the given circuit.
    NeedLargerBound(Bound),
    /// Other errors occurred during indexing.
    Other(E),
}

/// A SNARK with universal setup. That is, a SNARK where the trusted setup is
/// circuit-independent.
pub trait UniversalSetupSNARK<R: NPRelation>: SNARK<R> {
    /// Specifies how to bound the size of public parameters required to
    /// generate the index proving and verification keys for a given
    /// circuit.
    ///
    /// For example, for SNARKs that rely on polynomial commitments, this would
    /// be the maximum degree of polynomials required to prove a given
    /// instance.
    type IndexBound: Clone + Default + Debug;
    /// Specifies the type of universal public parameters.
    type PublicParameters: Clone + Debug;

    /// Specifies the bound size that is necessary and sufficient to
    /// generate public parameters for `index`.
    fn bound_for_index(index: &R::Index) -> Self::IndexBound;

    /// Specifies how to bound the size of public parameters required to
    /// generate the index proving and verification keys for a given
    /// circuit.
    fn universal_setup<Rng: RngCore + CryptoRng>(
        compute_bound: &Self::IndexBound,
        rng: &mut Rng,
    ) -> Result<Self::PublicParameters, Self::Error>;

    /// Indexes the public parameters according to the circuit `circuit`, and
    /// outputs circuit-specific proving and verification keys.
    ///
    /// This is a *deterministic* method.
    fn index(
        pp: &Self::PublicParameters,
        index: &R::Index,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), IndexingError<Self::IndexBound, Self::Error>>;
}

impl<R: NPRelation, S> CircuitSpecificSetupSNARK<R> for S
where
    S: UniversalSetupSNARK<R>,
{
    /// The setup algorithm for circuit-specific SNARKs.
    fn circuit_specific_setup<Rng: RngCore + CryptoRng>(
        index: &R::Index,
        rng: &mut Rng,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error> {
        let bound = Self::bound_for_index(index);
        let pp = Self::universal_setup(&bound, rng)?;
        Self::index(&pp, index).map_err(|e| {
            if let IndexingError::Other(e) = e {
                e
            } else {
                panic!("`bound_for_index` returned bound that is insufficient for indexing")
            }
        })
    }
}
