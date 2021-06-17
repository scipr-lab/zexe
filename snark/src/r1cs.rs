use crate::{CircuitSpecificSetupSNARK, IndexingError, UniversalSetupSNARK, SNARK};
use ark_ff::Field;
use ark_relations::r1cs::{ConstraintGenerator, ConstraintMatrices, Instance, Witness, R1CS};
use ark_std::rand::{CryptoRng, RngCore};

/// A [`SNARKForR1CS`] is a [`SNARK`] for the [`R1CS`] relation.
pub trait SNARKForR1CS<F: Field>: SNARK<R1CS<F>> {
    /// Does the proving algorithm explicitly require the matrices, or is it stored
    /// in the proving key?
    const PROVING_REQUIRES_MATRICES: bool;

    /// Generate inputs for the SNARK indexer from [`cs`].
    /// These inputs consist of the constraint matrices.
    fn indexer_inputs<CG: ConstraintGenerator<F>>(
        cs: &CG,
    ) -> Result<ConstraintMatrices<F>, Self::Error>;

    /// Generate inputs for the SNARK prover from [`cs`].
    /// These inputs consist of the instance and witness. Additionally,
    /// if `Self::PROVING_REQUIRES_MATRICES == true`, then this method returns
    /// `Some(index)` as well.
    fn prover_inputs<CG: ConstraintGenerator<F>>(
        cs: &CG,
    ) -> Result<(Option<ConstraintMatrices<F>>, Instance<F>, Witness<F>), Self::Error>;

    /// Generate inputs for the SNARK verifier from [`cs`].
    /// This input consists of the instance.
    fn verifier_inputs<CG: ConstraintGenerator<F>>(cs: &CG) -> Result<Instance<F>, Self::Error>;

    /// Indexes the public parameters according to the circuit `circuit`, and
    /// outputs circuit-specific proving and verification keys.
    fn circuit_specific_setup_with_cs<CG: ConstraintGenerator<F>, Rng: RngCore + CryptoRng>(
        c: &CG,
        rng: &mut Rng,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error>
    where
        Self: CircuitSpecificSetupSNARK<R1CS<F>>,
    {
        let index = Self::indexer_inputs(c)?;
        Self::circuit_specific_setup(&index, rng)
    }

    /// Indexes the public parameters according to the circuit `circuit`, and
    /// outputs circuit-specific proving and verification keys.
    ///
    /// This is a *deterministic* method.
    fn index_with_cs<CG: ConstraintGenerator<F>>(
        pp: &Self::PublicParameters,
        c: &CG,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), IndexingError<Self::IndexBound, Self::Error>>
    where
        Self: UniversalSetupSNARK<R1CS<F>>,
    {
        let index = Self::indexer_inputs(c).map_err(IndexingError::Other)?;
        Self::index(pp, &index)
    }

    /// Generates a proof of satisfaction of the constraint system induced by [`c`].
    fn prove_with_cs<CG: ConstraintGenerator<F>, Rng: RngCore + CryptoRng>(
        pk: &Self::ProvingKey,
        c: &CG,
        rng: &mut Rng,
    ) -> Result<Self::Proof, Self::Error> {
        let (index, instance, witness) = Self::prover_inputs(c)?;
        Self::prove(pk, &index, &instance, &witness, rng)
    }

    /// Verify that [`proof`] is a valid proof with respect to [`vk`] and to the
    /// instance induced by [`c`].
    fn verify_with_cs<CG: ConstraintGenerator<F>>(
        vk: &Self::VerifyingKey,
        c: &CG,
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error> {
        let instance = Self::verifier_inputs(c)?;
        Self::verify(vk, &instance, proof)
    }

    /// Checks that [`proof`] is a valid proof of with respect to [`pvk`] and
    /// the instance induced by [`cs`].
    fn verify_with_cs_and_processed_vk<CG: ConstraintGenerator<F>>(
        pvk: &Self::ProcessedVerifyingKey,
        c: &CG,
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error> {
        let instance = Self::verifier_inputs(c)?;
        Self::verify_with_processed_vk(pvk, &instance, proof)
    }
}
