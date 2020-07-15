use algebra_core::fields::PrimeField;

use crate::commitment::{
    injective_map::{InjectiveMap, PedersenCommCompressor},
    pedersen::{
        constraints::{
            PedersenCommitmentGadget, PedersenCommitmentGadgetParameters, PedersenRandomnessGadget,
        },
        PedersenWindow,
    },
    CommitmentGadget,
};

pub use crate::crh::injective_map::constraints::InjectiveMapGadget;
use algebra_core::groups::Group;
use r1cs_core::{ConstraintSystem, SynthesisError};
use r1cs_std::{groups::GroupGadget, uint8::UInt8};

use core::marker::PhantomData;

pub struct PedersenCommitmentCompressorGadget<G, I, ConstraintF, GG, IG>
where
    G: Group,
    I: InjectiveMap<G>,
    ConstraintF: PrimeField,
    GG: GroupGadget<G, ConstraintF>,
    IG: InjectiveMapGadget<G, I, ConstraintF, GG>,
{
    _compressor: PhantomData<I>,
    _compressor_gadget: PhantomData<IG>,
    _crh: PedersenCommitmentGadget<G, ConstraintF, GG>,
}

impl<G, I, ConstraintF, GG, IG, W> CommitmentGadget<PedersenCommCompressor<G, I, W>, ConstraintF>
    for PedersenCommitmentCompressorGadget<G, I, ConstraintF, GG, IG>
where
    G: Group,
    I: InjectiveMap<G>,
    ConstraintF: PrimeField,
    GG: GroupGadget<G, ConstraintF>,
    IG: InjectiveMapGadget<G, I, ConstraintF, GG>,
    W: PedersenWindow,
{
    type OutputGadget = IG::OutputGadget;
    type ParametersGadget = PedersenCommitmentGadgetParameters<G, W, ConstraintF>;
    type RandomnessGadget = PedersenRandomnessGadget;

    fn check_commitment_gadget<CS: ConstraintSystem<ConstraintF>>(
        mut cs: CS,
        parameters: &Self::ParametersGadget,
        input: &[UInt8],
        r: &Self::RandomnessGadget,
    ) -> Result<Self::OutputGadget, SynthesisError> {
        let result = PedersenCommitmentGadget::<G, ConstraintF, GG>::check_commitment_gadget(
            cs.ns(|| "PedersenComm"),
            parameters,
            input,
            r,
        )?;
        IG::evaluate_map(cs.ns(|| "InjectiveMap"), &result)
    }
}
