//! Core interfaces for working with Rank-1 Constraint Systems (R1CS), which is
//! an indexed NP relation.

use crate::{NPRelation, Relation};
use ark_std::{cmp::Ordering, marker::PhantomData, vec::Vec};

/// R1CS is an *indexed NP relation*.
/// An index consists of three matrices (A, B, C),
/// while the instance *x* and witness *w* are vectors of field elements
/// such that, for z := (1 || x || w), Az ○ Bz = Cz.
pub struct R1CS<F: Field> {
    f: PhantomData<F>,
}

/// An R1CS index consists of three matrices, as well as the number of instance variables
/// and number of witness variables. The "one" variable is treated as a defacto instance
/// variable for the purposes of counting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintMatrices<F: Field> {
    /// The number of variables that are "public instances" to the constraint
    /// system. This includes the one variable.
    pub num_instance_variables: usize,
    /// The number of variables that are "private witnesses" to the constraint
    /// system.
    pub num_witness_variables: usize,
    /// The number of constraints in the constraint system.
    pub num_constraints: usize,
    /// The number of non_zero entries in the A matrix.
    pub a_num_non_zero: usize,
    /// The number of non_zero entries in the B matrix.
    pub b_num_non_zero: usize,
    /// The number of non_zero entries in the C matrix.
    pub c_num_non_zero: usize,

    /// The A constraint matrix.
    pub a: Matrix<F>,
    /// The B constraint matrix.
    pub b: Matrix<F>,
    /// The C constraint matrix.
    pub c: Matrix<F>,
}

/// An R1CS instance consists of variable assignments to the instance variables.
#[derive(Eq, PartialEq, Debug, Hash, Clone)]
pub struct Instance<F: Field>(pub Vec<F>);

/// An R1CS instance consists of variable assignments to the witness variables.
#[derive(Eq, PartialEq, Debug, Hash, Clone)]
pub struct Witness<F: Field>(pub Vec<F>);

impl<F: Field> Relation for R1CS<F> {
    type Index = ConstraintMatrices<F>;
    type Instance = Instance<F>;
    type Witness = Witness<F>;
}

impl<F: Field> NPRelation for R1CS<F> {
    fn check_membership(
        index: &Self::Index,
        instance: &Self::Instance,
        witness: &Self::Witness,
    ) -> bool {
        let mut instance = instance.0.to_vec();
        instance.insert(0, F::one());
        // The number of instance variables does not match.
        if instance.len() != index.num_instance_variables + 1 {
            return false;
        }
        // The number of witness variables does not match.
        if witness.0.len() != index.num_witness_variables {
            return false;
        }

        // Let z = instance || witness. Check that Az ○ Bz = Cz, where ○ is the Hadamard product.
        // The Hadamard product of two vectors is the product of them entry-wise, so
        // [ 2 ]   [ 3 ]   [ 6 ]
        // [ 3 ] ○ [ 6 ] = [ 18 ]
        // [ 4 ]   [ 7 ]   [ 28 ]
        cfg_iter!(&index.a)
            .zip(&index.b)
            .zip(&index.c)
            .all(|((a_row, b_row), c_row)| {
                let a = inner_product(&a_row, index.num_instance_variables, &instance, &witness.0);
                let b = inner_product(&b_row, index.num_instance_variables, &instance, &witness.0);
                let c = inner_product(&c_row, index.num_instance_variables, &instance, &witness.0);
                a * b == c
            })
    }
}

// Compute the inner product of `row` with `instance.concat(witness)`.
fn inner_product<F: Field>(
    row: &[(F, usize)],
    num_instance_variables: usize,
    instance: &[F],
    witness: &[F],
) -> F {
    let mut acc = F::zero();
    for &(ref coeff, i) in row {
        let tmp = if i < num_instance_variables {
            instance[i]
        } else {
            witness[i - num_instance_variables]
        };
        acc += &(if coeff.is_one() { tmp } else { tmp * coeff });
    }
    acc
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#[macro_use]
mod impl_lc;
mod constraint_system;
mod error;
#[cfg(feature = "std")]
mod trace;

#[cfg(feature = "std")]
pub use crate::r1cs::trace::{ConstraintLayer, ConstraintTrace, TraceStep, TracingMode};

pub use tracing::info_span;

pub use ark_ff::{Field, ToConstraintField};
pub use constraint_system::{
    ConstraintGenerator, ConstraintSystem, ConstraintSystemRef, Namespace, OptimizationGoal,
    SynthesisMode,
};
pub use error::SynthesisError;
/// A result type specialized to `SynthesisError`.
pub type Result<T> = core::result::Result<T, SynthesisError>;

/// A sparse representation of constraint matrices.
pub type Matrix<F> = Vec<Vec<(F, usize)>>;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
/// An opaque counter for symbolic linear combinations.
pub struct LcIndex(usize);

/// Represents the different kinds of variables present in a constraint system.
#[derive(Copy, Clone, PartialEq, Debug, Eq)]
pub enum Variable {
    /// Represents the "zero" constant.
    Zero,
    /// Represents of the "one" constant.
    One,
    /// Represents a public instance variable.
    Instance(usize),
    /// Represents a private witness variable.
    Witness(usize),
    /// Represents of a linear combination.
    SymbolicLc(LcIndex),
}

/// A linear combination of variables according to associated coefficients.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LinearCombination<F: Field>(pub Vec<(F, Variable)>);

/// Generate a `Namespace` with name `name` from `ConstraintSystem` `cs`.
/// `name` must be a `&'static str`.
#[macro_export]
macro_rules! ns {
    ($cs:expr, $name:expr) => {{
        let span = $crate::r1cs::info_span!(target: "r1cs", $name);
        let id = span.id();
        let _enter_guard = span.enter();
        core::mem::forget(_enter_guard);
        core::mem::forget(span);
        $crate::r1cs::Namespace::new($cs.clone(), id)
    }};
}

impl Variable {
    /// Is `self` the zero variable?
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self, Variable::Zero)
    }

    /// Is `self` the one variable?
    #[inline]
    pub fn is_one(&self) -> bool {
        matches!(self, Variable::One)
    }

    /// Is `self` an instance variable?
    #[inline]
    pub fn is_instance(&self) -> bool {
        matches!(self, Variable::Instance(_))
    }

    /// Is `self` a witness variable?
    #[inline]
    pub fn is_witness(&self) -> bool {
        matches!(self, Variable::Witness(_))
    }

    /// Is `self` a linear combination?
    #[inline]
    pub fn is_lc(&self) -> bool {
        matches!(self, Variable::SymbolicLc(_))
    }

    /// Get the `LcIndex` in `self` if `self.is_lc()`.
    #[inline]
    pub fn get_lc_index(&self) -> Option<LcIndex> {
        match self {
            Variable::SymbolicLc(index) => Some(*index),
            _ => None,
        }
    }

    /// Returns `Some(usize)` if `!self.is_lc()`, and `None` otherwise.
    #[inline]
    pub fn get_index_unchecked(&self, witness_offset: usize) -> Option<usize> {
        match self {
            // The one variable always has index 0
            Variable::One => Some(0),
            Variable::Instance(i) => Some(*i),
            Variable::Witness(i) => Some(witness_offset + *i),
            _ => None,
        }
    }
}

impl PartialOrd for Variable {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Variable::*;
        match (self, other) {
            (Zero, Zero) => Some(Ordering::Equal),
            (One, One) => Some(Ordering::Equal),
            (Zero, _) => Some(Ordering::Less),
            (One, _) => Some(Ordering::Less),
            (_, Zero) => Some(Ordering::Greater),
            (_, One) => Some(Ordering::Greater),

            (Instance(i), Instance(j)) | (Witness(i), Witness(j)) => i.partial_cmp(j),
            (Instance(_), Witness(_)) => Some(Ordering::Less),
            (Witness(_), Instance(_)) => Some(Ordering::Greater),

            (SymbolicLc(i), SymbolicLc(j)) => i.partial_cmp(j),
            (_, SymbolicLc(_)) => Some(Ordering::Less),
            (SymbolicLc(_), _) => Some(Ordering::Greater),
        }
    }
}

impl Ord for Variable {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
