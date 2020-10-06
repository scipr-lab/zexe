use crate::mnt4_753::{self, Fq, Fq2, Fr, FQ_ZERO, G1_COEFF_A_NON_RESIDUE};
use algebra_core::{
    biginteger::BigInteger768,
    curves::{
        mnt4,
        mnt4::MNT4Parameters,
        models::{ModelParameters, SWModelParameters},
    },
    field_new, impl_scalar_mul_kernel,
};

pub type G2Affine = mnt4::G2Affine<mnt4_753::Parameters>;
pub type G2Projective = mnt4::G2Projective<mnt4_753::Parameters>;
pub type G2Prepared = mnt4::G2Prepared<mnt4_753::Parameters>;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Parameters;

impl ModelParameters for Parameters {
    type BaseField = Fq2;
    type ScalarField = Fr;
}

/// MUL_BY_A_C0 = NONRESIDUE * COEFF_A
#[rustfmt::skip]
pub const MUL_BY_A_C0: Fq = G1_COEFF_A_NON_RESIDUE;

/// MUL_BY_A_C1 = NONRESIDUE * COEFF_A
#[rustfmt::skip]
pub const MUL_BY_A_C1: Fq = G1_COEFF_A_NON_RESIDUE;

impl_scalar_mul_kernel!(mnt4_753, "mnt4_753", g2, G2Projective);

impl SWModelParameters for Parameters {
    const COEFF_A: Fq2 = mnt4_753::Parameters::TWIST_COEFF_A;
    // B coefficient of MNT4-753 G2 =
    // ```
    // mnt4753_twist_coeff_b = mnt4753_Fq2(mnt4753_Fq::zero(),
    //                                  mnt4753_G1::coeff_b * mnt4753_Fq2::non_residue);
    // non_residue = mnt4753_Fq2::non_residue = mnt4753_Fq("13");
    //  = (ZERO, G1_B_COEFF * NON_RESIDUE);
    //  =
    //  (0, 39196523001581428369576759982967177918859161321667605855515469914917622337081756705006832951954384669101573360625169461998308377011601613979275218690841934572954991361632773738259652003389826903175898479855893660378722437317212)
    // ```
    #[rustfmt::skip]
    const COEFF_B: Fq2 = field_new!(Fq2,
        FQ_ZERO,
        field_new!(Fq, BigInteger768([
            15129916544657421551,
            11332543254671606602,
            11913830318987286849,
            13905314883394440110,
            16479690325073358448,
            14869098639251228898,
            10663986895980443550,
            10768989312009479656,
            9469728929095040349,
            4512954369775881939,
            8788997129423430122,
            459763387588954,
        ])),
    );

    /// COFACTOR =
    /// 41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888049094905534395567574915333486969589229856772141392370549616644545554517640527237829320384324374366385444967219201
    #[rustfmt::skip]
    const COFACTOR: &'static [u64] = &[
        16436257212445032449,
        8690275530472745198,
        17315389657026393162,
        1645397558963170979,
        3544984605440726586,
        12665092767997125024,
        11083680675069097885,
        575819899841080717,
        6825179918269667443,
        13256793349531086829,
        1162650133526138285,
        497830423872529,
    ];

    /// COFACTOR^(-1) mod r =
    /// 102345604409665481004734934052318066391634848395005988700111949231215905051467807945653833683883449458834877235200
    #[rustfmt::skip]
    const COFACTOR_INV: Fr = field_new!(Fr, BigInteger768([
        1879390364380281898,
        7926794171490610432,
        6437239504245874253,
        16688141022047191858,
        17059416847145768464,
        15065047661241262516,
        16537575340937777321,
        3324411942044534547,
        3878293904770657570,
        18116939243856833744,
        7557533897589069385,
        78370361203778,
    ]));

    /// AFFINE_GENERATOR_COEFFS = (G2_GENERATOR_X, G2_GENERATOR_Y)
    const AFFINE_GENERATOR_COEFFS: (Self::BaseField, Self::BaseField) =
        (G2_GENERATOR_X, G2_GENERATOR_Y);

    #[inline(always)]
    fn mul_by_a(elt: &Fq2) -> Fq2 {
        field_new!(Fq2, MUL_BY_A_C0 * &elt.c0, MUL_BY_A_C1 * &elt.c1,)
    }

    fn scalar_mul_kernel(
        ctx: &Context,
        grid: impl Into<Grid>,
        block: impl Into<Block>,
        table: *const G2Projective,
        exps: *const u8,
        out: *mut G2Projective,
        n: isize,
    ) -> error::Result<()> {
        scalar_mul(ctx, grid, block, (table, exps, out, n))?;
        Ok(())
    }
}

const G2_GENERATOR_X: Fq2 = field_new!(Fq2, G2_GENERATOR_X_C0, G2_GENERATOR_X_C1);
const G2_GENERATOR_Y: Fq2 = field_new!(Fq2, G2_GENERATOR_Y_C0, G2_GENERATOR_Y_C1);

// Generator of G2
// These are two Fq elements each because X and Y (and Z) are elements of Fq^2
// X = 29483965110843144675703364744708836524643960105538608078862508397502447349913068434941060515343254862580437318493682762113105361632548148204806052114008731372757389645383891982211245013965175213456066452587869519098351487925167,
// 19706011319630172391076079624799753948158506771222147486237995321925443331396169656568431378974558350664383559981183980668976846806019030432389169137953988990802000581078994008283967768348275973921598166274857631001635633631000,
// Y = 39940152670760519653940320314827327941993141403708338666925204282084477074754642625849927569427860786384998614863651207257467076192649385174108085803168743803491780568503369317093191101779534035377266300185099318717465441820654,
// 17608637424964395737041291373756657139607306440193731804102457011726690702169238966996114255971643893157857311132388792357391583164125870757541009035041469463366528798593952884745987697403056488744603829437448927398468360797245,
#[rustfmt::skip]
pub const G2_GENERATOR_X_C0: Fq = field_new!(Fq, BigInteger768([
    7263636080534048301,
    7643227961769035653,
    15787777614640869937,
    17661317895964274771,
    4142647779394287783,
    15064949873659932676,
    4579259080596351332,
    2207443675339702626,
    5738725620118622838,
    4338467638707299712,
    6558861849926282439,
    341078935870328
]));

#[rustfmt::skip]
pub const G2_GENERATOR_X_C1: Fq = field_new!(Fq, BigInteger768([
    13073882729051113867,
    195909455108318710,
    10527070038778509320,
    16605710222187477118,
    10442518358308209073,
    7776589986153052354,
    16034091384000651523,
    17429232381273855185,
    512853344493546034,
    7982076214836075255,
    3601416800138513610,
    399028695285184
]));

#[rustfmt::skip]
pub const G2_GENERATOR_Y_C0: Fq = field_new!(Fq, BigInteger768([
    4867672648970469422,
    2453112364260322863,
    11918672197974895143,
    3923647310180624143,
    12707123323825700670,
    15781895092544451511,
    17747112377690960911,
    16511994611001933567,
    15360620366665804029,
    10252080383738480571,
    5722654046339742760,
    316853130349807
]));

#[rustfmt::skip]
pub const G2_GENERATOR_Y_C1: Fq = field_new!(Fq, BigInteger768([
    17322957246732430708,
    11825815087694023697,
    7654216682602683900,
    13544637981229618042,
    17057060382219081849,
    4038731408172002692,
    6631723222530012253,
    10585926166286435412,
    16050667328028620117,
    16598483946296156500,
    11771818132648686020,
    1230430296095
]));
