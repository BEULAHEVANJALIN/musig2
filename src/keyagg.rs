use secp256k1::{field::Field, point::Point, scalar::Scalar};
use sha2::{Digest, Sha256};


// Hashes a list of public keys into a 32-byte array using SHA-256
pub fn hash_keys<T: Field + Clone + PartialEq>(pubkeys: &[Point<T>]) -> [u8; 32] {
    let mut h = Sha256::new();
    for P in pubkeys {
        h.update(P.to_bytes());
    }
    h.finalize().into()
}

/// Compute coefficient a_i = H(L‖P_i) mod n, with the "second key = 1" optimization.
/// Every a_i should be deterministically derived from the entire key list L and the signer’s own key, 
/// so that no participant can tweak their weight later.
pub fn key_coef<T: Field + Clone + PartialEq>(
    all: &[Point<T>],
    target: &Point<T>,
    keys_hash: &[u8; 32],
) -> Scalar<T> {
    // first key's bytes
    let first_bytes = all[0].to_bytes();

    // scan the list until we find any key whose bytes differ from first_bytes
    let second_bytes = all
        .iter()
        .find(|p| p.to_bytes() != first_bytes)
        .map(|p| p.to_bytes());

    //  if the bytes of `target` is that "second" key, shortcut a_i = 1, so just return 1
    if let Some(bytes2) = second_bytes {
        if target.to_bytes() == bytes2 {
            return Scalar::from_bytes_be(&[1]);
        }
    }

    // else hash to a scalar, shortcut a_i = H(L||P_i)
    //  where L = H(P1||P2||P3...)
    //  and P_i = target
    let mut h = Sha256::new();
    h.update(keys_hash);
    h.update(target.to_bytes());
    let digest: [u8; 32] = h.finalize().into();
    Scalar::from_bytes_be(&digest)
}

/// Aggregate public keys into Q = ∑ a_i · P_i
pub fn aggregate_keys<T: Field + Clone + PartialEq>(
    pubkeys: &[Point<T>],
) -> (Point<T>, Vec<Scalar<T>>) {
    assert!(!pubkeys.is_empty(), "Need at least one public key");
    // L = {P1, . . . , Pn} is the ordered list of all signers' public keys.
    let L = hash_keys(pubkeys);
    let mut Q = Point::identity();
    // `coeffs` is used to store the coefficients a_i, so we know its exact capacity
    // and can avoid reallocations.
    //  we each signer's weight a_i (the coefficients) for computing and verifying their partial signatures
    let mut coeffs = Vec::with_capacity(pubkeys.len());
    for P in pubkeys {
        let a = key_coef(pubkeys, P, &L);
        Q = Q + &(a.clone() * P);
        coeffs.push(a);
    }
    (Q, coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use num_traits::{FromPrimitive, Zero};
    use secp256k1::field::Field;
    use secp256k1::point::Point;
    use secp256k1::scalar::Scalar;

    /// A tiny field for y² = x³ + 7  over F₍₂₂₃₎
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Field223;
    impl Field for Field223 {
        fn prime() -> BigUint {
            BigUint::from_u64(223).unwrap()
        }
        fn a() -> BigUint {
            BigUint::zero()
        }
        fn b() -> BigUint {
            BigUint::from_u64(7).unwrap()
        }
    }

    fn p1() -> Point<Field223> {
        Point::new(
            BigUint::from_u64(192).unwrap(),
            BigUint::from_u64(105).unwrap(),
            false,
        )
    }
    fn p2() -> Point<Field223> {
        Point::new(
            BigUint::from_u64(17).unwrap(),
            BigUint::from_u64(56).unwrap(),
            false,
        )
    }

    fn p3() -> Point<Field223> {
        Point::new(
            BigUint::from_u64(1).unwrap(),
            BigUint::from_u64(193).unwrap(),
            false,
        )
    }

    fn one() -> Scalar<Field223> {
        Scalar::from_bytes_be(&[1])
    }

    #[test]
    fn hash_keys_is_deterministic() {
        let pts = vec![p1(), p2()];
        let L1 = hash_keys(&pts);
        let L2 = hash_keys(&pts);
        assert_eq!(
            L1, L2,
            "hash_keys should always return the same L for the same input"
        );
    }

    #[test]
    fn key_coef_single_key() {
        let pts = vec![p1()];
        let L = hash_keys(&pts);
        let ai = key_coef(&pts, &p1(), &L);
        // With only one key there's no "second" to optimize, so ai = H(L||P)
        let mut h = Sha256::new();
        h.update(&L);
        h.update(p1().to_bytes());
        let digest: [u8; 32] = h.finalize().into();
        let expected = Scalar::from_bytes_be(&digest);
        assert_eq!(ai, expected, "key_coef should match H(L||P)");
    }

    #[test]
    fn aggregate_single_point_uses_coefficient() {
        let pts = vec![p1()];
        let L = hash_keys(&pts);
        let (Q, coeffs) = aggregate_keys(&pts);

        // should have exactly one coefficient
        assert_eq!(coeffs.len(), 1);
        assert_eq!(coeffs[0], key_coef(&pts, &p1(), &L));

        // Q = a₁ · P₁
        let a1 = coeffs[0].clone();
        let expected = a1 * &p1();
        assert_eq!(Q, expected);
    }

    #[test]
    fn two_key_second_gets_one() {
        let pts = vec![p1(), p2()];
        let L = hash_keys(&pts);

        let a1 = key_coef(&pts, &p1(), &L);
        let a2 = key_coef(&pts, &p2(), &L);
        assert_ne!(a1, one(), "first key should not get the trivial 1");
        assert_eq!(a2, one(), "second distinct key must get coefficient = 1");

        // Q = a₁·P₁ + 1·P₂
        let (Q, coeffs) = aggregate_keys(&pts);
        let expected = coeffs[0].clone() * &p1() + &p2();
        assert_eq!(Q, expected);
    }

    #[test]
    fn order_matters_for_two_keys() {
        let pts12 = vec![p1(), p2()];
        let pts21 = vec![p2(), p1()];

        let L12 = hash_keys(&pts12);
        let L21 = hash_keys(&pts21);

        // In [p1,p2], p2 is second → a2=1
        assert_eq!(key_coef(&pts12, &p2(), &L12), one());

        // In [p2,p1], p1 is second → a1=1
        assert_eq!(key_coef(&pts21, &p1(), &L21), one());
    }

    #[test]
    fn duplicate_keys_sum() {
        let pts = vec![p1(), p1()];
        let L = hash_keys(&pts);
        let (Q, coeffs) = aggregate_keys(&pts);

        // both coeffs computed by hash (no "second distinct" found)
        assert_eq!(coeffs.len(), 2);
        assert_eq!(coeffs[0], coeffs[1]);

        // Q = a·P + a·P = (a+a)·P
        let a = coeffs[0].clone();
        let expected = (a.clone() + &a) * &p1();
        assert_eq!(Q, expected);
    }

    #[test]
    fn three_keys_mixed() {
        let pts = vec![p1(), p2(), p3()];
        let L = hash_keys(&pts);
        let (Q, coeffs) = aggregate_keys(&pts);

        // second distinct key is p2 → a2 = 1
        assert_eq!(coeffs[1], one());

        // the others should be hashed
        assert_ne!(coeffs[0], one());
        assert_ne!(coeffs[2], one());

        // Q = a1·P1 + 1·P2 + a3·P3
        let a1 = coeffs[0].clone();
        let a3 = coeffs[2].clone();
        let expected = (a1 * &p1()) + &p2() + &(a3 * &p3());
        assert_eq!(Q, expected);
    }

    #[test]
    fn key_coef_two_keys_second_is_one() {
        let pts = vec![p1(), p2()];
        let L = hash_keys(&pts);
        // the “second” key = p2, so its coefficient must be exactly 1
        let a2 = key_coef(&pts, &p2(), &L);
        assert_eq!(a2, one());
    }

    #[test]
    fn key_coef_two_keys_first_not_one() {
        let pts = vec![p1(), p2()];
        let L = hash_keys(&pts);
        // the “first” key = p1 must *not* get the trivial 1 (unless by chance H = 1)
        let a1 = key_coef(&pts, &p1(), &L);
        assert_ne!(a1, one());
    }

    #[test]
    fn aggregate_single_point_returns_itself() {
        let pts = vec![p1()];
        let (Q, coeffs) = aggregate_keys(&pts);
        // Old (incorrect) expectation:
        // assert_eq!(Q, p1(), "Aggregate of single key should be that key");
        assert_eq!(coeffs.len(), 1);

        // New, correct: Q == a₁ · P₁
        let L = hash_keys(&pts);
        let a1 = key_coef(&pts, &p1(), &L);
        let expected = a1.clone() * &p1();
        assert_eq!(Q, expected, "Aggregate of a single key should be a₁ · P₁");
        assert_eq!(coeffs[0], a1);
    }

    #[test]
    fn aggregate_two_points_weighted_sum() {
        let pts = vec![p1(), p2()];
        let (Q, coeffs) = aggregate_keys(&pts);
        // Q = a1*P1 + 1*P2
        let a1 = coeffs[0].clone();
        let expected = a1.clone() * &p1() + &p2();
        assert_eq!(Q, expected, "Aggregate must be a1·p1 + p2");
    }

    #[test]
    #[should_panic(expected = "Need at least one public key")]
    fn aggregate_empty_panics() {
        let empty: Vec<Point<Field223>> = vec![];
        let _ = aggregate_keys(&empty);
    }
}
