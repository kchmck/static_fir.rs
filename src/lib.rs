//! Finite-impulse response (FIR) convolution with static tap coefficients.

#![feature(conservative_impl_trait)]

extern crate num;

use std::ops::{Add, Mul, Deref, DerefMut};

use num::traits::Zero;

/// Provides a sequence of coefficients and storage for sample history.
pub trait FIRCoefs: Default + Deref<Target = [<Self as FIRCoefs>::Sample]> + DerefMut {
    /// Type of sample stored in the history.
    type Sample: Copy + Clone + Zero + Add<Output = Self::Sample> +
        Mul<f32, Output = Self::Sample>;

    /// Number of coefficients/stored samples.
    fn size() -> usize;
    /// Sequence of coefficients.
    fn coefs() -> &'static [f32];

    /// Verify the requirement that the filter coefficients are symmetric around the
    /// center (either even or odd length.)
    fn verify_symmetry() {
        for i in 0..Self::size() / 2 {
            assert_eq!(Self::coefs()[i], Self::coefs()[Self::size() - i - 1]);
        }
    }
}

/// Implement `FIRCoefs` for a history buffer with the given name, input sample type,
/// storage size, and sequence of coefficients.
#[macro_export]
macro_rules! impl_fir {
    ($name:ident, $sample:ty, $size:expr, $coefs:expr) => {
        pub struct $name([$sample; $size]);

        impl $crate::FIRCoefs for $name {
            type Sample = $sample;
            fn size() -> usize { $size }
            fn coefs() -> &'static [f32] {
                static COEFS: [f32; $size] = $coefs;
                &COEFS[..]
            }
        }

        impl Default for $name {
            fn default() -> Self {
                $name([::num::traits::Zero::zero(); $size])
            }
        }

        impl std::ops::Deref for $name {
            type Target = [$sample];
            fn deref(&self) -> &Self::Target { &self.0[..] }
        }

        impl std::ops::DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0[..] }
        }
    };
}

/// A FIR filter for convolving with a series of samples.
pub struct FIRFilter<C: FIRCoefs> {
    /// Coefficients and history storage.
    inner: C,
    /// The index of the most-recently added sample.
    idx: usize,
}

impl<C: FIRCoefs> FIRFilter<C> {
    /// Create a new `FIRFilter` with empty history.
    pub fn new() -> FIRFilter<C> {
        FIRFilter {
            inner: C::default(),
            idx: 0,
        }
    }

    /// Add a sample to the current history and calculate the convolution.
    pub fn feed(&mut self, sample: C::Sample) -> C::Sample {
        // Store the given sample in the current history slot.
        self.inner[self.idx] = sample;

        // Move to the next slot and wrap around.
        self.idx += 1;
        self.idx %= C::size();

        self.calc()
    }

    /// Calculate the convolution of saved samples with coefficients, where the given
    /// index gives the position of the most recent sample in the history ring buffer.
    fn calc(&self) -> C::Sample {
        let (hleft, hright) = self.inner.split_at(self.idx);
        let (cleft, cright) = C::coefs().split_at(C::size() - self.idx);

        cleft.iter().zip(hright)
            .fold(C::Sample::zero(), |s, (&c, &x)| s + x * c) +
        cright.iter().zip(hleft)
            .fold(C::Sample::zero(), |s, (&c, &x)| s + x * c)
    }

    /// Iterate over the history of stored samples, with the oldest sample as the first
    /// item yielded and the newest as the last.
    #[inline]
    pub fn history<'a>(&'a self) -> impl Iterator<Item = &'a C::Sample> {
        let (left, right) = self.inner.split_at(self.idx);
        right.iter().chain(left.iter())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl_fir!(TestFIR, f32, 4, [
        1.0,
        0.0,
        2.0,
        0.0,
    ]);

    impl_fir!(SymmetricOddFIR, f32, 5, [
        0.2,
        0.4,
        1.0,
        0.4,
        0.2,
    ]);

    impl_fir!(SymmetricEvenFIR, f32, 6, [
        0.2,
        0.4,
        1.0,
        1.0,
        0.4,
        0.2,
    ]);

    impl_fir!(NonSymmetricOddFIR, f32, 5, [
        0.2,
        0.4,
        1.0,
        0.5,
        0.2,
    ]);

    impl_fir!(NonSymmetricEvenFIR, f32, 6, [
        0.2,
        0.4,
        1.0,
        1.0,
        0.5,
        0.2,
    ]);

    #[test]
    fn test_fir() {
        let mut f = FIRFilter::<TestFIR>::new();

        assert!(f.feed(100.0) == 0.0);
        assert!(f.feed(200.0) == 200.0);
        assert!(f.feed(300.0) == 400.0);
        assert!(f.feed(400.0) == 700.0);
        assert!(f.feed(0.0) == 1000.0);
        assert!(f.feed(0.0) == 300.0);
        assert!(f.feed(0.0) == 400.0);
        assert!(f.feed(0.0) == 0.0);
        assert!(f.feed(0.0) == 0.0);
        assert!(f.feed(100.0) == 0.0);
        assert!(f.feed(200.0) == 200.0);
        assert!(f.feed(300.0) == 400.0);
        assert!(f.feed(400.0) == 700.0);

        let mut iter = f.history();

        assert_eq!(iter.next().unwrap(), &100.0);
        assert_eq!(iter.next().unwrap(), &200.0);
        assert_eq!(iter.next().unwrap(), &300.0);
        assert_eq!(iter.next().unwrap(), &400.0);
    }

    #[test]
    fn test_verify_symmetry() {
        SymmetricOddFIR::verify_symmetry();
        SymmetricEvenFIR::verify_symmetry();
    }

    #[test]
    #[should_panic]
    fn test_verify_nonsymmetry_odd() {
        NonSymmetricOddFIR::verify_symmetry();
    }

    #[test]
    #[should_panic]
    fn test_verify_nonsymmetry_even() {
        NonSymmetricEvenFIR::verify_symmetry();
    }
}
