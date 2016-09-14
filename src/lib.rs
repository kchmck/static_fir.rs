extern crate num;

use std::ops::{Add, Mul};

use num::traits::Zero;

pub trait FIRCoefs: Default {
    type Sample: Copy + Clone + Zero + Add<Output = Self::Sample> +
        Mul<f32, Output = Self::Sample>;

    fn size() -> usize;
    fn coefs() -> &'static [f32];
    fn history(&self) -> &[Self::Sample];
    fn history_mut(&mut self) -> &mut [Self::Sample];

    fn calc(&self, idx: usize) -> Self::Sample {
        let (hleft, hright) = self.history().split_at(idx);
        let (cleft, cright) = Self::coefs().split_at(Self::size() - idx);

        cleft.iter().zip(hright)
            .fold(Self::Sample::zero(), |s, (&c, &x)| s + x * c) +
        cright.iter().zip(hleft)
            .fold(Self::Sample::zero(), |s, (&c, &x)| s + x * c)
    }

    fn verify_symmetry() {
        for i in 0..Self::size() / 2 {
            assert_eq!(Self::coefs()[i], Self::coefs()[Self::size() - i - 1]);
        }
    }
}

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
            fn history(&self) -> &[Self::Sample] { &self.0[..] }
            fn history_mut(&mut self) -> &mut [Self::Sample] { &mut self.0[..] }
        }

        impl Default for $name {
            fn default() -> Self {
                $name([::num::traits::Zero::zero(); $size])
            }
        }
    };
}

/// A FIR filter for convolving with a series of samples.
pub struct FIRFilter<C: FIRCoefs> {
    inner: C,
    /// The index of the most-recently added sample.
    idx: usize,
}

impl<C: FIRCoefs> FIRFilter<C> {
    /// Construct an order-N filter with the given N+1 coefficients.
    pub fn new() -> FIRFilter<C> {
        FIRFilter {
            inner: C::default(),
            idx: 0,
        }
    }

    /// Add a sample to the current history and calculate the convolution.
    pub fn feed(&mut self, sample: C::Sample) -> C::Sample {
        // Store the given sample in the current history slot.
        self.inner.history_mut()[self.idx] = sample;

        // Move to the next slot and wrap around.
        self.idx += 1;
        self.idx %= C::size();

        self.inner.calc(self.idx)
    }

    pub fn hist(&self) -> (&[C::Sample], &[C::Sample]) {
        self.inner.history().split_at(self.idx)
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

    impl_fir!(SymmetricFIR, f32, 5, [
        0.2,
        0.4,
        1.0,
        0.4,
        0.2,
    ]);

    impl_fir!(NonSymmetricFIR, f32, 5, [
        0.2,
        0.4,
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
    }

    #[test]
    fn test_verify_symmetry() {
        SymmetricFIR::verify_symmetry();
    }

    #[test]
    #[should_panic]
    fn test_very_nonsymmetry() {
        NonSymmetricFIR::verify_symmetry();
    }
}
