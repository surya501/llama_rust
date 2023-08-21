use rayon::prelude::*;

pub fn sample(probabilities: &[f32], n: usize) -> i32 {
    // sample index from probabilities, they must sum to 1
    // TODO: use rand seed from args and accept as parameter
    let r = rand::random::<u32>() as f32 / (u32::MAX as f32);
    let mut cdf = 0.0f32;
    for i in 0..n {
        cdf += probabilities[i];
        if r < cdf {
            return i as i32;
        }
    }
    (n - 1) as i32 // in case of rounding errors
}

pub fn argmax(v: &[f32], n: usize) -> i32 {
    // return argmax of v in elements 0..n
    let mut max_i = 0;
    let mut max_p = v[0];
    for i in 1..n {
        if v[i] > max_p {
            max_i = i;
            max_p = v[i];
        }
    }
    max_i as i32
}

pub fn accum(a: &mut [f32], b: &[f32], size: usize) {
    for i in 0..size {
        a[i] += b[i];
    }
}

pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
    // calculate sum of squares
    let mut ss = 0.0f32;
    (0..size).for_each(|j| {
        ss += x[j] * x[j];
    });
    ss /= size as f32;
    ss += 1e-5f32;
    ss = 1.0f32 / ss.sqrt();
    // normalize and scale
    for j in 0..size {
        o[j] = weight[j] * (ss * x[j]);
    }
}

pub fn softmax(x: &mut [f32], size: usize) {
    // find max value (for numerical stability)
    let mut max_val = x[0];
    (1..size).for_each(|i| {
        if x[i] > max_val {
            max_val = x[i];
        }
    });
    // exp and sum
    let mut sum = 0.0f32;
    (0..size).for_each(|i| {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    });
    // normalize
    (0..size).for_each(|i| {
        x[i] /= sum;
    });
}

#[cfg(not(feature = "simd"))]
pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    let result = (0..d)
        .into_par_iter()
        .map(|i| {
            let mut val = 0.0f32;
            for j in (0..n).step_by(4) {
                // Unroll the loop by 4 for better performance (and SIMD utilization?)
                val += w[i * n + j] * x[j];
                val += w[i * n + j + 1] * x[j + 1];
                val += w[i * n + j + 2] * x[j + 2];
                val += w[i * n + j + 3] * x[j + 3];
            }
            val
        })
        .collect::<Vec<f32>>();
    xout.copy_from_slice(&result);
}

// Reimplement the matmul program using iterators
// #[cfg(not(feature = "simd"))]
// fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
//     let result = (0..d)
//         .into_par_iter()
//         .map(|i| (0..n).map(|j| w[i * n + j] * x[j]).sum())
//         .collect::<Vec<f32>>();
//     xout.copy_from_slice(&result);
// }

#[cfg(feature = "simd")]
use packed_simd::f32x8; // For AVX (256-bit SIMD) use packed_simd::f32x8
#[cfg(feature = "simd")]
pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    assert!(n % 16 == 0); // Make sure n is divisible by 4 for this example
    let simd_width = 8; // Change to 8 for AVX (256-bit SIMD)

    let result = (0..d)
        .into_par_iter()
        .map(|i| {
            let mut sum = f32x8::splat(0.0); // For AVX, use f32x8::splat(0.0)

            for j in (0..n).step_by(simd_width) {
                let x_chunk = f32x8::from_slice_unaligned(&x[j..]);
                let w_chunk = f32x8::from_slice_unaligned(&w[i * n + j..]);
                sum += x_chunk * w_chunk;
            }

            sum.sum() // Sum the SIMD vector elements to get the final value
        })
        .collect::<Vec<f32>>();
    xout.copy_from_slice(&result);
}
