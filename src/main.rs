use clap::Parser;
use rand::prelude::*;
use rayon::prelude::*;
use std::{
    fs::File,
    io::{Read, Result},
};

#[derive(Parser, Debug)]
struct Args {
    checkpoint_file: String,
    temperature: Option<f32>,
    seed: Option<u64>,
}
fn main() {
    let args = Args::parse();
    println!("{:?}", args);

    let mut _rng = rand_chacha::ChaCha8Rng::seed_from_u64(
        args.seed.unwrap_or(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        ),
    );
    let temperature: f32 = args.temperature.unwrap_or(0.9f32);

    let mut checkpoint_file =
        std::fs::File::open(args.checkpoint_file).expect("Unable to open the checkpoint file");
    let config = read_config(&mut checkpoint_file);
    println!("config: {:?}", config);
    let weights =
        checkpoint_init_weights(&config, &mut checkpoint_file).expect("Unable to read weights");
    // println!("weights: {:?}", weights);
    // close the checkpoint file
    drop(checkpoint_file);

    let vocab = read_vocab(&config);
    // println!("vocab: {:?}; vocab length is: {}", vocab, vocab.len());

    // the current position we are in
    let start_time = std::time::Instant::now();

    let mut state = RunState::new(&config);
    let mut next: i32;
    let mut token: usize = 1; // 1 = BOS token in Llama-2 sentencepiece
    let mut pos: i32 = 0;
    while pos < config.seq_len {
        // forward the transformer to get logits for the next token
        transformer(token, pos as usize, &config, &mut state, &weights);

        if temperature == 0.0f32 {
            // greedy argmax sampling
            next = argmax(&state.logits, config.vocab_size as usize);
        } else {
            // apply the temperature to the logits
            for q in 0..config.vocab_size {
                state.logits[q as usize] /= temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            softmax(&mut state.logits, config.vocab_size as usize);
            // we now want to sample from this distribution to get the next token
            next = sample(&state.logits, config.vocab_size as usize);
        }

        // printf("%s", vocab[next]);
        print!("{}", vocab[next as usize]);
        // advance forward
        token = next as usize;
        pos += 1;
        // println!("pos: {}", pos);
    }

    // report our achieved tok/s
    let end_time = std::time::Instant::now();
    println!(
        "achieved tok/s: {}",
        config.seq_len as f32 / end_time.duration_since(start_time).as_secs_f32()
    );
}

fn read_vocab(config: &Config) -> Vec<String> {
    let mut vocab_file = File::open("tokenizer.bin").expect("Unable to open the tokenizer file");
    let mut vocab = Vec::new();
    for _ in 0..config.vocab_size {
        let mut len_buffer = [0_u8; std::mem::size_of::<i32>()];
        vocab_file
            .read_exact(&mut len_buffer)
            .expect("Unable to read word length");
        let len: i32 = unsafe { std::mem::transmute(len_buffer) };
        let mut word_buffer = vec![0_u8; len as usize];
        vocab_file
            .read_exact(&mut word_buffer)
            .expect("Unable to read word");
        vocab.push(String::from_utf8(word_buffer).expect("Unable to parse word"));
    }
    vocab
}

// read in the config header using unsafe
fn read_config(file: &mut File) -> Config {
    let mut buffer = [0_u8; std::mem::size_of::<Config>()];
    file.read_exact(&mut buffer)
        .expect("Read failed for Config");

    let config: Config = unsafe { std::mem::transmute(buffer) };
    config
}

fn read_weights_from_file(num_weights: i32, file: &mut File) -> Result<Vec<f32>> {
    // Calculate the number of bytes to read based on the number of floats
    let num_bytes = num_weights as usize * std::mem::size_of::<f32>();

    // Read the specified number of bytes from the file
    let mut buffer = vec![0; num_bytes];
    file.read_exact(&mut buffer)?;

    // Convert the bytes to f32 values
    let mut float_values = Vec::new();
    for i in (0..buffer.len()).step_by(4) {
        let bytes = [buffer[i], buffer[i + 1], buffer[i + 2], buffer[i + 3]];

        // Assuming the bytes are in little-endian format
        let float_value = f32::from_le_bytes(bytes);
        float_values.push(float_value);
    }

    Ok(float_values)
}

#[derive(Debug)]
#[repr(C)]
struct Config {
    dim: i32,        // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32,   // number of layers
    n_heads: i32,    // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32,    // max sequence length
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls
    wq: Vec<f32>, // (layer, dim, dim)
    wk: Vec<f32>, // (layer, dim, dim)
    wv: Vec<f32>, // (layer, dim, dim)
    wo: Vec<f32>, // (layer, dim, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
}

struct RunState {
    // current wave of activations
    x: Vec<f32>,      // activation at current time stamp (dim,)
    xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,      // query (dim,)
    k: Vec<f32>,      // key (dim,)
    v: Vec<f32>,      // value (dim,)
    att: Vec<f32>,    // buffer for scores/attention values (seq_len,)
    logits: Vec<f32>, // output logits
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}
impl RunState {
    fn new(config: &Config) -> Self {
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        let seq_len = config.seq_len as usize;
        let vocab_size = config.vocab_size as usize;
        let n_layers = config.n_layers as usize;
        let cache_size = n_layers * seq_len * dim;

        Self {
            x: vec![0.0f32; dim],
            xb: vec![0.0f32; dim],
            xb2: vec![0.0f32; dim],
            hb: vec![0.0f32; hidden_dim],
            hb2: vec![0.0f32; hidden_dim],
            q: vec![0.0f32; dim],
            k: vec![0.0f32; dim],
            v: vec![0.0f32; dim],
            att: vec![0.0f32; seq_len],
            logits: vec![0.0f32; vocab_size],
            key_cache: vec![0.0f32; cache_size],
            value_cache: vec![0.0f32; cache_size],
        }
    }
}

fn checkpoint_init_weights(p: &Config, f: &mut std::fs::File) -> Result<TransformerWeights> {
    let head_size: i32 = p.dim / p.n_heads;
    let w = TransformerWeights {
        token_embedding_table: read_weights_from_file(p.vocab_size * p.dim, f)?,
        rms_att_weight: read_weights_from_file(p.n_layers * p.dim, f)?,
        wq: read_weights_from_file(p.n_layers * p.dim * p.dim, f)?,
        wk: read_weights_from_file(p.n_layers * p.dim * p.dim, f)?,
        wv: read_weights_from_file(p.n_layers * p.dim * p.dim, f)?,
        wo: read_weights_from_file(p.n_layers * p.dim * p.dim, f)?,
        rms_ffn_weight: read_weights_from_file(p.n_layers * p.dim, f)?,
        w1: read_weights_from_file(p.n_layers * p.dim * p.hidden_dim, f)?,
        w2: read_weights_from_file(p.n_layers * p.hidden_dim * p.dim, f)?,
        w3: read_weights_from_file(p.n_layers * p.dim * p.hidden_dim, f)?,
        rms_final_weight: read_weights_from_file(p.dim, f)?,
        freq_cis_real: read_weights_from_file(p.seq_len * head_size / 2, f)?,
        freq_cis_imag: read_weights_from_file(p.seq_len * head_size / 2, f)?,
    };

    println!(
        "token_embedding_table: {:?} \
        rms_ffn_weight: {:?} \
        rms_att_weight: {:?} \n\
        wq: {:?} \
        wk: {:?} \
        wv: {:?} \
        wo: {:?} \n\
        w1: {:?} \
        w2: {:?} \
        w3: {:?} \n\
        rms_final_weight: {:?} \
        freq_cis_real: {:?} \
        freq_cis_imag: {:?} \n\n",
        w.token_embedding_table.len(),
        w.rms_ffn_weight.len(),
        w.rms_att_weight.len(),
        w.wq.len(),
        w.wk.len(),
        w.wv.len(),
        w.wo.len(),
        w.w1.len(),
        w.w2.len(),
        w.w3.len(),
        w.rms_final_weight.len(),
        w.freq_cis_real.len(),
        w.freq_cis_imag.len(),
    );

    Ok(w)
}

fn accum(a: &mut [f32], b: &[f32], size: usize) {
    for i in 0..size {
        a[i] += b[i];
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
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

fn softmax(x: &mut [f32], size: usize) {
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
fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
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
fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    assert!(n % 16 == 0); // Make sure n is divisible by 4 for this example
    let simd_width = 8; // Change to 8 for AVX (256-bit SIMD)

    (0..d).for_each(|i| {
        let mut sum = f32x8::splat(0.0); // For AVX, use f32x8::splat(0.0)

        for j in (0..n).step_by(simd_width) {
            let x_chunk = f32x8::from_slice_unaligned(&x[j..]);
            let w_chunk = f32x8::from_slice_unaligned(&w[i * n + j..]);
            sum += x_chunk * w_chunk;
        }

        xout[i] = sum.sum(); // Sum the SIMD vector elements to get the final value
    });
}

fn transformer(token: usize, pos: usize, p: &Config, s: &mut RunState, w: &TransformerWeights) {
    // a few convenience variables
    let x = &mut s.x;
    let dim = p.dim as usize;
    let hidden_dim = p.hidden_dim as usize;
    let head_size = (p.dim / p.n_heads) as usize;
    // copy the token embedding into x
    let content_row = &w.token_embedding_table[token * dim..(token + 1) * dim];
    // println!(" content_row length: {:?}\n", content_row.len());
    x.copy_from_slice(content_row);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = &w.freq_cis_real[pos * head_size / 2..];
    let freq_cis_imag_row = &w.freq_cis_imag[pos * head_size / 2..];

    // forward all the layers
    for l in 0..p.n_layers as usize {
        // attention rmsnorm
        rmsnorm(&mut s.xb, x, &w.rms_att_weight[l * dim..], dim);

        // qkv matmuls for this position
        matmul(&mut s.q, &s.xb, &w.wq[l * dim * dim..], dim, dim);
        matmul(&mut s.k, &s.xb, &w.wk[l * dim * dim..], dim, dim);
        matmul(&mut s.v, &s.xb, &w.wv[l * dim * dim..], dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..p.n_heads as usize {
            // get the q and k vectors for this head
            let q = &mut s.q[h * head_size..];
            let k = &mut s.k[h * head_size..];
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for i in (0..head_size).step_by(2) {
                let q0 = q[i];
                let q1 = q[i + 1];
                let k0 = k[i];
                let k1 = k[i + 1];

                let fcr = freq_cis_real_row[i / 2];
                let fci = freq_cis_imag_row[i / 2];

                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;

                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        let loff = l * p.seq_len as usize * dim; // kv cache layer offset for convenience
        let key_cache_row = &mut s.key_cache[loff + pos * dim..loff + (pos + 1) * dim];
        let value_cache_row = &mut s.value_cache[loff + pos * dim..loff + (pos + 1) * dim];
        key_cache_row.copy_from_slice(&s.k);
        value_cache_row.copy_from_slice(&s.v);

        // multihead attention. iterate over all heads
        for h in 0..p.n_heads as usize {
            // get the query vector for this head
            let q = &s.q[h * head_size..];
            // iterate over all timesteps, including the current one
            for t in 0..pos + 1 {
                // get the key vector for this head and at this timestep
                let k = &s.key_cache[loff + t * dim + h * head_size..];
                // calculate the attention score as the dot product of q and k
                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += q[i] * k[i];
                }
                score /= (head_size as f32).sqrt();
                // save the score to the attention buffer
                s.att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&mut s.att, pos + 1);

            // weighted sum of the values, store back into xb
            for i in 0..head_size {
                let mut val = 0.0f32;
                for t in 0..pos + 1 {
                    val += s.att[t] * s.value_cache[loff + t * dim + h * head_size + i];
                    // note bad locality
                }
                s.xb[h * head_size + i] = val;
            }
        }

        // final matmul to get the output of the attention
        matmul(&mut s.xb2, &s.xb, &w.wo[l * dim * dim..], dim, dim);

        // residual connection back into x
        accum(x, &s.xb2, dim);

        // ffn rmsnorm
        rmsnorm(&mut s.xb, x, &w.rms_ffn_weight[l * dim..], dim);
        //
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(
            &mut s.hb,
            &s.xb,
            &w.w1[l * dim * hidden_dim..],
            dim,
            hidden_dim,
        );
        matmul(
            &mut s.hb2,
            &s.xb,
            &w.w3[l * dim * hidden_dim..],
            dim,
            hidden_dim,
        );

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..hidden_dim {
            s.hb[i] = s.hb[i] * (1.0f32 / (1.0f32 + (-s.hb[i]).exp()));
        }

        // elementwise multiply with w3(x)
        for i in 0..hidden_dim {
            s.hb[i] *= s.hb2[i];
        }
        // final matmul to get the output of the ffn
        matmul(
            &mut s.xb,
            &s.hb,
            &w.w2[l * dim * hidden_dim..],
            hidden_dim,
            dim,
        );
        // residual connection
        accum(x, &s.xb, dim);
    }

    // final rmsnorm
    let immutable_x = &x.clone();
    rmsnorm(x, immutable_x, &w.rms_final_weight, p.dim as usize);
    //
    // classifier into logits
    matmul(
        &mut s.logits,
        &s.x,
        &w.token_embedding_table,
        p.dim as usize,
        p.vocab_size as usize,
    );
}

fn sample(probabilities: &[f32], n: usize) -> i32 {
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

fn argmax(v: &[f32], n: usize) -> i32 {
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
