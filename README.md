
## llama_rust
This is a pure rust implementation of [llama2.c](https://github.com/karpathy/llama2.c) repository.

Why? Simply because you learn a bit by retyping code; I didn't see a point in typing in c or python/numpy. Performance optimization in rust has been on my todo list for sometime and this scratches that itch!!

This repo contains only the inference code and has been tested with fp32, dim 288 6-layer 6-head model (~15M params) pretrained model available in the original repo.

I've taken effort to keep the repo dependency free for core items, but have added ancillary crates like rand and clap to focus on the important stuff.

## How to run
To run the baby Llama 2 model in from the C repository, you need that model checkpoint. You also need the tokenizer.bin from that repository. It's better to checkout that repository and copy items from there.

### Below are instructions from the [original repo](https://github.com/karpathy/llama2.c) on how to obtain the trained model

Download this 15M parameter model I trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~58MB download) and place it into the default checkpoint directory `out`:

```bash
wget https://karpathy.ai/llama2c/model.bin -P out
```

(if that doesn't work try [google drive](https://drive.google.com/file/d/1aTimLdx3JktDXxcHySNrZJOOk8Vb1qBR/view?usp=share_link)).

Once we have the model.bin file, we can inference in Rust. Compile and run code as follows

```bash
cargo run --release --  out/model.bin
```

Simd code requires nightly compiler, so use the following command

```bash
cargo +nightly run --features simd  --release --  out/model.bin
```


## performance

*Add stuff as we optimize this*


| Version | Tokens/sec |
| ----------- | ---------- |
| Original code | 62 |
| Matmul code using iterators | 69 |
| Simd code (no AVX; 4 in parallel) | 120 |
| Simd code (AVX / 8 in parallel) | 180 |
| Rayon + matmul iterators | 217|
| Loop unrolling | 65|
| loop unrolling + Rayon| 214|
| SIMD + Rayon | 225 |

Install flamegraph following instructions in the [repo](https://github.com/flamegraph-rs/flamegraph). Profile using flamegraphs using the command:
```bash
cargo flamegraph  --  out/model.bin
```
For simd enabled code, profile using the following command:
```bash
cargo +nightly flamegraph  --features simd --  out/model.bin
```
## todos in no particular order

- [ ] Criterion for benchmarking
- [x] Profile / Flamegraph / etc
- [x] Rayon for Multithread
- [x] Simd via Rust
- [ ] Remove unsafe parse for config
- [ ] run in browser
- [ ] wgpu

## License

MIT
