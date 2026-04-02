# Weeks 5–6: GGUF Format Deep Dive

> **Goal:** Understand the GGUF binary format end-to-end — from the first magic byte to the last tensor data block — and build a minimal parser to prove it.

---

## Table of Contents

1. [GGUF Binary Format Specification](#1-gguf-binary-format-specification)
2. [KV Pair & Tensor Info Encoding](#2-kv-pair--tensor-info-encoding)
3. [Metadata Key Namespace Reference](#3-metadata-key-namespace-reference)
4. [Quantization Schemes](#4-quantization-schemes)
5. [Block Structures & Dequantization](#5-block-structures--dequantization)
6. [GGUF Parser Building Guide](#6-gguf-parser-building-guide)
7. [SHA-256 Hashing Approach](#7-sha-256-hashing-approach)
8. [Critical File References](#8-critical-file-references)
9. [Deliverables Checklist](#9-deliverables-checklist)

---

## 1. GGUF Binary Format Specification

### Constants

| Constant | Value | Notes |
|----------|-------|-------|
| `GGUF_MAGIC` | `0x46554747` | ASCII `"GGUF"` stored **little-endian** (bytes: `47 47 55 46`) |
| `GGUF_VERSION` | `3` | Current spec version |
| `GGUF_DEFAULT_ALIGNMENT` | `32` | Tensor data alignment in bytes (overridable via `general.alignment`) |

> **Source:** `gguf-py/gguf/constants.py:10-12`

### File Layout

A GGUF file is a single contiguous binary blob with four regions:

```
┌─────────────────────────────────────────┐  offset 0
│  Header                                 │
│    Magic          (uint32, 4 bytes)     │
│    Version        (uint32, 4 bytes)     │
│    Tensor Count   (uint64, 8 bytes)     │
│    KV Count       (uint64, 8 bytes)     │
├─────────────────────────────────────────┤  offset 24
│  Key-Value Pairs  (kv_count entries)    │
│    Each: key_string + type + value      │
├─────────────────────────────────────────┤  variable
│  Tensor Info      (tensor_count entries)│
│    Each: name + n_dims + dims[] +       │
│          type + offset                  │
├─────────────────────── alignment pad ───┤
│  Tensor Data                            │
│    Raw quantized/float tensor bytes     │
│    (each tensor starts at its offset    │
│     relative to data section start)     │
└─────────────────────────────────────────┘
```

### Header Encoding (24 bytes, fixed)

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `magic` | `uint32` | 4 | Must be `0x46554747` (little-endian) |
| `version` | `uint32` | 4 | File format version (currently `3`) |
| `tensor_count` | `uint64` | 8 | Number of tensors in the file |
| `kv_count` | `uint64` | 8 | Number of key-value metadata pairs |

### Endianness Detection

The reader checks if `version & 0xFFFF == 0`. If so, the file uses swapped byte order, and all subsequent multi-byte reads must be byte-swapped.

> **Source:** `gguf-py/gguf/gguf_reader.py:140-148`

### Alignment & Padding

After all tensor info entries, the file is padded to the next multiple of `alignment` (default 32 bytes). The `general.alignment` KV pair can override this — it must be a non-zero power of two.

```
padding = offset % alignment
if padding != 0:
    offset += alignment - padding
data_section_start = offset
```

Each tensor's data is located at `data_section_start + tensor_info.offset`.

---

## 2. KV Pair & Tensor Info Encoding

### Value Types

| Type ID | Name | Size | Encoding |
|---------|------|------|----------|
| 0 | `UINT8` | 1 | Raw byte |
| 1 | `INT8` | 1 | Signed byte |
| 2 | `UINT16` | 2 | Little-endian |
| 3 | `INT16` | 2 | Little-endian |
| 4 | `UINT32` | 4 | Little-endian |
| 5 | `INT32` | 4 | Little-endian |
| 6 | `FLOAT32` | 4 | IEEE 754 |
| 7 | `BOOL` | 1 | 0 = false, nonzero = true |
| 8 | `STRING` | variable | `uint64 length` + `length` UTF-8 bytes (no null terminator) |
| 9 | `ARRAY` | variable | `uint32 elem_type` + `uint64 count` + `count` elements |
| 10 | `UINT64` | 8 | Little-endian |
| 11 | `INT64` | 8 | Little-endian |
| 12 | `FLOAT64` | 8 | IEEE 754 double |

> **Source:** `gguf-py/gguf/constants.py:3836-3849` (`GGUFValueType` enum)

### KV Pair Wire Format

```
┌──────────────────────────────────────┐
│ key_length   : uint64                │
│ key_data     : uint8[key_length]     │  UTF-8 string, no null terminator
│ value_type   : uint32                │  GGUFValueType enum
│ value_data   : <type-dependent>      │
└──────────────────────────────────────┘
```

For `STRING` values: `uint64 str_length` followed by `str_length` bytes.
For `ARRAY` values: `uint32 elem_type` + `uint64 count` + inlined elements.

### Tensor Info Wire Format

Each tensor info entry encodes:

```
┌──────────────────────────────────────┐
│ name_length  : uint64                │
│ name_data    : uint8[name_length]    │  UTF-8 tensor name
│ n_dimensions : uint32                │  Number of dimensions (1-4)
│ dimensions   : uint64[n_dimensions]  │  Size of each dimension
│ type         : uint32                │  GGMLQuantizationType enum value
│ offset       : uint64                │  Byte offset from data section start
└──────────────────────────────────────┘
```

---

## 3. Metadata Key Namespace Reference

GGUF metadata keys follow a hierarchical dot-notation namespace. Below are the primary namespaces and their most important keys.

### `general.*` — File-Level Metadata

| Key | Type | Description |
|-----|------|-------------|
| `general.architecture` | STRING | Model architecture name (e.g., `"llama"`, `"gpt2"`) |
| `general.name` | STRING | Human-readable model name |
| `general.file_type` | UINT32 | Quantization type used for most tensors |
| `general.quantization_version` | UINT32 | Quantization format version |
| `general.alignment` | UINT32 | Data alignment override (must be power of 2) |
| `general.type` | STRING | Model type (e.g., `"model"`, `"adapter"`) |
| `general.author` | STRING | Model author |
| `general.version` | STRING | Model version string |
| `general.description` | STRING | Free-form description |
| `general.license` | STRING | License identifier |
| `general.source.url` | STRING | Original model URL |
| `general.base_model.count` | UINT32 | Number of base models (for merges) |

### `{arch}.*` — Architecture Parameters

`{arch}` is replaced by the value of `general.architecture` (e.g., `llama.context_length`).

| Key | Type | Description |
|-----|------|-------------|
| `{arch}.vocab_size` | UINT32 | Vocabulary size |
| `{arch}.context_length` | UINT32 | Maximum context length |
| `{arch}.embedding_length` | UINT32 | Embedding dimensionality |
| `{arch}.block_count` | UINT32 | Number of transformer blocks |
| `{arch}.feed_forward_length` | UINT32 | FFN hidden dimension |
| `{arch}.expert_count` | UINT32 | MoE: total experts |
| `{arch}.expert_used_count` | UINT32 | MoE: experts per token |

### `{arch}.attention.*` — Attention Configuration

| Key | Type | Description |
|-----|------|-------------|
| `{arch}.attention.head_count` | UINT32 | Number of attention heads |
| `{arch}.attention.head_count_kv` | UINT32 | Number of KV heads (for GQA/MQA) |
| `{arch}.attention.key_length` | UINT32 | Key dimension per head |
| `{arch}.attention.value_length` | UINT32 | Value dimension per head |
| `{arch}.attention.layer_norm_rms_epsilon` | FLOAT32 | RMSNorm epsilon |

### `{arch}.rope.*` — RoPE Configuration

| Key | Type | Description |
|-----|------|-------------|
| `{arch}.rope.dimension_count` | UINT32 | RoPE embedding dimensions |
| `{arch}.rope.freq_base` | FLOAT32 | Base frequency (default 10000) |
| `{arch}.rope.scaling.type` | STRING | Scaling type: `"none"`, `"linear"`, `"yarn"` |
| `{arch}.rope.scaling.factor` | FLOAT32 | Context extension scaling factor |

### `tokenizer.ggml.*` — Tokenizer Data

| Key | Type | Description |
|-----|------|-------------|
| `tokenizer.ggml.model` | STRING | Tokenizer type: `"llama"`, `"gpt2"`, `"bert"` |
| `tokenizer.ggml.pre` | STRING | Pre-tokenizer type |
| `tokenizer.ggml.tokens` | ARRAY[STRING] | Token vocabulary |
| `tokenizer.ggml.scores` | ARRAY[FLOAT32] | Token scores/priorities |
| `tokenizer.ggml.merges` | ARRAY[STRING] | BPE merge rules |
| `tokenizer.ggml.token_type` | ARRAY[INT32] | Token type IDs |
| `tokenizer.ggml.bos_token_id` | UINT32 | Beginning-of-sequence token |
| `tokenizer.ggml.eos_token_id` | UINT32 | End-of-sequence token |
| `tokenizer.ggml.eot_token_id` | UINT32 | End-of-turn token |

### Other Namespaces

| Namespace | Purpose |
|-----------|---------|
| `{arch}.ssm.*` | Mamba/state-space model parameters |
| `{arch}.wkv.*` | RWKV WKV parameters |
| `adapter.*` | LoRA adapter metadata |
| `clip.vision.*` | Vision encoder parameters |
| `clip.audio.*` | Audio encoder parameters |

> **Source:** `gguf-py/gguf/constants.py:20-340` (class `Keys`)

---

## 4. Quantization Schemes

### GGMLQuantizationType Enum

| Enum Value | Name | Notes |
|------------|------|-------|
| 0 | `F32` | Full precision |
| 1 | `F16` | Half precision |
| 2 | `Q4_0` | Legacy 4-bit |
| 3 | `Q4_1` | Legacy 4-bit with min |
| 6 | `Q5_0` | Legacy 5-bit |
| 7 | `Q5_1` | Legacy 5-bit with min |
| 8 | `Q8_0` | Legacy 8-bit |
| 9 | `Q8_1` | Legacy 8-bit (intermediate) |
| 10 | `Q2_K` | K-quant 2-bit |
| 11 | `Q3_K` | K-quant 3-bit |
| 12 | `Q4_K` | K-quant 4-bit |
| 13 | `Q5_K` | K-quant 5-bit |
| 14 | `Q6_K` | K-quant 6-bit |
| 15 | `Q8_K` | K-quant 8-bit (intermediate) |
| 16–23 | `IQ*` | Importance-matrix quants |
| 29 | `IQ1_M` | 1-bit importance quant |
| 30 | `BF16` | Brain float 16 |
| 34 | `TQ1_0` | Ternary 1-bit |
| 35 | `TQ2_0` | Ternary 2-bit |
| 39 | `MXFP4` | MX float 4-bit |

> **Source:** `gguf-py/gguf/constants.py:3743-3775`, `ggml/include/ggml.h:389-430`

### Quantization Comparison Table

#### Legacy Quants (block size = 32)

| Type | Block Size | Bytes/Block | Bits/Weight | Components |
|------|-----------|-------------|-------------|------------|
| `F32` | 1 | 4 | 32.0 | Raw float32 |
| `F16` | 1 | 2 | 16.0 | Raw float16 |
| `BF16` | 1 | 2 | 16.0 | Brain float16 |
| `Q4_0` | 32 | 18 | 4.5 | `f16 delta` + `4-bit quants[16]` |
| `Q4_1` | 32 | 20 | 5.0 | `f16 delta` + `f16 min` + `4-bit quants[16]` |
| `Q5_0` | 32 | 22 | 5.5 | `f16 delta` + `high-bits[4]` + `4-bit quants[16]` |
| `Q5_1` | 32 | 24 | 6.0 | `f16 delta` + `f16 min` + `high-bits[4]` + `4-bit quants[16]` |
| `Q8_0` | 32 | 34 | 8.5 | `f16 delta` + `int8 quants[32]` |
| `Q8_1` | 32 | 36 | 9.0 | `f16 delta` + `f16 sum` + `int8 quants[32]` |

#### K-Quants (super-block size = 256)

| Type | Block Size | Bytes/Block | Bits/Weight | Components |
|------|-----------|-------------|-------------|------------|
| `Q2_K` | 256 | 84 | 2.625 | `f16 d, dmin` + `scales[16]` + `2-bit quants[64]` |
| `Q3_K` | 256 | 110 | 3.4375 | `f16 d` + `hmask[32]` + `3-bit quants[64]` + `scales[12]` |
| `Q4_K` | 256 | 144 | 4.5 | `f16 d, dmin` + `scales[12]` + `4-bit quants[128]` |
| `Q5_K` | 256 | 176 | 5.5 | `f16 d, dmin` + `scales[12]` + `high-bits[32]` + `5-bit quants[128]` |
| `Q6_K` | 256 | 210 | 6.5625 | `f16 d` + `low-6[128]` + `high-2[64]` + `int8 scales[16]` |
| `Q8_K` | 256 | 292 | 9.125 | `f32 d` + `int8 quants[256]` + `int16 bsums[16]` |

#### Importance-Matrix Quants (IQ, super-block size = 256 unless noted)

| Type | Block Size | Bytes/Block | Bits/Weight | Notes |
|------|-----------|-------------|-------------|-------|
| `IQ1_S` | 256 | 50 | 1.5625 | Extreme compression, lattice codebook |
| `IQ1_M` | 256 | 58 | 1.8125 | Improved IQ1 with signs |
| `IQ2_XXS` | 256 | 66 | 2.0625 | Ultra-low 2-bit |
| `IQ2_XS` | 256 | 74 | 2.3125 | Extra-small 2-bit with scales |
| `IQ2_S` | 256 | 82 | 2.5625 | Small 2-bit |
| `IQ3_XXS` | 256 | 98 | 3.0625 | Ultra-low 3-bit |
| `IQ3_S` | 256 | 110 | 3.4375 | Small 3-bit with signs |
| `IQ4_NL` | 32 | 18 | 4.5 | Non-linear 4-bit (lookup table) |
| `IQ4_XS` | 256 | 136 | 4.25 | Extra-small 4-bit |

#### Ternary & MX Quants

| Type | Block Size | Bytes/Block | Bits/Weight | Notes |
|------|-----------|-------------|-------------|-------|
| `TQ1_0` | 256 | 54 | 1.6875 | Ternary {-1, 0, +1} |
| `TQ2_0` | 256 | 66 | 2.0625 | 2-bit ternary |
| `MXFP4` | 32 | 17 | 4.25 | MX floating point 4-bit |

> **Source:** `gguf-py/gguf/constants.py:3900-3932` (`GGML_QUANT_SIZES`), `ggml/src/ggml.c:609-899` (`type_traits`)

---

## 5. Block Structures & Dequantization

### Legacy Block Structures

All legacy quants use a block size of 32 weights (`QK4_0 = QK4_1 = QK5_0 = QK5_1 = QK8_0 = QK8_1 = 32`).

#### `block_q4_0` — 18 bytes / 32 weights

```c
typedef struct {
    ggml_half d;            // delta scale factor (2 bytes)
    uint8_t qs[QK4_0 / 2]; // packed 4-bit quants, two per byte (16 bytes)
} block_q4_0;
```

**Dequantization formula (Q4_0):**
```
weight[i] = delta * (quant_nibble[i] - 8)
```

Each byte in `qs[]` stores two 4-bit unsigned values (0–15). Subtract 8 to center around zero, then scale by `delta`.

#### `block_q4_1` — 20 bytes / 32 weights

```c
typedef struct {
    ggml_half d;            // delta (2 bytes)
    ggml_half m;            // minimum (2 bytes)
    uint8_t qs[QK4_1 / 2]; // packed 4-bit quants (16 bytes)
} block_q4_1;
```

**Dequantization formula (Q4_1):**
```
weight[i] = delta * quant_nibble[i] + min
```

Asymmetric quantization — `min` stores the block minimum so quants don't need centering.

#### `block_q8_0` — 34 bytes / 32 weights

```c
typedef struct {
    ggml_half d;       // delta (2 bytes)
    int8_t qs[QK8_0];  // signed 8-bit quants (32 bytes)
} block_q8_0;
```

**Dequantization formula (Q8_0):**
```
weight[i] = delta * quant[i]
```

> **Source:** `ggml/src/ggml-common.h:170-240`

### K-Quant Super-Block Structures

K-quants use a super-block of 256 weights (`QK_K = 256`), subdivided into sub-blocks of 16 or 32 weights each with their own scales.

#### `block_q4_K` — 144 bytes / 256 weights

```c
typedef struct {
    ggml_half d;                      // super-block delta (2 bytes)
    ggml_half dmin;                   // super-block min (2 bytes)
    uint8_t scales[K_SCALE_SIZE];     // sub-block scales, packed (12 bytes)
    uint8_t qs[QK_K / 2];            // packed 4-bit quants (128 bytes)
} block_q4_K;
```

**Dequantization (Q4_K) — simplified:**
```c
// For each group of 64 weights (4 groups per super-block):
get_scale_min_k4(sub_idx, scales, &sc, &m);
d_sub = d * sc;          // sub-block scale
m_sub = dmin * m;        // sub-block min
weight[i] = d_sub * (qs[i] & 0xF) - m_sub;   // low nibble
weight[j] = d_sub * (qs[j] >> 4)  - m_sub;   // high nibble
```

The `scales[12]` array packs per-sub-block scale and minimum values using a custom encoding extracted by `get_scale_min_k4()`.

> **Source:** `ggml/src/ggml-quants.c:1352-1374` (full `dequantize_row_q4_K`)

#### `block_q2_K` — 84 bytes / 256 weights

```c
typedef struct {
    uint8_t scales[QK_K / 16]; // sub-block scales (16 bytes)
    uint8_t qs[QK_K / 4];     // 2-bit quants, 4 per byte (64 bytes)
    ggml_half d;               // super-block delta (2 bytes)
    ggml_half dmin;            // super-block min (2 bytes)
} block_q2_K;
```

#### `block_q6_K` — 210 bytes / 256 weights

```c
typedef struct {
    uint8_t ql[QK_K / 2];     // lower 4 bits of 6-bit quants (128 bytes)
    uint8_t qh[QK_K / 4];     // upper 2 bits of 6-bit quants (64 bytes)
    int8_t scales[QK_K / 16]; // signed sub-block scales (16 bytes)
    ggml_half d;               // super-block delta (2 bytes)
} block_q6_K;
```

> **Source:** `ggml/src/ggml-common.h:245-344`

### Dequantization Function Signatures

All dequantization functions follow the same pattern:

```c
void dequantize_row_<TYPE>(const block_<TYPE> *x, float *y, int64_t k);
```

Where `k` is the total number of weights (must be a multiple of the block size).

Key functions in `ggml/src/ggml-quants.c` and `ggml/src/ggml-quants.h`:

| Function | Input Block | Block Size |
|----------|------------|------------|
| `dequantize_row_q4_0` | `block_q4_0` | 32 |
| `dequantize_row_q4_1` | `block_q4_1` | 32 |
| `dequantize_row_q5_0` | `block_q5_0` | 32 |
| `dequantize_row_q5_1` | `block_q5_1` | 32 |
| `dequantize_row_q8_0` | `block_q8_0` | 32 |
| `dequantize_row_q2_K` | `block_q2_K` | 256 |
| `dequantize_row_q3_K` | `block_q3_K` | 256 |
| `dequantize_row_q4_K` | `block_q4_K` | 256 |
| `dequantize_row_q5_K` | `block_q5_K` | 256 |
| `dequantize_row_q6_K` | `block_q6_K` | 256 |
| `dequantize_row_iq2_xxs` | `block_iq2_xxs` | 256 |
| `dequantize_row_iq3_s` | `block_iq3_s` | 256 |
| `dequantize_row_iq4_nl` | `block_iq4_nl` | 32 |

> **Source:** `ggml/src/ggml-quants.h:43-70`

---

## 6. GGUF Parser Building Guide (Rust)

### Overview

Build a minimal GGUF parser as a Rust CLI tool (`gguf-inspector`) that can:
- Read and validate the header
- Parse all KV metadata pairs
- Parse all tensor info entries
- Locate and optionally read tensor data (via `memmap2`)
- Compute SHA-256 hashes of individual tensors

### Project Setup

```bash
cargo init gguf-inspector
cd gguf-inspector
```

**Cargo.toml dependencies:**

```toml
[package]
name = "gguf-inspector"
version = "0.1.0"
edition = "2021"

[dependencies]
byteorder = "1"                              # Endian-aware binary reads
memmap2 = "0.9"                              # Memory-mapped file access
sha2 = "0.10"                                # SHA-256 hashing
half = "2"                                   # f16 <-> f32 conversion
clap = { version = "4", features = ["derive"] }  # CLI argument parsing
```

**Suggested module layout:**

```
src/
  main.rs          -- CLI entry point (clap), orchestration
  header.rs        -- 24-byte header parsing + endianness detection
  types.rs         -- Enums, constants, QUANT_SIZES table
  kv.rs            -- KV pair parsing (all 13 value types + arrays)
  tensor_info.rs   -- Tensor info entry parsing
  hash.rs          -- SHA-256 per-tensor and full-file hashing
  dequant.rs       -- Q4_0 dequantizer
```

### Day 1: Header Parsing

**Goal:** Read and validate the 24-byte fixed header.

```rust
use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use std::io::{self, BufReader, Read, Seek};
use std::fs::File;

const GGUF_MAGIC: u32 = 0x46554747;

/// Detected byte order of the GGUF file.
#[derive(Clone, Copy, PartialEq)]
enum Endian { Little, Big }

/// Parsed GGUF header (24 bytes).
struct GgufHeader {
    version: u32,
    tensor_count: u64,
    kv_count: u64,
    endian: Endian,
}

/// Wraps a BufReader and dispatches reads based on detected endianness.
struct GgufReader<R: Read + Seek> {
    inner: BufReader<R>,
    endian: Endian,
}

impl<R: Read + Seek> GgufReader<R> {
    fn read_u32(&mut self) -> io::Result<u32> {
        match self.endian {
            Endian::Little => self.inner.read_u32::<LittleEndian>(),
            Endian::Big    => self.inner.read_u32::<BigEndian>(),
        }
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        match self.endian {
            Endian::Little => self.inner.read_u64::<LittleEndian>(),
            Endian::Big    => self.inner.read_u64::<BigEndian>(),
        }
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        match self.endian {
            Endian::Little => self.inner.read_f32::<LittleEndian>(),
            Endian::Big    => self.inner.read_f32::<BigEndian>(),
        }
    }

    fn read_f64(&mut self) -> io::Result<f64> {
        match self.endian {
            Endian::Little => self.inner.read_f64::<LittleEndian>(),
            Endian::Big    => self.inner.read_f64::<BigEndian>(),
        }
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        self.inner.stream_position()
    }
}

fn read_header(reader: &mut GgufReader<File>) -> io::Result<GgufHeader> {
    // Magic is always read as little-endian (it's the fixed sentinel).
    let magic = reader.inner.read_u32::<LittleEndian>()?;
    assert!(magic == GGUF_MAGIC, "Bad magic: {magic:#x}");

    // Read version as little-endian first; detect swapped endianness.
    let version_raw = reader.inner.read_u32::<LittleEndian>()?;
    let endian = if version_raw & 0xFFFF == 0 {
        Endian::Big   // file uses swapped byte order
    } else {
        Endian::Little
    };
    let version = if endian == Endian::Big { version_raw.swap_bytes() } else { version_raw };
    assert!(version == 2 || version == 3, "Unsupported version: {version}");

    reader.endian = endian;
    let tensor_count = reader.read_u64()?;
    let kv_count = reader.read_u64()?;

    Ok(GgufHeader { version, tensor_count, kv_count, endian })
}
```

**Validation checks:**
- Magic must be `0x46554747`
- Version must be 2 or 3
- If `version & 0xFFFF == 0`, file uses big-endian byte order — all subsequent reads must swap
- Counts should be reasonable (not billions)

### Day 2: String & Value Type Reading

**Goal:** Implement readers for each `GGUFValueType`.

```rust
/// All possible GGUF metadata value types.
enum GgufValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    Str(String),
    Array { elem_type: u32, values: Vec<GgufValue> },
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

fn read_string(reader: &mut GgufReader<File>) -> io::Result<String> {
    let length = reader.read_u64()? as usize;
    let mut buf = vec![0u8; length];
    reader.inner.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_value(reader: &mut GgufReader<File>, type_id: u32) -> io::Result<GgufValue> {
    match type_id {
        0  => Ok(GgufValue::UInt8(reader.inner.read_u8()?)),
        1  => Ok(GgufValue::Int8(reader.inner.read_i8()?)),
        2  => Ok(GgufValue::UInt16(reader.read_u32()? as u16)),  // read via endian helper
        3  => Ok(GgufValue::Int16(reader.read_u32()? as i16)),
        4  => Ok(GgufValue::UInt32(reader.read_u32()?)),
        5  => Ok(GgufValue::Int32(reader.read_u32()? as i32)),
        6  => Ok(GgufValue::Float32(reader.read_f32()?)),
        7  => Ok(GgufValue::Bool(reader.inner.read_u8()? != 0)),
        8  => Ok(GgufValue::Str(read_string(reader)?)),
        9  => {
            // ARRAY: u32 elem_type + u64 count + count elements
            let elem_type = reader.read_u32()?;
            let count = reader.read_u64()? as usize;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_value(reader, elem_type)?);
            }
            Ok(GgufValue::Array { elem_type, values })
        }
        10 => Ok(GgufValue::UInt64(reader.read_u64()?)),
        11 => Ok(GgufValue::Int64(reader.read_u64()? as i64)),
        12 => Ok(GgufValue::Float64(reader.read_f64()?)),
        _  => Err(io::Error::new(io::ErrorKind::InvalidData,
                                  format!("Unknown value type: {type_id}"))),
    }
}
```

> **Note on type IDs 2 and 3:** The 16-bit reads should use proper 2-byte endian-aware reads. The snippet above simplifies them — in your real implementation, add `read_u16()` and `read_i16()` methods to `GgufReader`.

### Day 3: KV Pair Parsing

**Goal:** Parse all key-value metadata entries.

```rust
use std::collections::HashMap;

fn read_kv_pairs(
    reader: &mut GgufReader<File>,
    kv_count: u64,
) -> io::Result<(Vec<(String, GgufValue)>, HashMap<String, usize>)> {
    let mut pairs = Vec::with_capacity(kv_count as usize);
    let mut index = HashMap::new();

    for i in 0..kv_count as usize {
        let key = read_string(reader)?;
        let value_type = reader.read_u32()?;
        let value = read_value(reader, value_type)?;
        index.insert(key.clone(), i);
        pairs.push((key, value));
    }

    Ok((pairs, index))
}
```

**Things to inspect in the output:**
- `general.architecture` — confirms model family
- `general.file_type` — identifies quantization level
- `{arch}.context_length`, `{arch}.embedding_length`, `{arch}.block_count` — model dimensions
- `tokenizer.ggml.model` — tokenizer type

### Day 4: Tensor Info Parsing

**Goal:** Parse tensor metadata (name, shape, type, offset).

```rust
struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    dtype: u32,       // GGMLQuantizationType enum value
    offset: u64,      // relative to data section start
    n_elements: u64,  // computed: product of dims
    byte_size: u64,   // computed from QUANT_SIZES
}

/// Quantization type -> (block_size, type_size_bytes).
fn quant_sizes(dtype: u32) -> Option<(u64, u64)> {
    match dtype {
        0  => Some((1, 4)),       // F32
        1  => Some((1, 2)),       // F16
        2  => Some((32, 18)),     // Q4_0
        3  => Some((32, 20)),     // Q4_1
        6  => Some((32, 22)),     // Q5_0
        7  => Some((32, 24)),     // Q5_1
        8  => Some((32, 34)),     // Q8_0
        10 => Some((256, 84)),    // Q2_K
        11 => Some((256, 110)),   // Q3_K
        12 => Some((256, 144)),   // Q4_K
        13 => Some((256, 176)),   // Q5_K
        14 => Some((256, 210)),   // Q6_K
        15 => Some((256, 288)),   // Q8_K
        30 => Some((1, 2)),       // BF16
        _  => None,
    }
}

fn read_tensor_infos(
    reader: &mut GgufReader<File>,
    tensor_count: u64,
) -> io::Result<Vec<TensorInfo>> {
    let mut tensors = Vec::with_capacity(tensor_count as usize);

    for _ in 0..tensor_count {
        let name = read_string(reader)?;
        let n_dims = reader.read_u32()?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(reader.read_u64()?);
        }
        let dtype = reader.read_u32()?;
        let offset = reader.read_u64()?;

        let n_elements: u64 = dims.iter().product();
        let (block_size, type_size) = quant_sizes(dtype)
            .unwrap_or_else(|| panic!("Unknown quant type: {dtype}"));
        let byte_size = (n_elements / block_size) * type_size;

        tensors.push(TensorInfo {
            name, n_dims, dims, dtype, offset, n_elements, byte_size,
        });
    }

    Ok(tensors)
}
```

### Day 5: Data Section & Tensor Location

**Goal:** Calculate the data section start and read raw tensor bytes via memory mapping.

```rust
use memmap2::Mmap;

/// Calculate data section offset with proper alignment padding.
fn compute_data_offset(reader: &mut GgufReader<File>, alignment: u64) -> io::Result<u64> {
    let pos = reader.stream_position()?;
    let padding = pos % alignment;
    let data_offset = if padding != 0 { pos + alignment - padding } else { pos };
    Ok(data_offset)
}

/// Get a slice of raw bytes for a single tensor from the memory map.
fn get_tensor_data<'a>(mmap: &'a Mmap, data_offset: u64, tensor: &TensorInfo) -> &'a [u8] {
    let start = (data_offset + tensor.offset) as usize;
    let end = start + tensor.byte_size as usize;
    &mmap[start..end]
}

/// Hex-dump the first `max_bytes` of a tensor's data.
fn hexdump(data: &[u8], max_bytes: usize) {
    for (i, chunk) in data[..max_bytes.min(data.len())].chunks(16).enumerate() {
        print!("{:08x}  ", i * 16);
        for byte in chunk {
            print!("{:02x} ", byte);
        }
        println!();
    }
}
```

### Day 6: Alignment Handling & Full Parse

**Goal:** Put it all together with proper alignment.

```rust
struct ParsedGguf {
    header: GgufHeader,
    metadata: Vec<(String, GgufValue)>,
    metadata_index: HashMap<String, usize>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
}

fn parse_gguf(path: &str) -> io::Result<ParsedGguf> {
    let file = File::open(path)?;
    let mut reader = GgufReader {
        inner: BufReader::new(file),
        endian: Endian::Little, // will be updated by read_header
    };

    let header = read_header(&mut reader)?;
    let (metadata, metadata_index) = read_kv_pairs(&mut reader, header.kv_count)?;
    let tensors = read_tensor_infos(&mut reader, header.tensor_count)?;

    // Determine alignment from metadata (default 32).
    let alignment = metadata.iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| match v {
            GgufValue::UInt32(a) => Some(*a as u64),
            _ => None,
        })
        .unwrap_or(32);

    let data_offset = compute_data_offset(&mut reader, alignment)?;

    Ok(ParsedGguf { header, metadata, metadata_index, tensors, data_offset })
}
```

### Day 7: Verification Against Reference

Compare your parser output against the reference implementation:

```bash
# Using the bundled gguf-py tool
python -m gguf.scripts.gguf_dump your_model.gguf > reference.txt

# Your Rust tool
cargo run -- your_model.gguf > ours.txt

# Diff them
diff reference.txt ours.txt
```

Every metadata key/value and every tensor name/shape/type/offset should match.

> **Reference parser:** `gguf-py/gguf/gguf_reader.py:132-185`

---

## 7. SHA-256 Hashing Approach (Rust)

### Per-Tensor Hashing

Hash individual tensor data blocks for integrity verification:

```rust
use sha2::{Sha256, Digest};

fn hash_tensor(mmap: &Mmap, data_offset: u64, tensor: &TensorInfo) -> String {
    let data = get_tensor_data(mmap, data_offset, tensor);
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}
```

### Full-File Hashing

For overall file integrity, stream 64 KB chunks from the memory map:

```rust
fn hash_file(mmap: &Mmap) -> String {
    let mut hasher = Sha256::new();
    for chunk in mmap.chunks(64 * 1024) {
        hasher.update(chunk);
    }
    format!("{:x}", hasher.finalize())
}
```

### Data-Section-Only Hashing

Hash only the tensor data region (useful for comparing files with different metadata):

```rust
fn hash_data_section(mmap: &Mmap, data_offset: u64) -> String {
    let mut hasher = Sha256::new();
    let data_section = &mmap[data_offset as usize..];
    for chunk in data_section.chunks(64 * 1024) {
        hasher.update(chunk);
    }
    format!("{:x}", hasher.finalize())
}
```

---

## 8. Critical File References

### GGUF Format Definition

| File | What's There |
|------|-------------|
| `gguf-py/gguf/constants.py` | `GGUF_MAGIC`, `GGUF_VERSION`, `GGUFValueType`, `GGMLQuantizationType`, `GGML_QUANT_SIZES`, `Keys` namespace |
| `gguf-py/gguf/gguf_reader.py` | Reference GGUF parser (Python, uses numpy memmap) |
| `gguf-py/gguf/gguf_writer.py` | GGUF file writer |
| `gguf-py/gguf/scripts/gguf_dump.py` | CLI tool to dump GGUF file contents |

### Quantization Implementation

| File | What's There |
|------|-------------|
| `ggml/include/ggml.h` | `ggml_type` enum, public API declarations |
| `ggml/src/ggml.c` | `type_traits[]` array — block sizes, type sizes, function pointers |
| `ggml/src/ggml-common.h` | All `block_*` struct definitions (`block_q4_0`, `block_q4_K`, etc.) |
| `ggml/src/ggml-quants.h` | Dequantize/quantize function declarations |
| `ggml/src/ggml-quants.c` | Dequantize/quantize implementations (reference C code) |

### Model Loading

| File | What's There |
|------|-------------|
| `src/llama-model-loader.h` | `llama_model_loader` class declaration |
| `src/llama-model-loader.cpp` | GGUF file loading, tensor mapping, mmap handling |
| `src/llama-model.cpp` | Model construction from loaded tensors |

### Conversion & Quantization Tools

| File | What's There |
|------|-------------|
| `convert_hf_to_gguf.py` | HuggingFace → GGUF converter |
| `tools/quantize/quantize.cpp` | CLI quantization tool (post-conversion) |

### Rust Crate Documentation

| Crate | Purpose | Docs |
|-------|---------|------|
| `byteorder` | Endian-aware `ReadBytesExt` trait | docs.rs/byteorder |
| `memmap2` | Memory-mapped file I/O | docs.rs/memmap2 |
| `sha2` | SHA-256 via `Digest` trait | docs.rs/sha2 |
| `half` | `f16::from_bits().to_f32()` for ggml_half | docs.rs/half |
| `clap` | CLI argument parsing with derive macros | docs.rs/clap |

---

## 9. Deliverables Checklist

### Week 5: GGUF Format Mastery

- [ ] **Read and annotate** the GGUF spec: understand every field in the header, KV pairs, and tensor info sections
- [ ] **Hex-dump analysis:** Open a small GGUF file in a hex editor and manually identify the magic bytes, version, counts, first KV pair, and first tensor info
- [ ] **Scaffold the Rust project:** `cargo init gguf-inspector`, add dependencies (`byteorder`, `memmap2`, `sha2`, `half`, `clap`), set up module layout
- [ ] **Build the parser:** Implement `read_header()`, `read_string()`, `read_value()`, `read_kv_pairs()`, `read_tensor_infos()` using `GgufReader` with endianness dispatch
- [ ] **Verify metadata:** Parse a real model file and confirm `general.architecture`, tensor count, and tokenizer keys match `gguf_dump` output
- [ ] **Tensor location:** Calculate data section offset with alignment, memory-map the file, read raw bytes for one tensor, confirm byte count matches expected `(n_elements / block_size) * type_size`

### Week 6: Quantization Deep Dive

- [ ] **Map every quant type:** For each type in the comparison table above, identify its `ggml_type` enum value, block struct, and dequantize function
- [ ] **Trace Q4_0 dequantization:** Step through `dequantize_row_q4_0` line by line — extract delta, unpack nibbles, reconstruct floats
- [ ] **Trace Q4_K dequantization:** Step through `dequantize_row_q4_K` — understand super-block scales, `get_scale_min_k4`, sub-block iteration
- [ ] **Compare legacy vs. K-quants:** Write a summary of why K-quants are more accurate (per-sub-block scales, larger blocks, scale packing)
- [ ] **SHA-256 integrity:** Hash at least 3 tensors from a GGUF file using your Rust tool and record hashes for future verification
- [ ] **Build a Q4_0 dequantizer in Rust:** Manually dequantize one block of Q4_0 data — use `half::f16::from_bits().to_f32()` for the delta, unpack nibbles with bitwise ops, and compare the 32 reconstructed floats against reference output:

```rust
use half::f16;

/// Dequantize a single Q4_0 block (18 bytes -> 32 f32 values).
/// Mirrors ggml/src/ggml-quants.c:307-325 (dequantize_row_q4_0).
fn dequantize_q4_0_block(block_data: &[u8]) -> [f32; 32] {
    assert!(block_data.len() >= 18, "Q4_0 block must be 18 bytes");

    // First 2 bytes: f16 delta scale factor
    let d = f16::from_bits(u16::from_le_bytes([block_data[0], block_data[1]])).to_f32();
    let qs = &block_data[2..18]; // 16 bytes of packed 4-bit quants

    let mut output = [0.0f32; 32];
    for j in 0..16 {
        // Low nibble -> positions [0..16), high nibble -> positions [16..32)
        let x0 = (qs[j] & 0x0F) as i32 - 8;   // center around zero
        let x1 = (qs[j] >> 4) as i32 - 8;
        output[j]      = x0 as f32 * d;
        output[j + 16] = x1 as f32 * d;
    }
    output
}

/// Dequantize an entire Q4_0 tensor row.
fn dequantize_q4_0_row(data: &[u8], n_elements: u64) -> Vec<f32> {
    let n_blocks = (n_elements / 32) as usize;
    let mut result = Vec::with_capacity(n_elements as usize);
    for i in 0..n_blocks {
        let block = &data[i * 18..(i + 1) * 18];
        result.extend_from_slice(&dequantize_q4_0_block(block));
    }
    result
}
```

- [ ] **Study IQ quants conceptually:** Read the IQ2/IQ3 block structures — understand that they use lookup tables and lattice codebooks rather than simple linear scaling

---

*Study guide generated from llama.cpp source — cross-reference against the codebase files listed in [Section 8](#8-critical-file-references) for the most up-to-date details.*
