# GPACK v1 Model Packaging Format (`GPCK` / `GPKE`)

GPACK (`.gpack`) is Guild's portable, versioned, hardware-independent model container format introduced in Phase 4. It consolidates multi-shard GGUF or safetensors models, pre-parsed binary tensor and dense/expert layout indices, precomputed `GBT1` binary tokenizer assets, and optional auxiliary components (PLE n-gram tables and MTP speculative draft layers) into a single page-aligned, `mmap`-ready artifact.

GPACK is **strictly hardware-independent**: it never embeds machine-specific CUDA SM architectures, host NUMA node bindings, or autotuned kernel winners. Hardware-specific artifacts belong exclusively to the disposable GEXEC (`GEXC`) cache layer (`docs/architecture/GEXEC_CONTRACT.md`).

---

## 1. Binary Container Layout

All integers in GPACK v1 are little-endian. All data payload sections are aligned to `4096` bytes (OS page and `O_DIRECT` sector alignment) or `64` bytes (CPU cacheline and AVX-512 alignment).

```text
+-----------------------------------------------------------------------+ 0x0000
| GPackHeaderV1 (64 bytes, magic = "GPCK" / 0x4B435047)                 |
+-----------------------------------------------------------------------+ 0x0040 (aligned)
| Section 0: MetadataKvBin       (64B aligned, binary key-value table)  |
| Section 1: TensorIndexBin      (64B aligned, sorted tensor directory) |
| Section 2: DenseIndexBin       (64B aligned, pre-parsed dense rows)   |
| Section 3: DensePackBlob       (4096B aligned, packed dense weights)  |
| Section 4: NativeDenseIndexBin (64B aligned, single-arena MMVQ index) |
| Section 5: NativeDenseBlob     (4096B aligned, contiguous MMVQ arena) |
| Section 6: EmbedBlob           (4096B aligned, token_embd.weight)     |
| Section 7: HeadBlob            (4096B aligned, output.weight)         |
| Section 8: PleBlob             (4096B aligned, optional PLE table)    |
| Section 9: ExpertLayoutBin     (64B aligned, per-layer expert table)  |
| Section 10: ExpertBlobs        (4096B aligned, optional [l][e] blobs) |
| Section 11: TokenizerBin       (64B aligned, GBT1 binary tokenizer)   |
| Section 12: TokenizerAssets    (64B aligned, raw tokenizer files)     |
| Section 13: MtpBlob            (4096B aligned, optional MTP draft)    |
| Section 14: ManifestJson       (64B aligned, guild.gpack.v1 JSON)     |
+-----------------------------------------------------------------------+ section_table_offset (64B aligned)
| GPackSectionEntryV1[section_count] (96 bytes per entry, with SHA-256) |
+-----------------------------------------------------------------------+ footer_offset (64B aligned)
| GPackFooterV1 (64 bytes, magic = "GPKE" / 0x454B5047)                 |
+-----------------------------------------------------------------------+ EOF
```

### Header (`GPackHeaderV1`, 64 bytes)
- `magic`: `0x4B435047` (`"GPCK"`)
- `format_version`: `1` (`uint16_t`)
- `min_reader_version`: `1` (`uint16_t`) — readers with `kGPackVersionV1 < min_reader_version` fail closed with a clear version error.
- `flags`: Bitmask of `GPackFlags`:
  - `GUILD_GPACK_FLAG_LITTLE_ENDIAN` (`1 << 0`)
  - `GUILD_GPACK_FLAG_HAS_DENSE_PACK` (`1 << 1`)
  - `GUILD_GPACK_FLAG_HAS_NATIVE_DENSE` (`1 << 2`)
  - `GUILD_GPACK_FLAG_HAS_EXPERT_BLOBS` (`1 << 3`)
  - `GUILD_GPACK_FLAG_HAS_TOKENIZER_BIN` (`1 << 4`)
  - `GUILD_GPACK_FLAG_HAS_MTP` (`1 << 5`)
  - `GUILD_GPACK_FLAG_HAS_PLE` (`1 << 6`)
  - `GUILD_GPACK_FLAG_EXTERNAL_EXPERTS` (`1 << 7`)
- `section_count`: Number of `GPackSectionEntryV1` records in the section directory.
- `section_table_offset`: Byte offset of the section directory.
- `manifest_offset`, `manifest_size`: Byte offset and length of the canonical `ManifestJson` section.
- `total_file_size`: Exact expected file size in bytes (detects file truncation or trailing garbage immediately upon open).

### Section Directory Entry (`GPackSectionEntryV1`, 96 bytes)
- `kind`: `GPackSectionKind` (`uint32_t`)
- `flags`: Reserved section flags (`uint32_t`)
- `offset`: Byte offset from start of file (`uint64_t`, must be a multiple of `alignment`)
- `size`: Section payload size in bytes (`uint64_t`)
- `alignment`: Required byte alignment (`64` or `4096`)
- `item_count`: Number of logical items in the section (`uint64_t`)
- `sha256[32]`: Full SHA-256 digest of the section payload bytes
- `name[24]`: Null-terminated ASCII section identifier

### Footer (`GPackFooterV1`, 64 bytes)
- `manifest_offset`, `section_table_offset`, `total_payload_bytes`
- `manifest_sha256[32]`: SHA-256 digest of the `ManifestJson` section (which itself embeds the SHA-256 digests of every other section, forming a two-level Merkle tree over the entire file).
- `footer_magic`: `0x454B5047` (`"GPKE"`)

---

## 2. Precomputed Portable Sections

### 2.1 `DenseIndexBin` & `DensePackBlob`
Replaces text parsing of `index.txt` (3 separate `sscanf` passes during GGUF+pack load) with a binary table of `GPackDenseRowV1` records containing precomputed `skip_in_native` flags and both uncompacted (`dense_pool_bytes`) and compacted (`dense_compacted_pool_bytes`) VRAM arena sizes. `WeightTable::load_gpack` copies or expands BF16/F16->F32 weights directly from the `mmap`'d `DensePackBlob` into VRAM without opening external files.

### 2.2 `NativeDenseIndexBin` & `NativeDenseBlob`
Replaces 262–305 individual `cudaMalloc` and `cudaMemcpy` calls in `NativeDense::load` with a pre-laid-out 256-byte-aligned contiguous weight arena (`native_dense_arena_bytes`). At model load time, `NativeDense::load_gpack` allocates **one** device arena via `cudaMalloc` and uploads all native MMVQ weights in **one** `cudaMemcpyAsync` transfer.

### 2.3 `EmbedBlob`, `HeadBlob`, and `PleBlob`
Each section begins with a 64-byte `GPackSingleTensorHeaderV1` (`ggml_type`, `ne0`, `ne1`, `byte_size`, `data_offset = 64`), followed immediately by the tensor payload aligned within the 4096-byte section.
- `NativeEmbed::load_gpack` and `NativeHead::load_gpack` bind directly to the `mmap`'d pointers without re-opening GGUF shards.
- `PleTable::open_raw` binds the PLE n-gram table at its exact file offset inside the `.gpack` container, supporting both `PleIo::Direct` (4 KiB page-aligned unbuffered SSD reads + set-associative row cache) and `PleIo::Mmap`.

### 2.4 `ExpertLayoutBin` & `ExpertBlobs` (Embedded vs. External Modes)
GPACK supports two routed-expert storage modes:
1. **Embedded Experts (default)**: All routed experts across all layers (`n_layers * n_expert`) are concatenated in canonical `[gate_slice | up_slice | down_slice]` order into the 4096-byte-aligned `ExpertBlobs` section.
   - Enables true single-file portability.
   - Supports both NUMA-interleaved RAM resident caching (`FileExpertSource::open_gpack`) and zero-copy direct `mmap` execution (`GUILD_EXPERT_MMAP=1`).
2. **External Experts (`--external-experts`)**: `ExpertLayoutBin` records the pre-validated per-layer byte offsets and shard filenames in the source GGUF shards alongside the `.gpack` file, omitting `ExpertBlobs`.
   - Builds in ~12 seconds for Ornith-1.5-35B (1.54 GiB `.gpack` instead of 19.71 GiB) without duplicating 18–55 GiB of expert weights on disk.

### 2.5 `TokenizerBin` (`GBT1`) & `TokenizerAssets`
The `TokenizerBin` section stores the precomputed `GBT1` binary tokenizer (`magic = 0x31544247`):
- Header (`GbtHeaderV1`): `vocab_size`, `n_merges`, `eos_token_id`, `flags`, `vocab_blob_bytes`, `chat_template_bytes`.
- Contiguous `uint32_t offsets[vocab_size + 1]` and `uint8_t token_types[vocab_size]`.
- Contiguous UTF-8 vocabulary string blob (`vocab_blob_bytes`).
- Pre-resolved integer BPE merge triples `int32_t triples[n_merges][3]` (`left_token_id`, `right_token_id`, `merged_token_id`) ordered by merge rank `0 .. n_merges - 1`.
- Raw Jinja chat template string (`chat_template_bytes`).

Loading `GBT1` avoids parsing 11 MiB of `vocab.json`, `merges.txt`, and `token_type.json`, reducing 248,320-token vocabulary initialization from `355 ms` to `69 ms` (**5.1x faster**) while producing 100% identical token IDs.

---

## 3. Atomic Conversion, Verification, and GGUF Fallback

- **Crash-Safe Atomic Conversion**: `GPackBuilder::build` writes to `<output_path>.tmp.<pid>`, flushes and `fsync`s the file descriptor, verifies the written container headers and footer before closing, and atomically renames it to `<output_path>`. Any conversion failure unlinks the temporary file so a partial `.gpack` never exists at the destination path.
- **Two-Tier Integrity Verification**:
  - *Quick open* (`verify_all_hashes = false`, used at normal startup): Validates header/footer magics, format version, `min_reader_version`, exact `total_file_size`, section bounds/alignments, and the `ManifestJson` SHA-256 (~1–2 ms).
  - *Full verification* (`verify_all_hashes = true` or `guild verify <file.gpack>`): Streams every section payload through SHA-256 and checks every `GPackSectionEntryV1::sha256` digest.
- **Automatic GGUF Fallback**: In `GuildModel::load`, if `paths.gpack_path` is unreadable or fails validation **and** `paths.primary_model_path` (GGUF) is present, Guild logs a warning and falls back cleanly to the GGUF + pack loader. If no fallback GGUF is configured, `GuildModel::load` fails closed with the exact validation error.

---

## 4. CLI Usage

```bash
# Convert a single or multi-shard GGUF (+ pack directory) to a self-contained .gpack:
guild convert /path/to/Ornith-1.5-35B-Q4_K_M.gguf \
  --pack-dir /path/to/packs/ornith-1.5-35b \
  --name ornith-1.5-35b \
  -o /path/to/ornith-1.5-35b.gpack --json

# Convert with external experts (keeps routed experts in GGUF shards):
guild convert /path/to/Qwen3.8-Flash-Next-00001-of-00003.gguf \
  --pack-dir /path/to/packs/unsloth-ud-iq4_xs \
  --mtp-dir /path/to/mtp/rt \
  --external-experts \
  -o /path/to/qwen3.8-flash-next-ext.gpack --json

# Import a safetensors directory or model.safetensors.index.json into .gpack:
guild convert /path/to/safetensors_dir -o /path/to/model.gpack

# Verify SHA-256 digests of all sections (or --quick for header/manifest check):
guild verify /path/to/ornith-1.5-35b.gpack --json

# Inspect manifest metadata and section directory:
guild inspect /path/to/ornith-1.5-35b.gpack --json

# Run or serve directly from a .gpack file:
guild run /path/to/ornith-1.5-35b.gpack -p "Hello!"
guild serve /path/to/ornith-1.5-35b.gpack --port 8080
```
