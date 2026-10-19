# Independent operation module example

This C99 module supplies `example.sequence.prefix_sum` and
`example.sequence.exclusive_sum`, version 1.0.0. Both take one contiguous rank-1
F32 input and a same-sized output. Capability bit 0 means deterministic serial
addition. Input `[1,2,3,4]` produces `[1,3,6,10]` or `[0,1,3,6]`. CPU scratch is
one caller-owned float; no model weights or downloads are needed.

Build using the installed SDK (paths below are examples):

```sh
cmake -S sdk -B build-module-sdk -DCMAKE_INSTALL_PREFIX="$PWD/build-module-sdk/install"
cmake --install build-module-sdk
cmake -S plugins/scan_module -B build-scan -DCMAKE_PREFIX_PATH="$PWD/build-module-sdk/install"
cmake --build build-scan
```

`build-scan/package/.gkmetadata` is an **unsigned template** with placeholder
artifact hashes. Set its publisher to your own identity and update the matching
publisher string in `scan.c` before distributing it. Signing recomputes hashes:

```sh
guild kernel keygen --json
# Save private_seed_hex to /private/path/module.seed (owner-readable only).
guild kernel trust-key --publisher org.illatio.guild --key-id example --pubkey PUBLIC_KEY
guild kernel sign build-scan/package --key-id example --seed-file /private/path/module.seed --bundle build-scan/scan.gkmodule
guild kernel install build-scan/scan.gkmodule
guild kernel inspect example.scan --json
guild kernel disable example.scan
guild kernel remove example.scan
```

Use a private, locally generated seed. The repository test key is only a public
test vector; it is not a distribution identity. Tests use isolated trust stores.

Consumers copy `guild_operation_api_v1_t` from the host API during initialization,
then populate a `guild_operation_query_v1_t` with operation ID, version bounds,
backends, capabilities, budgets and tensor descriptors. Call `resolve`, allocate
the scratch reported by `scratch`, then `execute` and `release`. A complete
consumer is in `tests/core/gkmodule_test.cpp`; ownership and stream requirements
are in [GKMODULE.md](../../docs/architecture/GKMODULE.md).

The example is built as two independent providers in tests to exercise priority,
fallback and dependency pins. The higher-priority provider forwards through the
copied C service to the pinned base implementation, proving extension-to-extension
resolution and execution. Its `fast` fixture name denotes priority, not measured speed.
`scan_cuda.cu` is a separate CUDA lifecycle fixture;
the standalone CMake project intentionally builds the CPU module.
