# Security Policy

Guild is a native inference runtime that loads large model files, allocates significant amounts of CPU and GPU memory, and can expose an HTTP API.

Security issues should be reported privately whenever public disclosure could put users at risk.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository:

https://github.com/guild-inference/Guild/security/advisories/new

Include as much of the following information as possible:

- affected Guild commit or version
- operating system
- CPU and GPU
- build configuration
- steps to reproduce
- relevant model or model format
- request or input that triggers the issue
- expected behavior
- observed behavior
- potential security impact

Please do not open a public issue for vulnerabilities that could be actively exploited.

Ordinary bugs, hardening suggestions, crashes without a security impact, and performance problems may be reported through the normal issue tracker:

https://github.com/guild-inference/Guild/issues

## Supported versions

Guild is currently under active development and does not yet maintain long-term security branches.

Security fixes are normally applied to the latest development version and included in subsequent releases.

Older commits or releases may not receive backported fixes.

## Security status

Guild is a young project and has not undergone an independent external security audit.

Although the project includes tests and defensive validation, users should treat Guild as developing systems software, particularly when:

- loading models from untrusted sources
- exposing the HTTP server beyond localhost
- using experimental model architectures
- using experimental numeric formats or kernels
- running development or unreviewed builds

## Default network exposure

Guild's HTTP server is intended to bind to localhost by default.

A typical local endpoint is:

```text
http://127.0.0.1:11434
```

Binding Guild to an external interface, for example:

```bash
guild serve <model> --host 0.0.0.0
```

may make the server reachable by other devices.

Do not expose Guild directly to an untrusted network unless you understand and control the surrounding network security.

For remote access, consider placing Guild behind an authenticated reverse proxy, VPN, SSH tunnel, firewall, or other access-control layer.

Future Guild releases may provide additional native authentication mechanisms.

## HTTP server protections

Guild's native HTTP server includes defensive limits intended to reduce accidental or malicious resource exhaustion.

Current request limits include:

- maximum request headers: 64 KiB
- maximum number of headers: 100
- maximum request body: 32 MiB
- validation of `Content-Length`
- bounded connection handling
- clean handling of disconnected clients

These limits are defense-in-depth measures and should not be treated as a complete security boundary.

The server also uses protections intended to prevent a disconnected client from terminating the Guild process while responses are being streamed.

## Model files

Model files should be treated as untrusted input.

Guild parses model metadata, tensor descriptions, manifests, quantization metadata, tokenizer data, and other model-controlled structures.

A malicious or malformed model may attempt to trigger:

- invalid tensor dimensions
- integer overflow
- excessive allocations
- out-of-bounds accesses
- path traversal
- malformed tokenizer data
- malformed metadata
- denial of service
- excessive CPU, GPU, RAM, or disk use

Guild should validate model-controlled values before using them for memory allocation, pointer arithmetic, file access, or kernel dispatch.

Please report any model file capable of causing memory corruption or escaping expected filesystem boundaries as a security vulnerability.

## Model manifests and local storage

Guild uses a native model registry and local model store.

The model store validates manifest-controlled file names and rejects unsafe path components such as:

```text
..
/
\
```

Model files may be stored using content-addressed blobs and references.

Guild should never allow a model manifest to cause arbitrary filesystem paths outside the configured model store to be deleted or overwritten.

Imported local models may use references or symbolic links to avoid copying very large files. Users should therefore avoid importing files from untrusted or unexpectedly mutable locations.

## Downloads

Guild's native downloader uses HTTPS for remote model downloads.

Downloads are staged as incomplete files and finalized only after successful completion.

Where a manifest provides an expected SHA-256 hash, Guild verifies the downloaded file before accepting it.

Interrupted downloads may be resumed.

A model source using HTTPS is not automatically trustworthy. Prefer models from sources you recognize, and use hash-verified manifests where possible.

## SHA-256

Guild contains a native SHA-256 implementation used for model and blob verification.

Hash verification establishes that downloaded data matches the expected content.

It does not establish that the expected content itself is safe.

## GPU and native kernels

Guild contains native C++ and CUDA code and may execute highly optimized hardware-specific kernels.

Incorrect tensor metadata, unsupported instruction sets, malformed execution plans, or kernel bugs may result in crashes or memory corruption.

Guild performs hardware capability checks before entering instruction-set-specific code paths.

For example, CPU kernels requiring particular AVX-512 capabilities must not execute before the relevant CPU and operating-system state support has been validated.

Experimental hardware paths should be treated accordingly.

## Memory planner

Guild's execution memory planner determines placement of:

- model weights
- routed experts
- GPU expert cache
- host expert residency
- file-backed experts
- KV cache
- GPU staging buffers
- speculative decoding resources
- prompt-processing scratch memory

The planner attempts to reject configurations that exceed expected memory limits.

These calculations are estimates and are not a security boundary against hostile workloads or hostile model files.

Operating-system resource controls may still be appropriate when running Guild in shared or untrusted environments.

## Experimental architectures

Architecture support may be marked as:

- stable
- experimental
- parse-only
- unsupported

Experimental support may have received limited hardware and model validation.

An architecture being recognized by Guild does not necessarily mean that every checkpoint, quantization, auxiliary model, or hardware configuration has been validated.

## GPACK and GEXEC

Guild is developing native model packaging and execution formats.

### GPACK

GPACK is intended to be a portable model package containing model weights and related hardware-independent assets.

### GEXEC

GEXEC is intended to be a machine-optimized execution artifact containing model data and hardware-specific execution decisions.

Both formats should be treated as untrusted input when obtained from an untrusted source.

Their parsers must validate:

- lengths
- offsets
- tensor dimensions
- integer arithmetic
- file references
- section boundaries
- hardware compatibility metadata

before allocating memory or accessing packaged data.

Until these formats are declared stable, their security properties and validation rules may change.

## Output is not trusted code

Model output should always be treated as untrusted data.

Guild may generate:

- shell commands
- source code
- URLs
- scripts
- configuration files
- tool-call requests

Do not automatically execute model-generated content merely because it was produced by Guild.

Applications integrating Guild are responsible for applying appropriate authorization and validation before performing external actions.

## Denial of service

Inference is intentionally resource-intensive.

A request can consume substantial:

- CPU time
- GPU time
- RAM
- VRAM
- context memory
- disk bandwidth

Anyone able to submit inference requests may therefore be able to consume significant system resources.

Do not expose an unrestricted Guild server to users you do not trust.

## Secrets and logs

Do not place passwords, API keys, private keys, authentication tokens, or other secrets in:

- command-line arguments
- public bug reports
- benchmark logs
- telemetry dumps
- model manifests
- diagnostic output

When reporting issues, inspect logs and configuration files before uploading them.

## Privileges

Guild should normally run as an ordinary user.

Running Guild as `root` or another privileged account unnecessarily increases the impact of a vulnerability.

Do not grant Guild elevated privileges unless they are specifically required and understood.

## Third-party components

Guild may include, link against, or derive functionality from third-party open-source projects.

Security issues belonging primarily to a third-party dependency should normally also be reported to that project's maintainers.

Guild will update or patch affected dependencies when appropriate.

## Project origin

Guild was originally derived from:

https://github.com/Niko1221/Strata

The projects have since diverged substantially.

Security issues affecting Guild should be reported to the Guild project, not automatically to the Strata project.

Likewise, a vulnerability in Strata does not necessarily imply that Guild is affected, and a vulnerability in Guild does not necessarily imply that Strata is affected.

## Coordinated disclosure

For vulnerabilities with meaningful exploitation potential, please allow reasonable time for investigation and remediation before public disclosure.

Where appropriate, the project may publish a GitHub Security Advisory describing:

- affected versions
- impact
- mitigation
- fixed versions or commits
- reporter credit

Reporter credit will be included unless anonymity is requested.