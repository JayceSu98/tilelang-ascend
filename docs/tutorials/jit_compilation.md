# JIT compilation on Ascend

TileLang offers two public entry points:

- `tilelang.compile(program, ...)` compiles an existing `T.prim_func`.
- `@tilelang.jit(...)` compiles the program returned by a Python kernel factory.

Both entry points accept `pass_configs` for TileLang compiler passes and
`compile_flags` for extra Bisheng command-line options.

## Runtime selection and cached libraries

The current `LibraryGenerator` links against `runtime` for NPU builds and
`runtime_camodel` for simulator builds, adding simulator library paths and RPATH
in the latter case. Kernel cache keys include `TL_RUN_MODE` to preserve that
existing link policy when restoring a library. This conservative partition can
compile the same device program twice; it does not establish that NPU and camodel
require different device code or incompatible shared libraries.

CANN's [msProf Kernel launch instructions](https://www.hiascend.com/document/detail/en/mindstudio/700/optools/Operatordevelopmenttools/atlasopdev_16_0082.html)
describe simulation without recompilation by providing a `libruntime.so` symlink
to `libruntime_camodel.so` and selecting it through the launch environment.
Sharing a TileLang library through that mechanism still needs validation for the
specific CANN version, target and host ABI. The current cache partition does not
implement a runtime-neutral build path.

Set `TL_RUN_MODE` and the runtime library search path before starting the process,
and keep the mode fixed for that process. Only `sim` selects the existing camodel
link path; other values retain the NPU path. Changing `LD_LIBRARY_PATH` after
Python or `torch_npu` has loaded libraries is not a supported way to switch
runtimes; see the [dynamic loader's search rules](https://man7.org/linux/man-pages/man3/dlopen.3.html).

Autotune results also include the mode in their keys: a reusable binary does not
make latency measurements or the best configuration interchangeable between
runtimes. The current `AutotuneResult` format stores both measurements and a copy
of the linked library, so this is result-bundle isolation, not separate binary
and measurement caches. Loaded libraries, tensors and streams remain local to
their process and runtime.

The Cython JIT adapter requires the `torch_npu` integration and uses torch NPU
tensors and streams. For A5 simulation on a host without a physical NPU, use the
camodel runner described in the
[programming guide](../TileLang-Ascend%20Programming%20Guide.md).
That runner initializes the simulator runtime, clears the cache, and builds
through `tilelang.lower` and `LibraryGenerator`, bypassing the JIT kernel cache.

Before replacing the current link policy with a shared-library build, validate
one unchanged TileLang-generated `.so` in two independently launched processes.
Record its hash, ELF dependencies and RPATH/RUNPATH, confirm the runtime libraries
actually loaded, and check the same computation in both environments without
another compiler invocation. Test the raw runtime ABI separately from Cython's
`torch_npu` integration. Mode-tagged test artifacts only validate cache routing;
they cannot establish whether a real CANN library is interchangeable.

## Kernel-scoped Bisheng flags

`compile_flags` accepts `list[str]`, `str`, or `None`:

```python
@tilelang.jit(
    out_idx=[1],
    target="ascendc",
    compile_flags=["-O3"],
)
def build_kernel(M, N):
    ...
```

Without the decorator:

```python
kernel = tilelang.compile(program, target="ascendc", compile_flags=["-O3"])
```

Flags are resolved per kernel and included in its cache key. TileLang derives
Bisheng defaults from `pass_configs`, applies supported legacy environment
overrides, then appends explicit `compile_flags`. For optimization level and
auto-sync, explicit flags override the defaults because Bisheng uses the last value.
`pass_configs` still controls which TileLang passes run.

The legacy variables are `TL_CCE_AUTO_SYNC`, `TL_CCE_OPT_LEVEL`, and
`TL_PTO_DEBUG`; the last affects only `target="pto"`. They are read-only:
compiling a kernel does not change the process environment or later kernels'
defaults. Prefer `compile_flags` for per-kernel options.

## Synchronization and debugging

For `target="ascendc"`, setting `TL_ASCEND_AUTO_SYNC_VS=True` in `pass_configs`
enables the TileLang VS synchronization pass and defaults Bisheng to
`-O3 --cce-auto-sync=off`. Passing
`compile_flags=["-O2", "--cce-auto-sync=on"]` changes Bisheng to O2 with auto-sync
enabled; the TileLang VS pass still runs.

Likewise, `TL_ASCEND_AUTO_SYNC=True` still enables TileLang synchronization
insertion when `compile_flags=["--cce-auto-sync=off"]` disables Bisheng's later
auto-sync stage. Disabling Bisheng auto-sync requires all dependencies to be
covered by the TileLang passes or manual synchronization.

Enable Ascend device printing for one kernel with:

```python
@tilelang.jit(
    target="ascendc",
    compile_flags=["-D_DEBUG", "--cce-enable-print"],
)
def debug_kernel(...):
    ...
```

Each entry is whitespace-split; exact duplicates of base-command flags are omitted.
TileLang does not validate target support: use one supported option per list entry. See
[`examples/compile_flags/compile_flags_example.py`](../../examples/compile_flags/compile_flags_example.py)
for an executable kernel-scoping example.

## Ahead-of-time compilation

For a deployable shared library, use the framework's `LibraryGenerator`
workflow instead of maintaining a copied Bisheng command. The canonical
end-to-end example is
[`examples/gemm_aot`](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples/gemm_aot).
