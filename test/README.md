# Native inference regression tests

Run from the repository root. The tests use the existing FLOAT model in
`example/assets/models/test_types_FLOAT.pb` and the bundled ONNX Runtime library.

Linux x64:

```sh
LD_LIBRARY_PATH="$PWD/linux${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" flutter test
```

macOS:

```sh
DYLD_LIBRARY_PATH="$PWD/macos${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" flutter test
```

Windows x64 (PowerShell):

```powershell
$env:PATH = "$PWD\windows;$env:PATH"
flutter test
```

The tests cover concurrent first calls, repeated concurrent batches after
initialization, unique output ownership, per-request values, and sequential calls
with both default and explicit output names.

Failure regressions cover invalid input names/shapes and output names, successful
requests after an error, mixed successful/failing concurrent requests, and abrupt
worker exit. Failed asynchronous calls retain the existing empty-list result.
Test-only five-second timeouts detect unresolved futures; no timeout is added to
the library API.

Lifecycle regressions cover release before initialization, release during
initialization, a gated busy worker, release after a failed request, and repeated
release of native wrappers. `OrtSession.release()` retains its `void` signature;
native destruction is deferred until accepted runs finish and the worker exits.
Inputs and run options must remain alive until their pending inference futures
complete, as with any asynchronous inference call.

File-session tests check UTF-16 (Windows) and UTF-8 (other platforms), including
terminators and surrogate pairs, on every host. They also load and run models in
paths containing spaces, Chinese characters, and emoji using the host's native
runtime, and verify missing files still throw. Run the suite on Windows to verify
the Windows DLL integration in addition to the platform-independent encoding
checks.
