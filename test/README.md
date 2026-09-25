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
