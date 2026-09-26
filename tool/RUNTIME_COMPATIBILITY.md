# Native runtime compatibility

The Dart API remains on C API version 14. Native runtime versions are deliberately
preserved: Android uses 1.23.2, and iOS, macOS, Linux and Windows use 1.15.1.
Changing those dependencies can change supported models, graph optimizations and
numerical results even when all Dart method signatures stay the same.

`native_runtime_versions.json` records those versions and hashes of bundled
libraries. Packaging tests detect unreviewed dependency or binary drift. The
native test suite asserts the actual loaded runtime version and C API table,
then exercises fixed outputs and the existing ownership/error regressions.
The common generated fixtures use IR 8 / opset 13.

To check another runtime without replacing a bundled library, put a compatible
library under the loader's existing filename in an isolated directory and set
its native search path plus `ORT_TEST_VERSION`. For example, on Linux:

```sh
ORT_TEST_VERSION=1.23.2 LD_LIBRARY_PATH=/path/to/isolated/runtime flutter test
```

The variable only changes test expectations; it is not read by the library.
Linux host runs against both 1.15.1 and 1.23.2 check the shared wrapper contract.
They do not substitute for Android/iOS device tests, provider-specific tests,
or comparisons of an application's own models on its target devices. The
version discrepancy remains; this compatibility guard does not promise identical
results or model support for arbitrary models across platforms.

A future version unification must update the manifest, build dependencies,
loader filenames and all binary artifacts together, and explicitly assess model
compatibility. This change intentionally does not upgrade or downgrade them.

One observed native difference is the invalid-input-name error text:
1.15.1 reports `Invalid Feed Input Name`, while 1.23.2 reports
`Invalid input name: missing` for the test input. Tests assert each pinned
version's original message; the wrapper continues to pass it through unchanged.
