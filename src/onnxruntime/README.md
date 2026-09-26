# FFI header snapshot

These headers and `lib/src/bindings/onnxruntime_bindings_generated.dart` describe
the existing C API 14 contract. They are not compiled into the prebuilt native
libraries. Keep them pinned when upgrading native binaries: newer ONNX Runtime
versions expose older C API tables through `OrtGetApiBase()->GetApi(14)`.

Regenerating against current upstream headers would expand the generated Dart
surface and needs a separate API review. Runtime versions and binary provenance
are maintained in `tool/native_runtime_versions.json`.
