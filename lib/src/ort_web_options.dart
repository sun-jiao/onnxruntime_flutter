enum OrtWebBackend { wasm, webgpu }

class OrtWebOptions {
  final OrtWebBackend backend;
  final bool fallbackToWasm;
  const OrtWebOptions({this.backend = OrtWebBackend.wasm, this.fallbackToWasm = true});
}

class OrtWebGpuAvailability {
  final bool available;
  final String? reason;
  const OrtWebGpuAvailability(this.available, [this.reason]);
}

/// Selected session backend, not per-node execution placement.
class OrtWebInitializationInfo {
  final OrtWebBackend requested;
  final OrtWebBackend selected;
  final String? fallbackReason;
  const OrtWebInitializationInfo(this.requested, this.selected, this.fallbackReason);
}

class WebBackendResult<T> {
  final T session;
  final OrtWebInitializationInfo info;
  WebBackendResult(this.session, this.info);
}

// Internal selection policy, independently testable without a GPU.
Future<WebBackendResult<T>> createWebBackend<T>(OrtWebOptions options,
    Future<OrtWebGpuAvailability> Function() probe,
    Future<T> Function(OrtWebBackend) create) async {
  String? reason;
  if (options.backend == OrtWebBackend.webgpu) {
    try {
      final availability = await probe();
      if (!availability.available) throw UnsupportedError(availability.reason ?? 'WebGPU unavailable.');
      final session = await create(OrtWebBackend.webgpu);
      return WebBackendResult(session, const OrtWebInitializationInfo(
          OrtWebBackend.webgpu, OrtWebBackend.webgpu, null));
    } catch (error) {
      if (!options.fallbackToWasm) rethrow;
      reason = error.toString();
    }
  }
  final session = await create(OrtWebBackend.wasm);
  return WebBackendResult(session, OrtWebInitializationInfo(options.backend, OrtWebBackend.wasm, reason));
}
