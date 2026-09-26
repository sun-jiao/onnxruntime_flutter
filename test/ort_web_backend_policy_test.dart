import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/src/ort_web_options.dart';
void main() {
  test('GPU success, unavailable adapter, initialization failure and strict policy', () async {
    for (final adapter in [true, false]) {
      final selected = <OrtWebBackend>[];
      final result = await createWebBackend(const OrtWebOptions(backend: OrtWebBackend.webgpu),
        () async => OrtWebGpuAvailability(adapter, 'no adapter'), (backend) async { selected.add(backend); return backend.name; });
      expect(result.info.selected, adapter ? OrtWebBackend.webgpu : OrtWebBackend.wasm);
      expect(selected, [result.info.selected]);
      expect(result.info.fallbackReason == null, adapter);
    }
    final tried = <OrtWebBackend>[];
    final fallback = await createWebBackend(const OrtWebOptions(backend: OrtWebBackend.webgpu),
        () async => const OrtWebGpuAvailability(true), (backend) async {
      tried.add(backend);
      if (backend == OrtWebBackend.webgpu) throw StateError('missing GPU runtime');
      return 'wasm session';
    });
    expect(tried, [OrtWebBackend.webgpu, OrtWebBackend.wasm]);
    expect(fallback.info.fallbackReason, contains('missing GPU runtime'));
    await expectLater(createWebBackend(const OrtWebOptions(backend: OrtWebBackend.webgpu, fallbackToWasm: false),
      () async => const OrtWebGpuAvailability(false, 'absent'), (_) async => fail('must not create WASM')),
      throwsUnsupportedError);
  });
}
