import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/ort_provider_config.dart' show registerOrtProviders;
void main() {
  test('provider ordering, flag composition, fallback and diagnostics', () {
    final seen = <OrtProviderConfig>[];
    final report = registerOrtProviders([
      OrtProviderConfig.qnn(performanceMode: QnnPerformanceMode.burst),
      OrtProviderConfig.coreml(flags: {CoreMLFlags.useCpuOnly, CoreMLFlags.enableOnSubgraph}),
    ], OrtProviderFallback.cpu, [OrtProvider.coreml.value], (c) { seen.add(c); return true; });
    expect(report.registered, [OrtProvider.coreml, OrtProvider.cpu]);
    expect(report.skipped.keys, [OrtProvider.qnn]);
    expect(seen.first.flags, 3);
    expect(OrtProviderConfig.qnn(backendPath: 'custom.so').options['backend_path'], 'custom.so');
    expect(() => OrtProviderConfig.xnnpack(intraOpNumThreads: -1), throwsArgumentError);
    expect(() => OrtProviderConfig.qnn(extraOptions: {'bad\u0000': 'x'}), throwsArgumentError);
    expect(() => report.registered.clear(), throwsUnsupportedError);
    expect(() => registerOrtProviders([OrtProviderConfig.cpu()], OrtProviderFallback.error,
        [OrtProvider.cpu.value], (_) => throw StateError('registration')), throwsStateError);
  });
  test('failed atomic configuration preserves usable native options', () async {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final unsupported = [OrtProviderConfig.qnn(), OrtProviderConfig.coreml(), OrtProviderConfig.nnapi()]
        .where((c) => !OrtEnv.instance.availableProviderNames().contains(c.provider.value)).first;
    expect(() => options.configureProviders([OrtProviderConfig.cpu(), unsupported]), throwsUnsupportedError);
    final report = options.configureProviders([unsupported], fallback: OrtProviderFallback.cpu);
    expect(report.registered, [OrtProvider.cpu]);
    expect(OrtEnv.instance.capabilities.providerNames, contains(OrtProvider.cpu.value));
    final session = OrtSession.fromFile(File('test/fixtures/metadata.onnx'), options);
    await session.closeAsync(); options.release();
    expect(() => options.configureProviders([OrtProviderConfig.cpu()]), throwsStateError);
  });
}
