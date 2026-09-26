import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/bindings/bindings.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;

void main() {
  test('loaded runtime and default C API match the compatibility contract', () {
    final manifest = jsonDecode(
        File('tool/native_runtime_versions.json').readAsStringSync());
    final expected = Platform.environment['ORT_TEST_VERSION'] ??
        manifest['platforms'][Platform.operatingSystem];
    expect(expected, isNotNull);
    expect(OrtEnv.version, expected);
    final api = onnxRuntimeBinding.OrtGetApiBase()
            .ref
            .GetApi
            .asFunction<Pointer<bg.OrtApi> Function(int)>()(
        manifest['c_api_version'] as int);
    expect(api, isNot(nullptr));
    expect(OrtEnv.instance.ortApiPtr, api);
  });

  test('common IR/opset fixture has fixed sync and async values', () async {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session =
        OrtSession.fromFile(File('test/fixtures/metadata.onnx'), options);
    final runOptions = OrtRunOptions();
    final input = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([-1.25, 3.5]), [1, 2]);
    final values = <OrtValue?>[];
    try {
      final sync = session.run(runOptions, {'input': input});
      values.addAll(sync);
      final async = await session.runAsync(runOptions, {'input': input})!;
      values.addAll(async);
      for (final outputs in [sync, async]) {
        expect(outputs, hasLength(1));
        expect(outputs.single, isA<OrtValueTensor>());
        expect(outputs.single!.value, [
          [-1.25, 3.5]
        ]);
      }
      expect(session.getMetadatas('author'), 'onnxruntime_flutter');
    } finally {
      final worker = session.isolateSession;
      session.release();
      await worker?.release();
      for (final value in values) {
        value?.release();
      }
      input.release();
      runOptions.release();
      options.release();
    }
  });
}
