import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'strict_async_checks.dart';

void main() {
  test('strict failures retain diagnostics, isolate requests and recover', () async {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session = OrtSession.fromFile(File('test/fixtures/metadata.onnx'), options);
    options.release();
    await checkStrictAsync(session, web: false);
  });
}
