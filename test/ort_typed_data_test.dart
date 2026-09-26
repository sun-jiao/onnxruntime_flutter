import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'typed_data_checks.dart';

void main() {
  test(
    'flat typed copies preserve dtype, full-width integers and ownership',
    () {
      checkTypedData(web: false);
    },
  );
  test('real inference output can be copied and used after release', () async {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session = OrtSession.fromFile(
      File('test/fixtures/metadata.onnx'),
      options,
    );
    options.release();
    await checkTypedInference(session);
  });
}
