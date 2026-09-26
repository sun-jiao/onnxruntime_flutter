import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:onnxruntime/onnxruntime.dart';

class ModelTypeTest {

  static Future<List<OrtValue?>> testFloat() {
    return testType(Float32List.fromList([1, 2, -3, -99.0, 99999]), [1, 5], 'assets/models/test_types_FLOAT.pb');
  }

  static Future<List<OrtValue?>> testInt64() {
    return testType(kIsWeb ? <int>[1, 2, -3, -9007199254740991, 9007199254740991] : Int64List.fromList([1, 2, -3, -9223372036854775808, int.parse('9223372036854775807')]), [1, 5], 'assets/models/test_types_INT64.pb');
  }

  static Future<List<OrtValue?>> testBool() {
    return testType([true, false, true, false, true], [1, 5], 'assets/models/test_types_BOOL.pb');
  }

  static Future<List<OrtValue?>> testString() {
    return testType(['a', 'b', 'c', 'd', 'e'], [1, 5], 'assets/models/test_types_STRING.pb');
  }

  static Future<List<OrtValue?>> testType(List data, List<int> shape, String assetModelName) async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    final rawAssetFile = await rootBundle.load(assetModelName);
    final bytes = rawAssetFile.buffer.asUint8List();
    final session = OrtSession.fromBuffer(bytes, sessionOptions);
    final runOptions = OrtRunOptions();
    final inputOrt = OrtValueTensor.createTensorWithDataList(data, shape);
    final inputs = {'input': inputOrt};
    final outputs = await session.runAsync(runOptions, inputs)!;
    inputOrt.release();
    runOptions.release();
    sessionOptions.release();
    session.release();
    OrtEnv.instance.release();
    return outputs;
  }
}