/// Browser implementation of the existing ONNX Runtime API.
library;

import 'dart:async';
import 'dart:io' show File;
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;
import '../ort_provider.dart';
import '../ort_provider_config.dart';
import '../ort_web_options.dart';
import '../ort_sparse_data.dart';
import '../util/half_float.dart';
import '../providers/ort_flags.dart';
import '../util/list_shape_extension.dart';
import 'model_info.dart';
import '../ort_model_info.dart';
import '../ort_inference_exception.dart';

import '../ort_types.dart';
export '../ort_types.dart';

part 'env.dart';
part 'session.dart';
part 'value.dart';
part 'enums.dart';

@JS('ort')
external JSObject? get _ortGlobal;

JSObject get _runtime =>
    _ortGlobal ??
    (throw StateError(
      'ONNX Runtime Web is not loaded. Load ort.min.js before '
      'flutter_bootstrap.js; see the Web setup in README.md.',
    ));

@JS('ort.Tensor')
extension type _Tensor._(JSObject _) implements JSObject {
  external _Tensor(JSString type, JSAny data, JSArray<JSNumber> dims);
  external JSString get type;
  external JSAny get data;
  external JSArray<JSNumber> get dims;
  external void dispose();
}

@JS('ort.InferenceSession.create')
external JSPromise<_Session> _createSession(
  JSUint8Array model,
  JSObject options,
);

extension type _Session(JSObject _) implements JSObject {
  external JSPromise<JSObject> run(
    JSObject feeds,
    JSArray<JSString> fetches,
    JSObject options,
  );
  external JSPromise<JSAny?> release();
  external void endProfiling();
}

@JS('Object.keys')
external JSArray<JSString> _objectKeys(JSObject object);

@JS('String')
external JSString _jsString(JSAny value);

@JS('BigInt')
external JSBigInt _bigInt(JSString value);

@JS('BigInt64Array')
extension type _BigInt64Array._(JSObject _) implements JSObject {
  external _BigInt64Array(JSArray<JSBigInt> values);
}

@JS('BigUint64Array')
extension type _BigUint64Array._(JSObject _) implements JSObject {
  external _BigUint64Array(JSArray<JSBigInt> values);
}

Never _nativeOnly(String operation) =>
    throw UnsupportedError(
      '$operation requires the native runtime and is not available on Web.',
    );

class OrtStatus {
  OrtStatus._();
  static void checkOrtStatus(Object? ptr) {
    if (ptr != null) _nativeOnly('OrtStatus.checkOrtStatus');
  }
}

@JS('navigator')
external JSObject get _navigator;
