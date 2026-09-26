import 'dart:async';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;
import 'package:onnxruntime/src/util/execution_provider.dart';
import 'package:onnxruntime/src/util/native_memory.dart';

String? _name;
Map<String, String> _options = {};
int? _sessionAddress;
bool _fail = false;

bg.OrtStatusPtr _append(
    Pointer<bg.OrtSessionOptions> session,
    Pointer<Char> name,
    Pointer<Pointer<Char>> keys,
    Pointer<Pointer<Char>> values,
    int length) {
  _sessionAddress = 1234;
  _name = name.cast<Utf8>().toDartString();
  _options = {
    for (var i = 0; i < length; ++i)
      keys[i].cast<Utf8>().toDartString(): values[i].cast<Utf8>().toDartString()
  };
  return _fail
      ? using((arena) => OrtEnv.instance.ortApiPtr.ref.CreateStatus
              .asFunction<bg.OrtStatusPtr Function(int, Pointer<Char>)>()(
          bg.OrtErrorCode.ORT_FAIL,
          'provider failed'.toNativeUtf8(allocator: arena).cast()))
      : nullptr;
}

void main() {
  late Pointer<bg.OrtApi> api;
  late OrtSessionOptions session;
  late _Allocator memory;
  setUp(() {
    _name = null;
    _options = {};
    _sessionAddress = null;
    _fail = false;
    memory = _Allocator();
    session = OrtSessionOptions();
    api = calloc<bg.OrtApi>();
    api.ref.SessionOptionsAppendExecutionProvider = Pointer.fromFunction<
        bg.OrtStatusPtr Function(Pointer<bg.OrtSessionOptions>, Pointer<Char>,
            Pointer<Pointer<Char>>, Pointer<Pointer<Char>>, Size)>(_append);
  });
  tearDown(() {
    session.release();
    calloc.free(api);
    expect(memory.live, isEmpty);
  });

  bool append(OrtProvider provider, List<OrtProvider> available,
          [Map<String, String> options = const {}]) =>
      runZoned(
          () => appendExecutionProvider(
              api, Pointer.fromAddress(1234), provider, options,
              availableProviders: () => available),
          zoneValues: {nativeAllocatorZoneKey: memory});

  test('QNN registers with the native name and platform HTP backend', () {
    expect(append(OrtProvider.qnn, [OrtProvider.cpu, OrtProvider.qnn]), isTrue);
    expect(_sessionAddress, 1234);
    expect(_name, 'QNN');
    expect(_options,
        {'backend_path': Platform.isWindows ? 'QnnHtp.dll' : 'libQnnHtp.so'});
    expect(memory.allocations, greaterThan(0));
  });

  test('unavailable QNN retains false without attempting native registration',
      () {
    expect(append(OrtProvider.qnn, [OrtProvider.cpu]), isFalse);
    expect(_name, isNull);
    expect(memory.allocations, 0);
    if (!OrtEnv.instance.availableProviders().contains(OrtProvider.qnn)) {
      expect(session.appendQnnProvider(), isFalse);
    }
  });

  test('QNN native errors propagate and release temporary allocations', () {
    _fail = true;
    expect(() => append(OrtProvider.qnn, [OrtProvider.qnn]), throwsException);
    expect(_name, 'QNN');
    expect(memory.allocations, greaterThan(0));
  });

  test('XNNPACK keeps its options and native error behavior', () {
    expect(
        append(OrtProvider.xnnpack, [], {'intra_op_num_threads': '3'}), isTrue);
    expect(_name, 'XNNPACK');
    expect(_options, {'intra_op_num_threads': '3'});
    _fail = true;
    expect(() => append(OrtProvider.xnnpack, []), throwsException);
  });
}

class _Allocator implements Allocator {
  final live = <int>{};
  int allocations = 0;

  @override
  Pointer<T> allocate<T extends NativeType>(int byteCount, {int? alignment}) {
    final pointer = calloc.allocate<T>(byteCount, alignment: alignment);
    live.add(pointer.address);
    allocations++;
    return pointer;
  }

  @override
  void free(Pointer pointer) {
    expect(live.remove(pointer.address), isTrue);
    calloc.free(pointer);
  }
}
