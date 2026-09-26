import 'dart:async';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;
import 'package:onnxruntime/src/util/model_metadata.dart';
import 'package:onnxruntime/src/util/native_memory.dart';

const _missingMessage = "Operation 'toDartString' not allowed on a 'nullptr'.";

void main() {
  group('real model metadata', () {
    late OrtSessionOptions options;
    late OrtSession session;
    setUp(() {
      options = OrtSessionOptions()..setIntraOpNumThreads(1);
      session = OrtSession.fromBuffer(
          File('test/fixtures/metadata.onnx').readAsBytesSync(), options);
    });
    tearDown(() {
      session.release();
      options.release();
    });

    test('preserves normal, empty, Unicode and NUL-terminated values', () {
      final memory = _TrackedAllocator();
      for (var i = 0; i < 10; i++) {
        for (final entry in {
          'author': 'onnxruntime_flutter',
          'empty': '',
          '说明🧠': '中文元数据🧠',
          'nul': 'prefix',
        }.entries) {
          expect(
              memory.run(() => session.getMetadatas(entry.key)), entry.value);
          expect(memory.live, isEmpty);
        }
      }
      expect(memory.allocations, greaterThan(0));
    });

    test('missing keys keep UnsupportedError and clean up temporary memory',
        () {
      final memory = _TrackedAllocator();
      for (var i = 0; i < 3; i++) {
        expect(
            () => memory.run(() => session.getMetadatas('不存在')),
            throwsA(isA<UnsupportedError>()
                .having((error) => error.message, 'message', _missingMessage)));
        expect(memory.live, isEmpty);
      }
      expect(memory.allocations, greaterThan(0));
      expect(session.getMetadatas('author'), 'onnxruntime_flutter');
    });

    test('a released session still rejects metadata access', () {
      session.release();
      expect(() => session.getMetadatas('author'), throwsStateError);
    });
  });

  group('native metadata ownership', () {
    late Pointer<bg.OrtApi> api;
    late _TrackedAllocator memory;
    setUp(() {
      _native = _NativeState();
      memory = _TrackedAllocator();
      api = calloc<bg.OrtApi>();
      api.ref.SessionGetModelMetadata = Pointer.fromFunction<
          bg.OrtStatusPtr Function(Pointer<bg.OrtSession>,
              Pointer<Pointer<bg.OrtModelMetadata>>)>(_getMetadata);
      api.ref.ModelMetadataLookupCustomMetadataMap = Pointer.fromFunction<
          bg.OrtStatusPtr Function(
              Pointer<bg.OrtModelMetadata>,
              Pointer<bg.OrtAllocator>,
              Pointer<Char>,
              Pointer<Pointer<Char>>)>(_lookup);
      api.ref.ReleaseModelMetadata =
          Pointer.fromFunction<Void Function(Pointer<bg.OrtModelMetadata>)>(
              _releaseMetadata);
      api.ref.AllocatorFree = Pointer.fromFunction<
          bg.OrtStatusPtr Function(
              Pointer<bg.OrtAllocator>, Pointer<Void>)>(_freeString);
    });
    tearDown(() {
      // Also make failures in the test itself safe to rerun.
      for (final address in [..._native.metadata, ..._native.strings]) {
        calloc.free(Pointer<Void>.fromAddress(address));
      }
      calloc.free(api);
    });

    String read() => memory.run(() => readModelMetadata(
        api,
        Pointer<bg.OrtSession>.fromAddress(1),
        OrtAllocator.instance.ptr,
        '说明🧠'));

    void expectFreed({required int strings}) {
      expect(_native.metadata, isEmpty);
      expect(_native.strings, isEmpty);
      expect(_native.metadataReleases, 1);
      expect(_native.stringReleases, strings);
      expect(_native.invalidFree, isFalse);
      expect(memory.live, isEmpty);
    }

    test('success releases metadata and allocator-owned string exactly once',
        () {
      expect(read(), '中文🧠');
      expect(_native.key, '说明🧠');
      expect(_native.freeAllocator, OrtAllocator.instance.ptr.address);
      expectFreed(strings: 1);
    });

    test('a null lookup result frees metadata without freeing nullptr', () {
      _native.mode = _Mode.missing;
      expect(read, throwsA(isA<UnsupportedError>()));
      expectFreed(strings: 0);
    });

    test('UTF-8 decoding failure releases both native resources', () {
      _native.mode = _Mode.invalidUtf8;
      expect(read, throwsFormatException);
      expectFreed(strings: 1);
    });

    test('lookup status is checked before reading its result', () {
      _native.mode = _Mode.lookupError;
      expect(
          read,
          throwsA(isA<Exception>().having((error) => error.toString(),
              'message', 'code=2, message=lookup failed')));
      expectFreed(strings: 1);
    });

    test('metadata retrieval errors do not attempt lookup', () {
      _native.mode = _Mode.metadataError;
      expect(
          read,
          throwsA(isA<Exception>().having((error) => error.toString(),
              'message', 'code=2, message=metadata failed')));
      expect(_native.lookups, 0);
      expectFreed(strings: 0);
    });

    test('allocator cleanup errors still release metadata and temporary memory',
        () {
      _native.mode = _Mode.freeError;
      expect(
          read,
          throwsA(isA<Exception>().having((error) => error.toString(),
              'message', 'code=2, message=free failed')));
      expectFreed(strings: 1);
    });

    test('allocation failures unwind already acquired metadata', () {
      expect(read(), '中文🧠');
      final allocationCount = memory.allocations;
      for (var i = 1; i <= allocationCount; i++) {
        _native = _NativeState();
        memory = _TrackedAllocator(failAt: i);
        expect(read, throwsA(isA<_AllocationFailure>()));
        expect(_native.metadata, isEmpty);
        expect(_native.strings, isEmpty);
        expect(_native.invalidFree, isFalse);
        expect(memory.live, isEmpty);
      }
    });
  });
}

enum _Mode {
  success,
  missing,
  invalidUtf8,
  lookupError,
  metadataError,
  freeError
}

class _NativeState {
  var mode = _Mode.success;
  final metadata = <int>{};
  final strings = <int>{};
  var metadataReleases = 0;
  var stringReleases = 0;
  var lookups = 0;
  var invalidFree = false;
  String? key;
  int? freeAllocator;
}

late _NativeState _native;

// These callbacks model the C API's distinct ownership rules. Error statuses
// are real ORT statuses, so the library's existing error/release path is tested.
bg.OrtStatusPtr _getMetadata(Pointer<bg.OrtSession> session,
    Pointer<Pointer<bg.OrtModelMetadata>> output) {
  output.value = calloc<Uint8>().cast();
  _native.metadata.add(output.value.address);
  return _native.mode == _Mode.metadataError
      ? _error('metadata failed')
      : nullptr;
}

bg.OrtStatusPtr _lookup(
    Pointer<bg.OrtModelMetadata> metadata,
    Pointer<bg.OrtAllocator> allocator,
    Pointer<Char> key,
    Pointer<Pointer<Char>> output) {
  _native.lookups++;
  _native.key = key.cast<Utf8>().toDartString();
  if (_native.mode == _Mode.missing) {
    return nullptr;
  }
  if (_native.mode == _Mode.invalidUtf8) {
    final invalid = calloc<Uint8>(2)..value = 255;
    output.value = invalid.cast();
  } else {
    output.value = '中文🧠'.toNativeUtf8(allocator: calloc).cast();
  }
  _native.strings.add(output.value.address);
  return _native.mode == _Mode.lookupError ? _error('lookup failed') : nullptr;
}

void _releaseMetadata(Pointer<bg.OrtModelMetadata> metadata) {
  _native.metadataReleases++;
  if (_native.metadata.remove(metadata.address)) {
    calloc.free(metadata);
  } else {
    _native.invalidFree = true;
  }
}

bg.OrtStatusPtr _freeString(
    Pointer<bg.OrtAllocator> allocator, Pointer<Void> value) {
  _native.stringReleases++;
  _native.freeAllocator = allocator.address;
  if (_native.strings.remove(value.address)) {
    calloc.free(value);
  } else {
    _native.invalidFree = true;
  }
  return _native.mode == _Mode.freeError ? _error('free failed') : nullptr;
}

bg.OrtStatusPtr _error(String message) =>
    using((arena) => OrtEnv.instance.ortApiPtr.ref.CreateStatus
            .asFunction<bg.OrtStatusPtr Function(int, Pointer<Char>)>()(
        bg.OrtErrorCode.ORT_INVALID_ARGUMENT,
        message.toNativeUtf8(allocator: arena).cast()));

class _AllocationFailure implements Exception {}

class _TrackedAllocator implements Allocator {
  _TrackedAllocator({this.failAt});
  final int? failAt;
  var allocations = 0;
  final live = <int>{};

  T run<T>(T Function() action) =>
      runZoned(action, zoneValues: {nativeAllocatorZoneKey: this});

  @override
  Pointer<T> allocate<T extends NativeType>(int byteCount, {int? alignment}) {
    if (++allocations == failAt) {
      throw _AllocationFailure();
    }
    final pointer = calloc.allocate<T>(byteCount, alignment: alignment);
    live.add(pointer.address);
    return pointer;
  }

  @override
  void free(Pointer pointer) {
    if (!live.remove(pointer.address)) {
      throw StateError('Unexpected free');
    }
    calloc.free(pointer);
  }
}
