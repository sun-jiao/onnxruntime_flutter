import 'dart:async';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'lifecycle_checks.dart';
void main() {
  test('scope awaits reverse disposal, continues after errors, closes once', () async {
    final order = <int>[];
    final gate = Completer<void>();
    final scope = OrtScope();
    scope.defer(() { order.add(1); });
    scope.defer(() { order.add(2); throw StateError('cleanup'); });
    scope.own(3, (n) async { await gate.future; order.add(n); });
    final closing = scope.closeAsync();
    final checked = expectLater(closing, throwsA(isA<OrtScopeException>()));
    expect(identical(closing, scope.closeAsync()), isTrue);
    expect(() => scope.defer(() {}), throwsStateError);
    expect(order, isEmpty);
    gate.complete();
    await checked;
    expect(order, [3, 2, 1]);
    expect(scope.disposalErrors, hasLength(1));
  });
  test('scope preserves body error and still cleans up', () async {
    var released = false;
    final error = StateError('body');
    await expectLater(usingOrtScope((scope) {
      scope.defer(() { released = true; throw StateError('cleanup'); });
      throw error;
    }), throwsA(same(error)));
    expect(released, isTrue);
  });
  test('close drains native runs and is awaitable after legacy release', () async {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session = OrtSession.fromFile(File('test/fixtures/metadata.onnx'), options);
    options.release();
    await checkClose(session);
  });
  test('usingSession closes after body failure', () async {
    final options = OrtSessionOptions();
    final session = OrtSession.fromFile(File('test/fixtures/metadata.onnx'), options);
    options.release();
    await expectLater(usingSession(session, (_) => throw StateError('body')), throwsStateError);
    expect(() => session.address, throwsStateError);
    await session.closeAsync();
  });
}
