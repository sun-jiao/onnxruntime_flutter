import 'dart:ffi';

import 'package:ffi/ffi.dart';

// ORTCHAR_T is wchar_t (UTF-16) on Windows and char (UTF-8) elsewhere.
// The generated binding uses Pointer<Char>; casting preserves the pointer ABI
// while the allocation must still contain the platform's actual code units.
Pointer<Char> allocateOrtPath(String path,
    {required bool isWindows, required Allocator allocator}) {
  return isWindows
      ? path.toNativeUtf16(allocator: allocator).cast<Char>()
      : path.toNativeUtf8(allocator: allocator).cast<Char>();
}
