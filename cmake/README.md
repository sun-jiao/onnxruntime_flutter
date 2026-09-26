# Desktop native libraries

Linux and Windows bundle ONNX Runtime 1.15.1 for x64 (the original platform
folder) and ARM64 (`arm64/`). CMake selects `FLUTTER_TARGET_PLATFORM` first,
then the Windows generator platform or target system processor for older builds.
Unknown targets fail at configuration time rather than packaging an x64 library.
Library basenames and the Dart API remain unchanged.

ARM64 binaries were extracted from the official
[ONNX Runtime v1.15.1 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1).
The upstream license and third-party notices accompany each binary.

| Artifact | SHA-256 |
| --- | --- |
| `onnxruntime-linux-aarch64-1.15.1.tgz` | `85272e75d8dd841138de4b774a9672ea93c1be108d96038c6c34a62d7f976aee` |
| `linux/arm64/libonnxruntime.so.1.15.1` | `5c6de97d2a2dbdd706c3f312d41e909b063ceb3284856174eb4f645d4cca9f88` |
| `onnxruntime-win-arm64-1.15.1.zip` | `7d9a837c02b1fbed8ee5698e7e18976fe73988df411e97693fd5cf5b09ee0552` |
| `windows/arm64/onnxruntime.dll` | `ab84e76a41fc404b98e55b1d394f83b9c932f49bf65395b56a1cdf5862713e7e` |

Run the offline packaging tests with:

```sh
python3 -m unittest discover -s test/packaging -v
```

They run both plugin CMake configurations using simulated target variables and
inspect the actual ELF/PE machine headers. This verifies artifact selection, not
ARM64 inference or a Windows toolchain build. Run the Dart native suite on each
target for execution coverage, setting the library path to the `arm64` folder
when testing the package directly on an ARM64 host.
