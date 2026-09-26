require_relative 'download_runtime'
OnnxruntimeDownload.prepare(__dir__, File.expand_path('../tool/native_runtime_versions.json', __dir__))

#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint onnxruntime.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'onnxruntime'
  s.version          = '0.0.1'
  s.summary          = 'OnnxRuntime plugin for Flutter apps.'
  s.description      = <<-DESC
OnnxRuntime plugin for Flutter apps.
                       DESC
  s.homepage         = 'https://github.com/gtbluesky/onnxruntime_flutter'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  # s.source_files = 'Classes/**/*'
  s.dependency 'FlutterMacOS'
  s.vendored_libraries = '.onnxruntime/libonnxruntime.1.23.2.dylib'
  s.preserve_paths = '.onnxruntime/LICENSE', '.onnxruntime/ThirdPartyNotices.txt'
  s.platform = :osx, '13.4'
  s.static_framework = true
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'
end
