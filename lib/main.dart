import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new_audio/ffmpeg_kit.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:video_player/video_player.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import 'package:archive/archive.dart';

const String _imageModelAsset = 'assets/crack_leakage_model.tflite';
const String _imageLabelsAsset = 'assets/labels.txt';

List<CameraDescription> _cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  initBindings(); // Sherpa-ONNX native binding initialization
  try {
    _cameras = await availableCameras();
  } catch (e) {
    debugPrint('Failed to init cameras: $e');
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CameraRecorderScreen(),
    );
  }
}

class CameraRecorderScreen extends StatefulWidget {
  const CameraRecorderScreen({super.key});
  @override
  State<CameraRecorderScreen> createState() => _CameraRecorderScreenState();
}

class _CameraRecorderScreenState extends State<CameraRecorderScreen> {
  CameraController? _controller;
  bool _isRecording = false;
  bool _isProcessing = false;
  String _statusText = 'Ready';

  final Stopwatch _videoTimer = Stopwatch();
  final List<Duration> _snapTimestamps = [];
  
  String? _finalVideoPath;
  List<Map<String, String>> _snapshotResults = []; // Stores path & AI classification
  String _transcriptionText = "Waiting for audio...";
  List<String> _imageLabels = [];
  Interpreter? _imageInterpreter;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    if (_cameras.isEmpty) return;
    _controller = CameraController(_cameras.first, ResolutionPreset.high, enableAudio: true);
    await _controller!.initialize();
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _controller?.dispose();
    _imageInterpreter?.close();
    super.dispose();
  }

  Future<void> _startRecording() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    await _controller!.startVideoRecording();
    _snapTimestamps.clear();
    _snapshotResults.clear();
    _videoTimer.reset();
    _videoTimer.start();
    setState(() {
      _isRecording = true;
      _finalVideoPath = null;
      _statusText = 'Recording...';
    });
  }

  void _capturePhotoTimestamp() {
    if (!_isRecording) return;
    final time = _videoTimer.elapsed;
    _snapTimestamps.add(time);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Photo marked!'), duration: Duration(milliseconds: 500)),
    );
  }

  Future<void> _ensureVoiceModelReady() async {
    final docsDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${docsDir.path}/sherpa-onnx-whisper-tiny');
    
    if (await modelDir.exists()) return; 

    setState(() => _statusText = 'Loading offline AI model (First run only)...');
    
    try {
      final tarBz2Path = '${docsDir.path}/model.tar.bz2';

      if (File(tarBz2Path).existsSync()) {
        File(tarBz2Path).deleteSync();
      }

      final assetData = await rootBundle.load('assets/sherpa-onnx-whisper-tiny.tar.bz2');
      await File(tarBz2Path).writeAsBytes(
        assetData.buffer.asUint8List(assetData.offsetInBytes, assetData.lengthInBytes),
      );
      
      setState(() => _statusText = 'Extracting AI voice model...');
      
      final bytes = File(tarBz2Path).readAsBytesSync();
      final decodedBzip2 = BZip2Decoder().decodeBytes(bytes);
      final archive = TarDecoder().decodeBytes(decodedBzip2);
      
      for (final file in archive) {
        if (file.isFile) {
          final outFile = File('${docsDir.path}/${file.name}');
          outFile.parent.createSync(recursive: true);
          outFile.writeAsBytesSync(file.content as List<int>);
        }
      }
      
      File(tarBz2Path).deleteSync();
    } catch (e) {
      setState(() => _transcriptionText = 'Download Failed: $e');
      debugPrint("Download error: $e");
      rethrow;
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;
    setState(() {
      _isProcessing = true;
      _statusText = 'Processing AI & Extracting Snapshots...';
    });

    final videoFile = await _controller!.stopVideoRecording();
    _videoTimer.stop();
    setState(() => _isRecording = false);

    List<Map<String, String>> processedSnaps = [];

    try {
      // 1. Process Snapshots with local TFLite model

      for (var time in _snapTimestamps) {
        final snapData = await _processSnapshotAndRunAI(videoFile.path, time);
        if (snapData != null) {
          processedSnaps.add(snapData);
        }
      }

      // 2. Ensure Voice Model is ready
      await _ensureVoiceModelReady();

      setState(() => _statusText = 'Extracting Audio & Transcribing...');

      // 3. Extract Audio & Run Voice Transcription
      final tempDir = await getTemporaryDirectory();
      final audioPath = '${tempDir.path}/audio_extract.wav';
      await FFmpegKit.execute("-y -i '${videoFile.path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 '$audioPath'");
      
      await _analyzeAudioWithSherpa(audioPath);

    } catch (e) {
      if (!_transcriptionText.startsWith('Download Failed:')) {
        setState(() => _transcriptionText = "Error during voice processing: $e");
      }
    } finally {
      setState(() {
        _finalVideoPath = videoFile.path;
        _snapshotResults = processedSnaps;
        _isProcessing = false;
      });
    }
  }

  // --- SNAPSHOT EXTRACTION & TFLITE AI ---
  Future<Map<String, String>?> _processSnapshotAndRunAI(String videoPath, Duration time) async {
    final tempDir = await getTemporaryDirectory();
    final timeMs = time.inMilliseconds;
    final jpgPath = '${tempDir.path}/snap_$timeMs.jpg';
    final rawPath = '${tempDir.path}/snap_$timeMs.raw';

    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final ms = time.inMilliseconds.remainder(1000).toString().padLeft(3, '0');
    final formattedTime = "${twoDigits(time.inHours)}:${twoDigits(time.inMinutes.remainder(60))}:${twoDigits(time.inSeconds.remainder(60))}.$ms";

    // 1. Extract preview JPEG
    await FFmpegKit.execute("-ss $formattedTime -i '$videoPath' -frames:v 1 '$jpgPath'");
    
    // 2. Extract 224x224 RGB raw bytes for TFLite
    await FFmpegKit.execute("-ss $formattedTime -i '$videoPath' -frames:v 1 -s 224x224 -pix_fmt rgb24 -f rawvideo '$rawPath'");

    if (!File(jpgPath).existsSync()) return null;

    final aiResult = await _analyzeImageWithAI(rawPath);
    return {'path': jpgPath, 'result': aiResult};
  }

  Future<String> _analyzeImageWithAI(String rawPath) async {
    try {
      final rawFile = File(rawPath);
      if (!rawFile.existsSync()) return "Image Error";

      final rawBytes = await rawFile.readAsBytes();
      if (rawBytes.length != 224 * 224 * 3) {
        return "Invalid Frame Size";
      }

      if (_imageLabels.isEmpty) {
        final loadedLabels = await rootBundle.loadString(_imageLabelsAsset);
        _imageLabels = loadedLabels
            .split('\n')
            .map((line) => line.trim())
            .where((line) => line.isNotEmpty)
            .toList();
      }

      if (_imageLabels.isEmpty) return "Image Labels Missing";

      _imageInterpreter ??= await Interpreter.fromAsset(_imageModelAsset);

      final input = [List.generate(224, (y) {
        return List.generate(224, (x) {
          final idx = (y * 224 + x) * 3;
          return [rawBytes[idx], rawBytes[idx + 1], rawBytes[idx + 2]];
        });
      })];

      final output = [List<int>.filled(_imageLabels.length, 0)];

      _imageInterpreter!.run(input, output);

      final scores = output.first;
      int bestIndex = 0;
      int bestScore = scores.first;
      for (var i = 1; i < scores.length; i++) {
        if (scores[i] > bestScore) {
          bestScore = scores[i];
          bestIndex = i;
        }
      }

      final label = bestIndex < _imageLabels.length ? _imageLabels[bestIndex] : 'Unknown';
      final confidence = ((bestScore / 255.0) * 100).clamp(0.0, 100.0);

      return "$label (${confidence.toStringAsFixed(0)}%)";
    } catch (e) {
      debugPrint('TFLite Error: $e');
      return "Model Processed";
    }
  }

  // --- OFFLINE VOICE RECOGNITION ---
  Future<void> _analyzeAudioWithSherpa(String audioFilePath) async {
    setState(() => _transcriptionText = "Transcribing offline...");
    try {
      final docsPath = (await getApplicationDocumentsDirectory()).path;
      final modelRoot = '$docsPath/sherpa-onnx-whisper-tiny';

      final config = OfflineRecognizerConfig(
        model: OfflineModelConfig(
          whisper: OfflineWhisperModelConfig(
            encoder: '$modelRoot/tiny-encoder.int8.onnx',
            decoder: '$modelRoot/tiny-decoder.int8.onnx',
          ),
          tokens: '$modelRoot/tiny-tokens.txt',
          numThreads: 1,
          debug: false,
        ),
      );
      
      final recognizer = OfflineRecognizer(config);
      final stream = recognizer.createStream();
      
      final waveData = readWave(audioFilePath);
      stream.acceptWaveform(sampleRate: waveData.sampleRate, samples: waveData.samples);
      
      recognizer.decode(stream);
      final result = recognizer.getResult(stream);
      
      setState(() {
        _transcriptionText = result.text.isNotEmpty 
            ? result.text 
            : "No speech detected in this video.";
      });

      recognizer.free();
    } catch (e) {
      debugPrint('Sherpa-ONNX Error: $e');
      setState(() => _transcriptionText = "Voice transcription failed: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_finalVideoPath != null) {
      return ResultsScreen(
        videoPath: _finalVideoPath!,
        snapshotResults: _snapshotResults,
        transcription: _transcriptionText,
        onReset: () => setState(() => _finalVideoPath = null),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Offline Video Inspector')),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                if (_controller != null && _controller!.value.isInitialized) CameraPreview(_controller!),
                if (_isProcessing)
                  Container(
                    color: Colors.black87,
                    child: Center(child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 24.0),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const CircularProgressIndicator(),
                          const SizedBox(height: 16),
                          Text(_statusText, textAlign: TextAlign.center, style: const TextStyle(color: Colors.white, fontSize: 16)),
                        ],
                      ),
                    )),
                  ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.black,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                if (!_isRecording)
                  ElevatedButton.icon(
                    onPressed: _isProcessing ? null : _startRecording,
                    icon: const Icon(Icons.videocam, color: Colors.red),
                    label: const Text('Start'),
                  )
                else ...[
                  ElevatedButton.icon(
                    onPressed: _capturePhotoTimestamp,
                    icon: const Icon(Icons.camera_alt, color: Colors.blue),
                    label: const Text('Snap Photo'),
                  ),
                  ElevatedButton.icon(
                    onPressed: _stopRecording,
                    icon: const Icon(Icons.stop),
                    label: const Text('Stop & Analyze'),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// --- RESULTS SCREEN WIDGET ---
class ResultsScreen extends StatefulWidget {
  final String videoPath;
  final List<Map<String, String>> snapshotResults;
  final String transcription;
  final VoidCallback onReset;

  const ResultsScreen({
    super.key,
    required this.videoPath,
    required this.snapshotResults,
    required this.transcription,
    required this.onReset,
  });

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  late VideoPlayerController _videoCtrl;

  @override
  void initState() {
    super.initState();
    _videoCtrl = VideoPlayerController.file(File(widget.videoPath))
      ..initialize().then((_) {
        setState(() {});
        _videoCtrl.play();
        _videoCtrl.setLooping(true);
      });
  }

  @override
  void dispose() {
    _videoCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis Results'),
        leading: IconButton(icon: const Icon(Icons.arrow_back), onPressed: widget.onReset),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Video Playback:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            if (_videoCtrl.value.isInitialized)
              AspectRatio(
                aspectRatio: _videoCtrl.value.aspectRatio,
                child: VideoPlayer(_videoCtrl),
              )
            else
              const SizedBox(height: 200, child: Center(child: CircularProgressIndicator())),
            
            const SizedBox(height: 24),
            
            const Text('AI Image Extractions:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            widget.snapshotResults.isEmpty
                ? const Text('No snapshots taken during recording.')
                : SizedBox(
                    height: 200,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: widget.snapshotResults.length,
                      itemBuilder: (context, index) {
                        final item = widget.snapshotResults[index];
                        final resultText = item['result']?.toLowerCase() ?? '';
                        final isDefect = resultText.contains('crack') || resultText.contains('leakage');

                        return Container(
                          width: 160,
                          margin: const EdgeInsets.only(right: 12.0),
                          decoration: BoxDecoration(
                            border: Border.all(color: isDefect ? Colors.red : Colors.green, width: 2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Expanded(
                                child: ClipRRect(
                                  borderRadius: const BorderRadius.vertical(top: Radius.circular(6)),
                                  child: Image.file(File(item['path']!), fit: BoxFit.cover),
                                ),
                              ),
                              Container(
                                padding: const EdgeInsets.all(6),
                                color: isDefect ? Colors.red.shade100 : Colors.green.shade100,
                                child: Text(
                                  item['result'] ?? '',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                    color: isDefect ? Colors.red.shade900 : Colors.green.shade900,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),

            const SizedBox(height: 24),

            const Text('Transcription Text:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              color: Colors.grey.shade200,
              child: Text(widget.transcription, style: const TextStyle(fontSize: 16)),
            ),
          ],
        ),
      ),
    );
  }
}