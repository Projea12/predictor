// import 'package:flutter/material.dart';
// import 'package:http/http.dart' as http;
// import 'dart:convert';
// import 'dart:async';

// void main() {
//   runApp(MyApp());
// }

// class MyApp extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: 'Next Word Predictor',
//       theme: ThemeData(
//         primarySwatch: Colors.blue,
//         fontFamily: 'Inter',
//       ),
//       home: NextWordPredictorScreen(),
//       debugShowCheckedModeBanner: false,
//     );
//   }
// }

// class NextWordPredictorScreen extends StatefulWidget {
//   @override
//   _NextWordPredictorScreenState createState() => _NextWordPredictorScreenState();
// }

// class _NextWordPredictorScreenState extends State<NextWordPredictorScreen> {
//   final TextEditingController _controller = TextEditingController();
//   final FocusNode _focusNode = FocusNode();
//   List<WordPrediction> _predictions = [];
//   bool _isLoading = false;
//   Timer? _debounce;
//   OverlayEntry? _overlayEntry;
//   final GlobalKey _textFieldKey = GlobalKey();

//   // API configuration
//   static const String API_BASE_URL = 'http://localhost:8000';

//   @override
//   void initState() {
//     super.initState();
//     _controller.addListener(_onTextChanged);
//     _focusNode.addListener(_onFocusChanged);
//   }

//   @override
//   void dispose() {
//     _controller.removeListener(_onTextChanged);
//     _focusNode.removeListener(_onFocusChanged);
//     _controller.dispose();
//     _focusNode.dispose();
//     _debounce?.cancel();
//     _hideDropdown();
//     super.dispose();
//   }

//   void _onTextChanged() {
//     // Cancel previous timer
//     if (_debounce?.isActive ?? false) _debounce!.cancel();
    
//     // Start new timer
//     _debounce = Timer(const Duration(milliseconds: 300), () {
//       final text = _controller.text.trim();
//       if (text.isNotEmpty) {
//         _getPredictions(text);
//       } else {
//         _hideDropdown();
//       }
//     });
//   }

//   void _onFocusChanged() {
//     if (!_focusNode.hasFocus) {
//       // Delay hiding to allow for dropdown selection
//       Timer(Duration(milliseconds: 150), () {
//         if (!_focusNode.hasFocus) {
//           _hideDropdown();
//         }
//       });
//     }
//   }

//   Future<void> _getPredictions(String text) async {
//     setState(() {
//       _isLoading = true;
//     });

//     try {
//       final response = await http.post(
//         Uri.parse('$API_BASE_URL/predict'),
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: json.encode({
//           'text': text,
//           'top_k': 5,
//         }),
//       );

//       if (response.statusCode == 200) {
//         final data = json.decode(response.body);
//         final predictions = (data['predictions'] as List)
//             .map((pred) => WordPrediction.fromJson(pred))
//             .toList();
        
//         setState(() {
//           _predictions = predictions;
//           _isLoading = false;
//         });
        
//         _showDropdown();
//       } else {
//         print('API Error: ${response.statusCode}');
//         setState(() {
//           _predictions = [];
//           _isLoading = false;
//         });
//         _hideDropdown();
//       }
//     } catch (e) {
//       print('Network Error: $e');
//       setState(() {
//         _predictions = [];
//         _isLoading = false;
//       });
//       _hideDropdown();
//     }
//   }

//   void _showDropdown() {
//     if (_predictions.isEmpty) {
//       _hideDropdown();
//       return;
//     }

//     _hideDropdown(); // Remove existing overlay

//     final RenderBox renderBox = _textFieldKey.currentContext!.findRenderObject() as RenderBox;
//     final size = renderBox.size;
//     final position = renderBox.localToGlobal(Offset.zero);

//     _overlayEntry = OverlayEntry(
//       builder: (context) => Positioned(
//         left: position.dx,
//         top: position.dy + size.height + 4,
//         width: size.width,
//         child: Material(
//           elevation: 8,
//           borderRadius: BorderRadius.circular(8),
//           child: Container(
//             constraints: BoxConstraints(maxHeight: 200),
//             decoration: BoxDecoration(
//               color: Colors.white,
//               borderRadius: BorderRadius.circular(8),
//               border: Border.all(color: Colors.grey.shade300),
//             ),
//             child: ListView.builder(
//               padding: EdgeInsets.zero,
//               shrinkWrap: true,
//               itemCount: _predictions.length,
//               itemBuilder: (context, index) {
//                 final prediction = _predictions[index];
//                 return InkWell(
//                   onTap: () => _selectPrediction(prediction.word),
//                   child: Container(
//                     padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
//                     decoration: BoxDecoration(
//                       border: index < _predictions.length - 1
//                           ? Border(bottom: BorderSide(color: Colors.grey.shade200))
//                           : null,
//                     ),
//                     child: Row(
//                       children: [
//                         Expanded(
//                           child: Text(
//                             prediction.word,
//                             style: TextStyle(
//                               fontSize: 16,
//                               fontWeight: FontWeight.w500,
//                             ),
//                           ),
//                         ),
//                         Text(
//                           '${(prediction.probability * 100).toStringAsFixed(1)}%',
//                           style: TextStyle(
//                             fontSize: 12,
//                             color: Colors.grey.shade600,
//                           ),
//                         ),
//                       ],
//                     ),
//                   ),
//                 );
//               },
//             ),
//           ),
//         ),
//       ),
//     );

//     Overlay.of(context)?.insert(_overlayEntry!);
//   }

//   void _hideDropdown() {
//     _overlayEntry?.remove();
//     _overlayEntry = null;
//   }

//   void _selectPrediction(String word) {
//     final currentText = _controller.text;
//     final words = currentText.split(' ');
//     words.add(word);
    
//     _controller.text = words.join(' ') + ' ';
//     _controller.selection = TextSelection.fromPosition(
//       TextPosition(offset: _controller.text.length),
//     );
    
//     _hideDropdown();
//     _focusNode.requestFocus();
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: Colors.grey.shade50,
//       appBar: AppBar(
//         title: Text(
//           'Next Word Predictor',
//           style: TextStyle(
//             fontWeight: FontWeight.w600,
//             color: Colors.white,
//           ),
//         ),
//         backgroundColor: Colors.blue.shade700,
//         elevation: 0,
//       ),
//       body: Center(
//         child: Container(
//           width: double.infinity,
//           max_width: 800,
//           padding: EdgeInsets.all(24),
//           child: Column(
//             crossAxisAlignment: CrossAxisAlignment.start,
//             children: [
//               SizedBox(height: 32),
//               Text(
//                 'Type to get next word suggestions',
//                 style: TextStyle(
//                   fontSize: 28,
//                   fontWeight: FontWeight.w700,
//                   color: Colors.grey.shade800,
//                 ),
//               ),
//               SizedBox(height: 8),
//               Text(
//                 'Powered by LSTM neural network trained on text data',
//                 style: TextStyle(
//                   fontSize: 16,
//                   color: Colors.grey.shade600,
//                 ),
//               ),
//               SizedBox(height: 48),
//               Container(
//                 key: _textFieldKey,
//                 decoration: BoxDecoration(
//                   color: Colors.white,
//                   borderRadius: BorderRadius.circular(12),
//                   border: Border.all(color: Colors.grey.shade300),
//                   boxShadow: [
//                     BoxShadow(
//                       color: Colors.black.withOpacity(0.1),
//                       blurRadius: 8,
//                       offset: Offset(0, 2),
//                     ),
//                   ],
//                 ),
//                 child: TextField(
//                   controller: _controller,
//                   focusNode: _focusNode,
//                   maxLines: null,
//                   style: TextStyle(
//                     fontSize: 18,
//                     height: 1.5,
//                   ),
//                   decoration: InputDecoration(
//                     hintText: 'Start typing your text here...',
//                     hintStyle: TextStyle(
//                       color: Colors.grey.shade500,
//                       fontSize: 18,
//                     ),
//                     border: InputBorder.none,
//                     contentPadding: EdgeInsets.all(20),
//                     suffixIcon: _isLoading
//                         ? Padding(
//                             padding: EdgeInsets.all(16),
//                             child: SizedBox(
//                               width: 20,
//                               height: 20,
//                               child: CircularProgressIndicator(
//                                 strokeWidth: 2,
//                                 valueColor: AlwaysStoppedAnimation<Color>(
//                                   Colors.blue.shade400,
//                                 ),
//                               ),
//                             ),
//                           )
//                         : null,
//                   ),
//                 ),
//               ),
//               SizedBox(height: 24),
//               if (_predictions.isNotEmpty && !_isLoading)
//                 Container(
//                   padding: EdgeInsets.all(20),
//                   decoration: BoxDecoration(
//                     color: Colors.blue.shade50,
//                     borderRadius: BorderRadius.circular(12),
//                     border: Border.all(color: Colors.blue.shade200),
//                   ),
//                   child: Column(
//                     crossAxisAlignment: CrossAxisAlignment.start,
//                     children: [
//                       Text(
//                         'Predictions:',
//                         style: TextStyle(
//                           fontWeight: FontWeight.w600,
//                           color: Colors.blue.shade800,
//                           fontSize: 16,
//                         ),
//                       ),
//                       SizedBox(height: 12),
//                       Wrap(
//                         spacing: 8,
//                         runSpacing: 8,
//                         children: _predictions
//                             .map((prediction) => InkWell(
//                                   onTap: () => _selectPrediction(prediction.word),
//                                   child: Container(
//                                     padding: EdgeInsets.symmetric(
//                                       horizontal: 12,
//                                       vertical: 6,
//                                     ),
//                                     decoration: BoxDecoration(
//                                       color: Colors.white,
//                                       borderRadius: BorderRadius.circular(20),
//                                       border: Border.all(color: Colors.blue.shade300),
//                                     ),
//                                     child: Row(
//                                       mainAxisSize: MainAxisSize.min,
//                                       children: [
//                                         Text(
//                                           prediction.word,
//                                           style: TextStyle(
//                                             color: Colors.blue.shade800,
//                                             fontWeight: FontWeight.w500,
//                                           ),
//                                         ),
//                                         SizedBox(width: 4),
//                                         Text(
//                                           '${(prediction.probability * 100).toStringAsFixed(0)}%',
//                                           style: TextStyle(
//                                             color: Colors.blue.shade600,
//                                             fontSize: 12,
//                                           ),
//                                         ),
//                                       ],
//                                     ),
//                                   ),
//                                 ))
//                             .toList(),
//                       ),
//                     ],
//                   ),
//                 ),
//               SizedBox(height: 32),
//               Container(
//                 padding: EdgeInsets.all(16),
//                 decoration: BoxDecoration(
//                   color: Colors.amber.shade50,
//                   borderRadius: BorderRadius.circular(8),
//                   border: Border.all(color: Colors.amber.shade200),
//                 ),
//                 child: Row(
//                   children: [
//                     Icon(
//                       Icons.info_outline,
//                       color: Colors.amber.shade700,
//                     ),
//                     SizedBox(width: 12),
//                     Expanded(
//                       child: Text(
//                         'Make sure your FastAPI server is running on localhost:8000',
//                         style: TextStyle(
//                           color: Colors.amber.shade800,
//                           fontWeight: FontWeight.w500,
//                         ),
//                       ),
//                     ),
//                   ],
//                 ),
//               ),
//             ],
//           ),
//         ),
//       ),
//     );
//   }
// }

// class WordPrediction {
//   final String word;
//   final double probability;

//   WordPrediction({required this.word, required this.probability});

//   factory WordPrediction.fromJson(Map<String, dynamic> json) {
//     return WordPrediction(
//       word: json['word'],
//       probability: json['probability'].toDouble(),
//     );
//   }
// }
