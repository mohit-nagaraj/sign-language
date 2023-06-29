// import DeviceDetector from "https://cdn.skypack.dev/device-detector-js@2.2.10";
// import * as mpHands from "https://cdn.skypack.dev/@mediapipe/hands";
// import * as drawingUtils from "https://cdn.skypack.dev/@mediapipe/drawing_utils";
// import * as controls from "https://cdn.skypack.dev/@mediapipe/control_utils";
// import * as controls3d from "https://cdn.skypack.dev/@mediapipe/controls3d";
// import * as mpHands from "@mediapipe/hands";
// import * as drawingUtils from "@mediapipe/drawing_utils";
// import * as controls from "@mediapipe/controls";
// import * as controls3d from "@mediapipe/controls3d";

// const videoElement = document.createElement('video');
// document.body.appendChild(videoElement);

// const canvasElement = document.createElement('canvas');
// document.body.appendChild(canvasElement);
// const canvasCtx = canvasElement.getContext('2d');

// const landmarkContainer = document.createElement('div');
// document.body.appendChild(landmarkContainer);

// const grid = new controls3d.LandmarkGrid(landmarkContainer, {
//   connectionColor: 0xCCCCCC,
//   definedColors: [{ name: 'Left', value: 0xffa500 }, { name: 'Right', value: 0x00ffff }],
//   range: 0.2,
//   fitToGrid: false,
//   labelSuffix: 'm',
//   landmarkSize: 2,
//   numCellsPerAxis: 4,
//   showHidden: false,
//   centered: false,
// });

// function onResults(results) {
//   canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//   canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

//   if (results.multiHandLandmarks && results.multiHandedness) {
//     for (let index = 0; index < results.multiHandLandmarks.length; index++) {
//       const classification = results.multiHandedness[index];
//       const isRightHand = classification.label === 'Right';
//       const landmarks = results.multiHandLandmarks[index];
//       drawingUtils.drawConnectors(
//         canvasCtx, landmarks, mpHands.HAND_CONNECTIONS,
//         { color: isRightHand ? '#00FF00' : '#FF0000' }
//       );
//       drawingUtils.drawLandmarks(canvasCtx, landmarks, {
//         color: isRightHand ? '#00FF00' : '#FF0000',
//         fillColor: isRightHand ? '#FF0000' : '#00FF00',
//         radius: (data) => {
//           return drawingUtils.lerp(data.from.z, -0.15, 0.1, 10, 1);
//         }
//       });
//     }
//   }

//   if (results.multiHandWorldLandmarks) {
//     const landmarks = results.multiHandWorldLandmarks.reduce((prev, current) => [...prev, ...current], []);
//     const colors = [];
//     let connections = [];
//     for (let loop = 0; loop < results.multiHandWorldLandmarks.length; ++loop) {
//       const offset = loop * mpHands.HAND_CONNECTIONS.length;
//       const offsetConnections = mpHands.HAND_CONNECTIONS.map((connection) => {
//         return [connection[0] + offset, connection[1] + offset];
//       });
//       connections = connections.concat(offsetConnections);
//       const classification = results.multiHandedness[loop];
//       colors.push({
//         list: offsetConnections.map((unused, i) => i + offset),
//         color: classification.label,
//       });
//     }
//     grid.updateLandmarks(landmarks, connections, colors);
//   } else {
//     grid.updateLandmarks([]);
//   }

//   requestAnimationFrame(() => {
//     hands.send({ image: videoElement });
//   });
// }

// navigator.mediaDevices.getUserMedia({ video: true })
//   .then((stream) => {
//     videoElement.srcObject = stream;
//     videoElement.addEventListener('loadedmetadata', () => {
//       videoElement.play();
//       canvasElement.width = videoElement.videoWidth;
//       canvasElement.height = videoElement.videoHeight;
//       const hands = new mpHands.Hands({
//         locateFile: (file) => {
//           return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.version}/${file}`;
//         }
//       });
//       hands.onResults(onResults);
//       hands.send({ image: videoElement });
//     });
//   })
//   .catch((error) => {
//     console.error('Error accessing the camera:', error);
//   });
// import * as tf from '@tensorflow/tfjs';



// const modelPath = 'file:///"C:\Users\bhuva\Downloads\my_trained_models\digits\keypoint_classifier(digits)fresh.tflite';

// var interpreter = new tf.Interpreter({
//   modelUrl: modelPath,
//   backend: 'webgl',
//   preferSharedMemory: true,
// });

// var inputDetails = interpreter.getInputDetails();
// var outputDetails = interpreter.getOutputDetails();



// var landmarkList = [0.0,0.0,-0.2564102564102564,0.0,-0.41025641025641024,-0.15384615384615385,-0.5641025641025641,-0.3333333333333333,-0.6410256410256411,-0.48717948717948717,-0.05128205128205128,-0.6666666666666666,-0.28205128205128205,-0.9743589743589743,-0.4358974358974359,-0.7948717948717948,-0.48717948717948717,-0.6153846153846154,-0.05128205128205128,-0.717948717948718,-0.28205128205128205,-1.0,-0.46153846153846156,-0.7948717948717948,-0.46153846153846156,-0.6153846153846154,-0.05128205128205128,-0.717948717948718,-0.28205128205128205,-0.9743589743589743,-0.4358974358974359,-0.7692307692307693,-0.4358974358974359,-0.6153846153846154,-0.07692307692307693,-0.6923076923076923,-0.2564102564102564,-0.8974358974358975,-0.41025641025641024,-0.7692307692307693,-0.4358974358974359,-0.5897435897435898];

// var inputDetailsTensorIndex = inputDetails[0].index;
// var inputTensor = tf.tensor([landmarkList], [1, landmarkList.length], 'float32');
// interpreter.setTensor(inputDetailsTensorIndex, inputTensor);

// interpreter.invoke();

// var outputDetailsTensorIndex = outputDetails[0].index;
// var outputTensor = interpreter.getTensor(outputDetailsTensorIndex);
// var result = outputTensor.dataSync();
// var resultIndex = result.indexOf(Math.max(...result));

// console.log(resultIndex);

// inputTensor.dispose();
// outputTensor.dispose();
// interpreter.delete();

const model = await handpose.load();
 
// Pass in a video stream to the model to obtain 
// a prediction from the MediaPipe graph.
const video = document.querySelector("#videoElement");
const hands = await model.estimateHands(video);
 
// Each hand object contains a `landmarks` property,
// which is an array of 21 3-D landmarks.
hands.forEach(hand => console.log(hand.landmarks));