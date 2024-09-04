import sys
import os
import argparse
from tqdm import tqdm
from ultralytics import YOLO


def main(args):
    try:
        # Load the YOLOv8 model
        print("Loading model...")
        model = YOLO(args.weights)

        # Export the model to ONNX format
        print("Exporting to ONNX...")
        with tqdm(total=1, desc="Conversion", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            # Pass all parameters as keyword arguments
            model.export(
                format='onnx',
                imgsz=args.size,
                simplify=args.simplify,
                dynamic=args.dynamic,
                opset=args.opset,
                batch=args.batch
            )
            pbar.update(1)

        print("Conversion complete!")

        # Load the exported ONNX model and run inference (if input provided)
        if args.input:
            onnx_model = YOLO(args.weights.replace('.pt', '.onnx'))
            results = onnx_model(args.input, imgsz=args.size)

            # Process and display results (you'll need to customize this part)
            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:  # Explicitly check if no boxes were detected
                    print("No objects detected.")
                else:
                    for box in boxes:
                        # Assuming boxes contain [xyxy coordinates, confidence, class]
                        print(f"Coordinates: {box.xyxy}, Confidence: {box.conf}, Class: {box.cls}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 ONNX Conversion and Inference')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('-i', '--input', default='https://ultralytics.com/images/bus.jpg',
                        help='Input image file path or URL for inference (default: https://ultralytics.com/images/bus.jpg)')
    parser.add_argument('--opset', type=int, default=12, choices=range(7, 18), help='ONNX opset version (7-17)')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()

    # Validate arguments
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    if args.input and not (os.path.isfile(args.input) or args.input.startswith(('http://', 'https://'))):
        raise SystemExit('Invalid input. Provide a valid image file path or URL.')

    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
