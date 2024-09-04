import time
import argparse
import os
os.environ["ALWAYSAI_DBG_DISABLE_MODEL_VALIDATION"] = "1"
# from yolo_v8 import yolo_v8_pre_process, yolo_v8_post_process
from yolo_v8 import yolo_v8_pre_process_trt, yolo_v8_post_process
os.environ['ALWAYSAI_CONNECT_TO_DEVICE_AGENT'] = '1'
import edgeiq

"""
input_feed = "QNO-8010R_2024_08_02.mp4"
"""


def main(input_feed):
    obj_detect = edgeiq.ObjectDetection(
        "stevegriset/smalley-yolov8v2l-onnx-single-batch_size",
        pre_process=yolo_v8_pre_process_trt,
        post_process=yolo_v8_post_process
    )
    obj_detect.load(engine=edgeiq.Engine.TENSOR_RT)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()
    frame_count = 0

    try:
        with edgeiq.FileVideoStream(input_feed) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                start_time = time.time()
                frame = video_stream.read()
                print("[INFO] video read time %s seconds" % (time.time() - start_time))

                start_time = time.time()
                results = obj_detect.detect_objects(
                    frame, confidence_level=0.58, overlap_threshold=0.6)
                print("[INFO] total inference time %s seconds" % (time.time() - start_time))

                start_time = time.time()
                frame = edgeiq.markup_image(frame, results.predictions, colors=[(0, 255, 0)],
                                            line_thickness=2, font_size=0.5)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append("Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                for prediction in results.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))
                print("[INFO] markup image time %s seconds" % (time.time() - start_time))

                frame_count = +1

                start_time = time.time()
                obj_detect.publish_analytics(
                    results, tag=f"{frame_count}")
                print("[INFO] publish analytics time %s seconds" % (time.time() - start_time))
                start_time = time.time()
                streamer.send_data(frame, text)
                print("[INFO] Streamer send data time %s seconds" % (time.time() - start_time))

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input arguments for Movement Analysis Application.')
    parser.add_argument("--input-feed", default="", type=str, required=True)
    args = parser.parse_args()
    main(input_feed=args.input_feed)
