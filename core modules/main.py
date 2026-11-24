import cv2
import time

from stream_source import StreamSource
from input_contract import prepare_square_input
from vision_models import VisionModels
from postprocessor import PostProcessor
from smart_hud import SmartHUD


def main():
    stream = None
    models = None

    try:
        # Load models once before starting the camera
        models = VisionModels()
        stream = StreamSource()

        processor = PostProcessor(threshold=0.5)
        hud = SmartHUD()

        task = "classification"
        frame_skip = 1
        frame_count = 0

        frozen = False
        frozen_frame = None

        fps = 0.0
        fps_start = time.perf_counter()
        fps_frames = 0

        inference_ms = 0.0

        predictions = []
        detections = []
        hands = []

        # Create a large, resizable camera window
        window_name = "LIVE VISION AI - SMART HUD"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        print("\nLive Vision AI Started!")
        print("1 = Classification | 2 = Detection | 3 = Hand Tracking")
        print("F = Change frame skipping (1x / 2x / 3x)")
        print("+/- = Adjust confidence threshold")
        print("SPACE = Freeze / Resume")
        print("Q = Quit")

        while True:

            # Read camera frame unless frozen
            if not frozen:
                frame = stream.read()

                if frame is None:
                    print("Warning: Empty camera frame.")
                    continue

                frozen_frame = frame.copy()

            else:
                frame = frozen_frame.copy()

            frame_count += 1

            # Prepare model inputs
            rgb_crop, x_offset, y_offset, crop_size = prepare_square_input(frame)
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference according to frame skipping setting
            if not frozen and frame_count % frame_skip == 0:

                predictions = []
                detections = []
                hands = []

                if task == "classification":
                    result, inference_ms = models.classify(rgb_crop)
                    predictions = processor.process_classification(result)

                elif task == "detection":
                    result, inference_ms = models.detect_objects(rgb_crop)
                    detections = processor.process_detections(result)

                elif task == "hands":
                    result, inference_ms = models.track_hands(rgb_full)
                    hands = processor.process_hands(result)

            # Calculate FPS
            fps_frames += 1
            elapsed = time.perf_counter() - fps_start

            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.perf_counter()

            # Draw HUD on a copy, keeping the original frame clean
            display = frame.copy()

            if task == "classification":
                hud.draw_crop_brackets(
                    display, x_offset, y_offset, crop_size
                )
                hud.draw_classification(display, predictions)

            elif task == "detection":
                hud.draw_crop_brackets(
                    display, x_offset, y_offset, crop_size
                )
                hud.draw_detections(
                    display,
                    detections,
                    x_offset,
                    y_offset,
                    crop_size
                )

            elif task == "hands":
                hud.draw_hands(display, hands)

            # Draw telemetry and mode information
            hud.draw_telemetry(display, fps, inference_ms)
            hud.draw_mode(display, task)

            # Display frame skipping and confidence threshold
            cv2.putText(
                display,
                f"SKIP: {frame_skip}x | THRESHOLD: {processor.threshold:.0%}",
                (15, display.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # Frozen indicator
            if frozen:
                cv2.putText(
                    display,
                    "FROZEN",
                    (display.shape[1] - 130, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Show camera
            cv2.imshow(window_name, display)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("1"):
                task = "classification"
                processor.reset()
                predictions = []

            elif key == ord("2"):
                task = "detection"
                processor.reset()
                detections = []

            elif key == ord("3"):
                task = "hands"
                processor.reset()
                hands = []

            elif key == ord("f"):
                frame_skip = frame_skip % 3 + 1

            elif key in (ord("+"), ord("=")):
                processor.set_threshold(processor.threshold + 0.05)

            elif key in (ord("-"), ord("_")):
                processor.set_threshold(processor.threshold - 0.05)

            elif key == ord(" "):
                frozen = not frozen

    finally:
        # Release camera and models safely
        if stream is not None:
            stream.release()

        if models is not None:
            models.close()

        cv2.destroyAllWindows()
        print("Live Vision AI closed safely.")


if __name__ == "__main__":
    main()