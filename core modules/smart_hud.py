
import cv2


# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


class SmartHUD:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_telemetry(self, frame, fps, inference_ms):
        height, width = frame.shape[:2]

        if inference_ms < 15:
            color = (0, 220, 0)
        elif inference_ms <= 33.3:
            color = (0, 190, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (0, 0), (width, 48), (25, 25, 25), -1)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 30),
            self.font, 0.65, (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            f"INFERENCE: {inference_ms:.1f} ms",
            (width - 260, 30),
            self.font, 0.6, color, 2
        )

    def draw_crop_brackets(self, frame, x_offset, y_offset, crop_size):
        x1, y1 = x_offset, y_offset
        x2, y2 = x1 + crop_size, y1 + crop_size

        length = 30
        color = (255, 200, 0)
        thickness = 2

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)

        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)

        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)

        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

    def draw_classification(self, frame, predictions):
        height, width = frame.shape[:2]

        x, y = 15, 65

        cv2.rectangle(
            frame, (x, y), (x + 280, y + 145),
            (30, 30, 30), -1
        )

        cv2.putText(
            frame, "CLASSIFICATION",
            (x + 10, y + 25),
            self.font, 0.55, (0, 220, 255), 1
        )

        if not predictions:
            cv2.putText(
                frame, "Scanning...",
                (x + 10, y + 65),
                self.font, 0.7, (255, 255, 255), 2
            )
            return

        for index, (label, score) in enumerate(predictions):
            row_y = y + 55 + index * 28

            cv2.putText(
                frame,
                f"{label}: {score * 100:.1f}%",
                (x + 10, row_y),
                self.font, 0.48, (255, 255, 255), 1
            )

            bar_x = x + 170
            bar_y = row_y - 12
            bar_width = 90

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + 9),
                (80, 80, 80), -1
            )

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + int(bar_width * score), bar_y + 9),
                (0, 220, 0), -1
            )

    def draw_detections(self, frame, detections, x_offset, y_offset, crop_size):
        scale = crop_size / 224.0

        for detection in detections:
            if not detection.categories:
                continue

            category = max(
                detection.categories,
                key=lambda item: item.score
            )

            box = detection.bounding_box

            x1 = int(x_offset + box.origin_x * scale)
            y1 = int(y_offset + box.origin_y * scale)
            x2 = int(x1 + box.width * scale)
            y2 = int(y1 + box.height * scale)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 220, 0), 2
            )

            label = f"{category.category_name} {category.score * 100:.0f}%"

            cv2.putText(
                frame, label,
                (x1, max(20, y1 - 8)),
                self.font, 0.5, (0, 220, 0), 2
            )

    def draw_hands(self, frame, hands):
        height, width = frame.shape[:2]

        for hand in hands:
            landmarks = hand["landmarks"]
            label = hand["label"]

            points = []

            for landmark in landmarks:
                px = int(landmark.x * width)
                py = int(landmark.y * height)
                points.append((px, py))

            # Draw skeleton
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 200, 0),
                    2
                )

            # Draw joints
            for point in points:
                cv2.circle(frame, point, 4, (0, 255, 0), -1)

            wrist_x, wrist_y = points[0]

            cv2.putText(
                frame,
                f"{label} Hand",
                (wrist_x, max(25, wrist_y - 15)),
                self.font, 0.65, (255, 200, 0), 2
            )

    def draw_mode(self, frame, task):
        cv2.putText(
            frame,
            f"MODE: {task.upper()}",
            (15, frame.shape[0] - 20),
            self.font, 0.6, (255, 255, 255), 2
        )
