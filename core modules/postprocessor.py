
from collections import deque, defaultdict


class PostProcessor:
    def __init__(self, threshold=0.5, smoothing_frames=5):
        self.threshold = threshold
        self.history = deque(maxlen=smoothing_frames)

    def process_classification(self, result):
        if not result.classifications:
            return []

        categories = result.classifications[0].categories

        # Store current prediction scores
        current_scores = {
            category.category_name: category.score
            for category in categories
        }

        self.history.append(current_scores)

        # Average scores across the last 5 runs
        averaged_scores = defaultdict(float)

        for frame_scores in self.history:
            for name, score in frame_scores.items():
                averaged_scores[name] += score

        for name in averaged_scores:
            averaged_scores[name] /= len(self.history)

        # Sort predictions from highest to lowest
        predictions = sorted(
            averaged_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # Apply confidence threshold
        predictions = [
            (name, score)
            for name, score in predictions
            if score >= self.threshold
        ]

        # Return Top-3 predictions
        return predictions[:3]

    def process_detections(self, result):
        filtered = []

        for detection in result.detections:
            if not detection.categories:
                continue

            category = max(
                detection.categories,
                key=lambda item: item.score
            )

            if category.score >= self.threshold:
                filtered.append(detection)

        return filtered

    def process_hands(self, result):
        filtered = []

        for index, landmarks in enumerate(result.hand_landmarks):
            if index >= len(result.handedness):
                continue

            if not result.handedness[index]:
                continue

            category = result.handedness[index][0]

            if category.score >= self.threshold:
                filtered.append({
                    "landmarks": landmarks,
                    "label": category.category_name,
                    "confidence": category.score
                })

        return filtered

    def set_threshold(self, threshold):
        self.threshold = max(0.0, min(1.0, threshold))

    def reset(self):
        self.history.clear()
