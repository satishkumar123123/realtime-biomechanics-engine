import cv2
import mediapipe as mp
import numpy as np
import time
from core.biomechanics import BiomechanicsEngine


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    prev_time = time.time()
    print("Baseline engine started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference timer
        t_infer_start = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

        metrics = {}
        if results.pose_world_landmarks and results.pose_landmarks:
            world_pts = np.zeros((33, 3), dtype=np.float32)
            visibilities = {}

            for idx, lm in enumerate(results.pose_world_landmarks.landmark):
                world_pts[idx] = [lm.x, lm.y, lm.z]
                visibilities[idx] = results.pose_landmarks.landmark[idx].visibility

            metrics = BiomechanicsEngine.compute_joint_metrics(world_pts, visibilities)

            # Draw visual landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # Diagnostics HUD
        cv2.rectangle(frame, (10, 10), (280, 190), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Infer: {infer_latency_ms:.1f}ms",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        y = 55
        for joint, val in metrics.items():
            if val is not None:
                text = f"{joint}: {val:.1f} deg"
                color = (255, 255, 255)
            else:
                text = f"{joint}: Low Conf / Occluded"
                color = (70, 70, 255)
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += 20

        cv2.imshow("Biomechanics Analysis Engine - Baseline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
