import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Create output directory
os.makedirs("output", exist_ok=True)


def calculate_angle(a, b, c):
    """Returns angle between three points (in degrees)."""
    if None in [a, b, c]:
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))


def euclidean_distance(a, b):
    """Distance between two points."""
    if None in [a, b]:
        return None
    return np.linalg.norm(np.array(a) - np.array(b))


def extract_landmark(landmarks, idx, frame_w, frame_h):
    """Helper to get landmark in pixel coords."""
    try:
        return int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)
    except:
        return None





def analyze_video(video_path="input_video.mp4"):
    cap = cv2.VideoCapture(video_path)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter("output/annotated_video.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_w, frame_h))

    # Collect metrics across frames
    summary_metrics = {
        "elbow_angle": [],
        "spine_angle": [],
        "head_knee_dist": [],
        "foot_angle": []
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

            if landmarks:
                # Extract joints
                l_shoulder = extract_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, frame_w, frame_h)
                l_elbow = extract_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value, frame_w, frame_h)
                l_wrist = extract_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, frame_w, frame_h)

                l_hip = extract_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, frame_w, frame_h)
                r_hip = extract_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, frame_w, frame_h)

                l_knee = extract_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value, frame_w, frame_h)
                l_ankle = extract_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, frame_w, frame_h)
                nose = extract_landmark(landmarks, mp_pose.PoseLandmark.NOSE.value, frame_w, frame_h)
                l_foot_index = extract_landmark(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value, frame_w, frame_h)

                # --- Metrics ---
                elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                if elbow_angle: summary_metrics["elbow_angle"].append(elbow_angle)

                # Spine lean (angle vs vertical surrogate)
                if l_hip and l_shoulder:
                    vertical_ref = (l_shoulder[0], l_shoulder[1] - 100)
                    spine_angle = calculate_angle(l_hip, l_shoulder, vertical_ref)
                    if spine_angle: summary_metrics["spine_angle"].append(spine_angle)

                # Head-over-knee distance
                if nose and l_knee:
                    dist = euclidean_distance(nose, l_knee)
                    if dist: summary_metrics["head_knee_dist"].append(dist)

                # Foot direction (ankle-foot index vs x-axis)
                if l_ankle and l_foot_index:
                    dx = l_foot_index[0] - l_ankle[0]
                    dy = l_foot_index[1] - l_ankle[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    summary_metrics["foot_angle"].append(angle)

                # --- Overlays ---
                if elbow_angle:
                    cv2.putText(frame, f"Elbow: {elbow_angle} deg", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if summary_metrics["spine_angle"]:
                    cv2.putText(frame, f"Spine Lean: {summary_metrics['spine_angle'][-1]} deg", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if summary_metrics["head_knee_dist"]:
                    cv2.putText(frame, f"Head-Knee Dist: {int(summary_metrics['head_knee_dist'][-1])}", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(frame)

    cap.release()
    out.release()

    # Final Evaluation
    evaluation = evaluate_shot(summary_metrics)

    with open("output/evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=4)

    print("✅ Processing complete. Outputs saved in /output/")


if __name__ == "__main__":
    analyze_video("input_video.mp4")
