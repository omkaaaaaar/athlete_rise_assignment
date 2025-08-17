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


def evaluate_shot(metrics):
    """Aggregate metrics into final evaluation scores + feedback."""
    evaluation = {}

    # Footwork: based on foot angle
    if metrics["foot_angle"]:
        avg_angle = np.mean(metrics["foot_angle"])
        footwork_score = np.clip(10 - abs(avg_angle), 1, 10)
        evaluation["Footwork"] = {
            "score": int(footwork_score),
            "feedback": "Front foot aligned well." if footwork_score > 7 else "Work on aligning your front foot."
        }

    # Head Position: head-knee distance
    if metrics["head_knee_dist"]:
        avg_dist = np.mean(metrics["head_knee_dist"])
        head_score = 10 if avg_dist < 30 else 8 if avg_dist < 60 else 6
        evaluation["Head Position"] = {
            "score": int(head_score),
            "feedback": "Head steady over knee." if head_score >= 8 else "Keep your head closer to front knee."
        }

    # Swing Control: variance in elbow angle
    if metrics["elbow_angle"]:
        var = np.var(metrics["elbow_angle"])
        swing_score = 9 if var < 50 else 7 if var < 100 else 5
        evaluation["Swing Control"] = {
            "score": int(swing_score),
            "feedback": "Consistent elbow position." if swing_score >= 7 else "Work on reducing elbow variability."
        }

    # Balance: spine lean
    if metrics["spine_angle"]:
        avg_spine = np.mean(metrics["spine_angle"])
        balance_score = 9 if 10 < avg_spine < 25 else 7 if 25 <= avg_spine <= 40 else 5
        evaluation["Balance"] = {
            "score": int(balance_score),
            "feedback": "Good body balance." if balance_score >= 7 else "Avoid leaning too much."
        }

    # Follow-through: last frames elbow
    if metrics["elbow_angle"]:
        final_elbow = np.mean(metrics["elbow_angle"][-10:])  # last 10 frames
        follow_score = 9 if 150 < final_elbow < 180 else 7 if 120 < final_elbow <= 150 else 5
        evaluation["Follow-through"] = {
            "score": int(follow_score),
            "feedback": "Smooth controlled finish." if follow_score >= 7 else "Work on your finishing swing."
        }

    return evaluation


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
