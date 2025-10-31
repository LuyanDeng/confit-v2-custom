import os
import json
import pandas as pd

def make_toy_recruiting_v2_dataset(base_dir="dataset/recruiting_data_v2_toy"):
    os.makedirs(base_dir, exist_ok=True)

    resumes = [
        {
            "user_id": "r001",
            "text_resume": "Experienced data analyst skilled in Python, SQL, and data visualization."
        },
        {
            "user_id": "r002",
            "text_resume": "Frontend engineer skilled in React, JavaScript, and Node.js."
        },
        {
            "user_id": "r003",
            "text_resume": "Marketing manager with 5 years experience in social media strategy."
        }
    ]
    pd.DataFrame(resumes).to_csv(os.path.join(base_dir, "all_resume.csv"), index=False)

    jobs = [
        {
            "job_id": "j001",
            "text_job": "We are hiring a Data Scientist with experience in Python and SQL."
        },
        {
            "job_id": "j002",
            "text_job": "Looking for a Frontend Developer proficient in React and Node.js."
        },
        {
            "job_id": "j003",
            "text_job": "Hiring a Marketing Specialist for social media and brand promotion."
        }
    ]
    pd.DataFrame(jobs).to_csv(os.path.join(base_dir, "all_job.csv"), index=False)

    # ✅ train
    train_pairs = [
        {"user_id": "r001", "job_id": "j001", "label": 1},
        {"user_id": "r001", "job_id": "j002", "label": 0},
        {"user_id": "r002", "job_id": "j002", "label": 1},
        {"user_id": "r002", "job_id": "j003", "label": 0},
        {"user_id": "r003", "job_id": "j003", "label": 1},
        {"user_id": "r003", "job_id": "j001", "label": 0}
    ]
    with open(os.path.join(base_dir, "train_labeled_data.jsonl"), "w") as f:
        for item in train_pairs:
            f.write(json.dumps(item) + "\n")

    # ✅ valid
    valid_pairs = [
        {"user_id": "r001", "job_id": "j001", "label": 1},
        {"user_id": "r002", "job_id": "j003", "label": 0},
        {"user_id": "r003", "job_id": "j002", "label": 0}
    ]
    with open(os.path.join(base_dir, "valid_classification_data.jsonl"), "w") as f:
        for item in valid_pairs:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Toy recruiting_data_v2 dataset created at: {base_dir}")

if __name__ == "__main__":
    make_toy_recruiting_v2_dataset()
