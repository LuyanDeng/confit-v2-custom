import os
import json
import pandas as pd

def make_toy_recruiting_v2_dataset(base_dir="dataset/recruiting_data_v2_toy"):
    os.makedirs(base_dir, exist_ok=True)

    # ----------- 1️⃣ Resume 文件 -----------
    resumes = [
        {
            "user_id": "26d45a644f18ca57008d3b8c6bca34e4b26dd162297361ab244a2cc120efab6d",
            "text_resume": "Experienced data analyst skilled in Python, SQL, and data visualization. "
                           "Worked on machine learning pipelines and dashboard development."
        },
        {
            "user_id": "091579f7d6e4815f2dbaffdd38452032c2d867a1f85bd03ac2c8dce908d5ce9d",
            "text_resume": "Frontend engineer skilled in React, JavaScript, and Node.js. "
                           "Focused on responsive web design and UI performance optimization."
        },
        {
            "user_id": "b7f98484c19c37ef6424679f806f0d367c3c22797cb5d38cb95f85da746cb1f2",
            "text_resume": "Marketing manager with 5 years experience in social media strategy, "
                           "SEO optimization, and campaign analytics."
        }
    ]
    pd.DataFrame(resumes).to_csv(os.path.join(base_dir, "all_resume.csv"), index=False)

    # ----------- 2️⃣ JD 文件 -----------
    jobs = [
        {
            "jd_no": "3e281e56313473fb9a6e6962c8b7a8cf4c26e471b03eeb307e03678e008285eb",
            "text_job": "We are hiring a Data Scientist with experience in Python, machine learning, and SQL."
        },
        {
            "jd_no": "7e20a873023df0d8aecd6ba9b87f09b1be07c49beccbf902ac3e1de2d6f7278f",
            "text_job": "Looking for a Frontend Developer proficient in React and Node.js."
        },
        {
            "jd_no": "9f1b80ff1f5db93436f68ec4f1c2584bdbf34e93f78b8e889f1dc067e25fa333",
            "text_job": "Hiring a Marketing Specialist for social media content planning and SEO."
        }
    ]
    pd.DataFrame(jobs).to_csv(os.path.join(base_dir, "all_job.csv"), index=False)

    # ----------- 3️⃣ Train 数据 -----------
    train_pairs = [
        {"user_id": resumes[0]["user_id"], "jd_no": jobs[0]["jd_no"], "satisfied": 1},
        {"user_id": resumes[0]["user_id"], "jd_no": jobs[1]["jd_no"], "satisfied": 0},
        {"user_id": resumes[1]["user_id"], "jd_no": jobs[1]["jd_no"], "satisfied": 1},
        {"user_id": resumes[1]["user_id"], "jd_no": jobs[2]["jd_no"], "satisfied": 0},
        {"user_id": resumes[2]["user_id"], "jd_no": jobs[2]["jd_no"], "satisfied": 1},
        {"user_id": resumes[2]["user_id"], "jd_no": jobs[0]["jd_no"], "satisfied": 0}
    ]
    with open(os.path.join(base_dir, "train_labeled_data.jsonl"), "w") as f:
        for item in train_pairs:
            f.write(json.dumps(item) + "\n")

    # ----------- 4️⃣ Valid 数据 -----------
    valid_pairs = [
        {"user_id": resumes[0]["user_id"], "jd_no": jobs[0]["jd_no"], "satisfied": 1},
        {"user_id": resumes[1]["user_id"], "jd_no": jobs[2]["jd_no"], "satisfied": 0},
        {"user_id": resumes[2]["user_id"], "jd_no": jobs[1]["jd_no"], "satisfied": 0}
    ]
    with open(os.path.join(base_dir, "valid_classification_data.jsonl"), "w") as f:
        for item in valid_pairs:
            f.write(json.dumps(item) + "\n")

    # ----------- 5️⃣ Test 数据 -----------
    test_pairs = [
        {"user_id": resumes[0]["user_id"], "jd_no": jobs[0]["jd_no"], "satisfied": 1},
        {"user_id": resumes[1]["user_id"], "jd_no": jobs[2]["jd_no"], "satisfied": 0},
    ]
    with open(os.path.join(base_dir, "test_classification_data.jsonl"), "w") as f:
        for item in test_pairs:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Toy dataset (AliTianChi-style) created at: {base_dir}")


if __name__ == "__main__":
    make_toy_recruiting_v2_dataset()
