"""
Interactive Model Inference Demo for Fine-Tuned LLaMA-3 HAR Agent
Student Investigator: Shudhanshu Yadav
Summer Research Internship Project - UCI HAR Benchmark Extension
"""
import os
import sys
import random
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows terminals
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

print("="*70)
print("   LLaMA-3 HUMAN ACTIVITY RECOGNITION (HAR) - INTERACTIVE DEMO")
print("="*70)

# Check GPU
print(f"CUDA Available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: GPU not available. Running model inference on CPU will be extremely slow.")
    print("Please run this script on a machine with a CUDA-enabled GPU (e.g., Lightning Studio).")
    input("Press Enter to continue anyway or Ctrl+C to exit...")

# ----------------------------------------------------
# 1. Configuration & Constants
# ----------------------------------------------------
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "./llama3_har_finetuned/final"

ACTIVITY_LABELS = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}

ACTIVITY_KNOWLEDGE = {
    'WALKING': {
        'description': 'Walking is a rhythmic bipedal locomotion at normal pace on flat ground.',
        'sensor_patterns': [
            'Accelerometer shows periodic oscillations with regular step patterns at ~1-2 Hz',
            'Gyroscope shows consistent small fluctuations corresponding to leg swing',
            'Body acceleration shows characteristic peaks of ~1g corresponding to each step',
            'Cyclic vertical acceleration pattern with frequency matching stride rate',
            'Regular peaks in Acc_Z axis corresponding to heel-strike impact',
            'Symmetric acceleration pattern between left and right steps',
            'Moderate standard deviation in body_acc (0.3-0.8 m/s²)',
            'Dominant frequency typically between 1.5-2.5 Hz matching normal walking cadence',
            'Low-to-moderate spectral entropy indicating semi-regular periodic motion',
            'Zero crossing rate moderate due to alternating positive/negative acceleration phases',
        ]
    },
    'WALKING_UPSTAIRS': {
        'description': 'Walking upstairs involves ascending steps with higher energy expenditure and body lifting.',
        'sensor_patterns': [
            'Higher vertical acceleration due to lifting body weight against gravity',
            'Greater gyroscope readings indicating ankle and knee flexion',
            'Asymmetric acceleration pattern compared to walking on flat surface',
            'Increased energy in high-frequency components due to step impact',
            'Elevated RMS acceleration values compared to level walking',
            'Forward body tilt detected in gyroscope pitch axis',
            'Higher mean body_acc_z due to upward propulsion force',
            'Longer stance phase in each step cycle compared to level walking',
            'Greater range in body_acc_y axis due to vertical displacement',
            'Higher kurtosis in acceleration signals due to sharp push-off peaks',
        ]
    },
    'WALKING_DOWNSTAIRS': {
        'description': 'Walking downstairs involves controlled descent with impact absorption at each step.',
        'sensor_patterns': [
            'Impact spikes in accelerometer when foot contacts lower step',
            'Higher negative vertical acceleration during descent phase',
            'Significant high-frequency impact acceleration transients',
            'Asymmetric acceleration signature due to controlled braking/eccentric movement',
            'High correlation between Acc_Y and Acc_Z axes during gravity-assisted descent',
            'Pronounced peak in body acceleration standard deviation',
            'Increased high-frequency power spectral density matching stepping rate',
            'Frequent sharp changes in angular velocity pitch/roll axis',
            'Short stance duration and rapid flight phase compared to walking',
            'Higher variation in roll angle as hips adjust to step impact',
        ]
    },
    'SITTING': {
        'description': 'Sitting is a static sedentary posture where the body is supported by a seat.',
        'sensor_patterns': [
            'Very low acceleration standard deviation (near zero, typically <0.02 m/s²)',
            'Stable vertical gravity vector indicating upright torso orientation',
            'Flatline signal pattern in body gyroscope (no rotation)',
            'Absence of periodic or oscillatory peaks in frequency domain',
            'Signal energy concentrated almost entirely at 0 Hz (DC component)',
            'Total acceleration values match gravity vector projection (magnitude ~1.0g)',
            'Negligible body displacement or body acceleration variance',
            'Extremely high spectral entropy (flat noise floor) or zero energy in signal',
            'No cyclic gait phases or peaks matching stride patterns',
            'Pitch angle of the device remains highly consistent and horizontal/tilted',
        ]
    },
    'STANDING': {
        'description': 'Standing is a static active posture where the body remains upright on feet.',
        'sensor_patterns': [
            'Near-zero body acceleration standard deviation (typically <0.03 m/s²)',
            'Gravity vector alignment reflects vertical torso position (Acc_Y matches phone vertical axis)',
            'Subtle micro-movements (postural sway) visible in body accelerometer',
            'Flat angular velocity reading (no active rotation, <0.05 rad/s)',
            'Frequency spectrum shows peak only at 0 Hz (static gravity component)',
            'Slightly higher high-frequency noise compared to laying or sitting due to muscle tremor',
            'Total acceleration magnitude remains constant at ~1.0g',
            'Stable roll/pitch angles corresponding to pocket or waist belt placement',
            'Absence of gait oscillations or periodic stride signatures',
            'Highly stable gravity component along the primary device axis',
        ]
    },
    'LAYING': {
        'description': 'Laying is a static resting posture where the body is in a horizontal position.',
        'sensor_patterns': [
            'Extremely low body acceleration standard deviation (flatline, <0.01 m/s²)',
            'Gravity vector projection changes completely (Acc_X or Acc_Z matches ~1.0g)',
            'Horizontal device orientation indicated by accelerometer axes shift',
            'No rotational components in gyroscope signals',
            'No periodic peaks in frequency spectrum',
            'Lowest spectral entropy values due to complete lack of movement',
            'Minimal signal power across all frequency bands',
            'Highly stable and constant tilt angle showing horizontal placement',
            'No micro-tremor or postural sway signals compared to standing',
            'Gravity acceleration magnitude remains steady at ~1.0g along horizontal axis',
        ]
    }
}

SYS_PROMPT = (
    "You are an AI assistant specialized in Human Activity Recognition (HAR). "
    "You analyze IMU sensor data (accelerometer and gyroscope) from smartphones "
    "and classify human activities, provide reasoning, answer questions, and give "
    "health recommendations based on activity patterns."
)

# ----------------------------------------------------
# 2. Helper Functions
# ----------------------------------------------------
def features_to_text(feat_vec, feat_names):
    """Convert 561-dim feature vector to structured text description"""
    key_substrings = [
        'tBodyAcc-mean()', 'tBodyAcc-std()',
        'tGravityAcc-mean()', 'tGravityAcc-std()',
        'tBodyGyro-mean()', 'tBodyGyro-std()',
        'fBodyAcc-mean()', 'fBodyAcc-entropy()'
    ]
    
    selected_features = {sub: {} for sub in key_substrings}
    for idx, name in enumerate(feat_names):
        for sub in key_substrings:
            if sub in name:
                axis = 'X' if '-X' in name else ('Y' if '-Y' in name else ('Z' if '-Z' in name else 'scalar'))
                selected_features[sub][axis] = feat_vec[idx]
                break
                
    lines = []
    # 1. Body Acceleration
    t_body_acc_mean = selected_features['tBodyAcc-mean()']
    t_body_acc_std = selected_features['tBodyAcc-std()']
    if t_body_acc_mean and t_body_acc_std:
        lines.append(f"[Body Acceleration (Time Domain)] "
                     f"Mean: X={t_body_acc_mean.get('X', 0):.4f}, Y={t_body_acc_mean.get('Y', 0):.4f}, Z={t_body_acc_mean.get('Z', 0):.4f} | "
                     f"Std: X={t_body_acc_std.get('X', 0):.4f}, Y={t_body_acc_std.get('Y', 0):.4f}, Z={t_body_acc_std.get('Z', 0):.4f}")
                     
    # 2. Gravity Acceleration
    t_grav_acc_mean = selected_features['tGravityAcc-mean()']
    t_grav_acc_std = selected_features['tGravityAcc-std()']
    if t_grav_acc_mean and t_grav_acc_std:
        lines.append(f"[Gravity Acceleration (Phone Orientation)] "
                     f"Mean: X={t_grav_acc_mean.get('X', 0):.4f}, Y={t_grav_acc_mean.get('Y', 0):.4f}, Z={t_grav_acc_mean.get('Z', 0):.4f} | "
                     f"Std: X={t_grav_acc_std.get('X', 0):.4f}, Y={t_grav_acc_std.get('Y', 0):.4f}, Z={t_grav_acc_std.get('Z', 0):.4f}")

    # 3. Body Gyroscope
    t_gyro_mean = selected_features['tBodyGyro-mean()']
    t_gyro_std = selected_features['tBodyGyro-std()']
    if t_gyro_mean and t_gyro_std:
        lines.append(f"[Body Gyroscope (Angular Velocity)] "
                     f"Mean: X={t_gyro_mean.get('X', 0):.4f}, Y={t_gyro_mean.get('Y', 0):.4f}, Z={t_gyro_mean.get('Z', 0):.4f} | "
                     f"Std: X={t_gyro_std.get('X', 0):.4f}, Y={t_gyro_std.get('Y', 0):.4f}, Z={t_gyro_std.get('Z', 0):.4f}")

    # 4. Frequency Domain & Spectral Entropy
    f_body_acc_mean = selected_features['fBodyAcc-mean()']
    f_entropy = selected_features['fBodyAcc-entropy()']
    if f_body_acc_mean and f_entropy:
        lines.append(f"[Frequency Domain Features] "
                     f"Mean Freq: X={f_body_acc_mean.get('X', 0):.4f}, Y={f_body_acc_mean.get('Y', 0):.4f}, Z={f_body_acc_mean.get('Z', 0):.4f} | "
                     f"Spectral Entropy: X={f_entropy.get('X', 0):.4f}, Y={f_entropy.get('Y', 0):.4f}, Z={f_entropy.get('Z', 0):.4f}")

    # 5. Dynamic vs Static Classifier
    if t_body_acc_std:
        avg_body_acc_std = (t_body_acc_std.get('X', 0) + t_body_acc_std.get('Y', 0) + t_body_acc_std.get('Z', 0)) / 3.0
        motion_type = "DYNAMIC (Moving)" if avg_body_acc_std > 0.05 else "STATIC (Stationary)"
        lines.append(f"[Motion Indicator] Type: {motion_type} (Avg Body Acceleration Std = {avg_body_acc_std:.4f})")
        
    return "\n".join(lines)

# ----------------------------------------------------
# 3. Load Data & Prepare RAG
# ----------------------------------------------------
print("\n[1/4] Loading dataset files...")
if not os.path.exists('./feature_names.txt'):
    print("Error: feature_names.txt not found. Cannot construct textual features.")
    sys.exit(1)

with open('./feature_names.txt', 'r') as f:
    feature_names = [line.strip().split(' ', 1)[-1] if ' ' in line else line.strip() for line in f.readlines()]

X_test_feat = np.load('X_test_feat.npy')
y_test = np.load('y_test.npy')
subjects_test = np.load('subjects_test.npy')

X_train_feat = np.load('X_train_feat.npy')
y_train = np.load('y_train.npy')
subjects_train = np.load('subjects_train.npy')

print(f"      Loaded Test Features: {X_test_feat.shape}")
print(f"      Loaded Train Features: {X_train_feat.shape}")

# Unseen split construction
unseen_subjects = [25, 26, 27, 28, 29, 30]
unseen_mask = np.isin(subjects_train, unseen_subjects)
X_unseen = X_train_feat[unseen_mask]
y_unseen = y_train[unseen_mask]
subjects_unseen = subjects_train[unseen_mask]

print(f"      Unseen subject samples: {len(X_unseen)}")

# RAG Index creation
print("\n[2/4] Initializing RAG Knowledge Base index...")
from sentence_transformers import SentenceTransformer
import faiss

retriever = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_texts = []
for activity, info in ACTIVITY_KNOWLEDGE.items():
    for pattern in info['sensor_patterns']:
        text = f"Activity: {activity}. {info['description']} Sensor Pattern: {pattern}"
        knowledge_texts.append(text)

knowledge_embeds = retriever.encode(knowledge_texts)
d = knowledge_embeds.shape[1]
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(knowledge_embeds.astype(np.float32))
print("      FAISS Vector Index initialized.")

def get_rag_context(feat_text):
    query_embed = retriever.encode([feat_text])
    distances, indices = faiss_index.search(query_embed.astype(np.float32), 3)
    return "\n".join([knowledge_texts[i] for i in indices[0]])

# ----------------------------------------------------
# 4. Load Fine-Tuned LLM Model
# ----------------------------------------------------
print("\n[3/4] Loading Fine-Tuned LLaMA-3 model adapters...")
print("      This will initialize LLaMA-3-8B-Instruct (4-bit QLoRA)...")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto',
        token=HF_TOKEN,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        base_model, 
        ADAPTER_PATH
    )
    model.eval()
    print("      Model and Adapters loaded successfully!")
except Exception as e:
    print(f"\nERROR LOADING MODEL: {e}")
    print("Make sure you have Hugging Face access, a correct token, and your fine-tuning weights are at: ./llama3_har_finetuned/final")
    sys.exit(1)

# ----------------------------------------------------
# 5. Interactive Demo Loop
# ----------------------------------------------------
print("\n[4/4] Starting Interactive Demo...")

while True:
    print("\n" + "="*70)
    print("SELECT DATA CATEGORY:")
    print("  1. SEEN Subject (From standard Test Set)")
    print("  2. UNSEEN Subject (From Held-Out Subjects 25-30)")
    print("  3. Exit Demo")
    print("="*70)
    
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == '3':
        print("\nExiting. Thank you!")
        break
        
    if choice not in ['1', '2']:
        print("Invalid choice, please select 1 or 2.")
        continue

    # Choose Seen vs Unseen
    if choice == '1':
        subset_name = "SEEN"
        X_active = X_test_feat
        y_active = y_test
        subj_active = subjects_test
    else:
        subset_name = "UNSEEN"
        X_active = X_unseen
        y_active = y_unseen
        subj_active = subjects_unseen
        if len(X_active) == 0:
            print("No unseen subject data found. Make sure subjects_train.npy and X_train_feat.npy are loaded.")
            continue

    # Select random index
    idx = random.randint(0, len(X_active) - 1)
    feat_vec = X_active[idx]
    true_label = ACTIVITY_LABELS[y_active[idx] + 1]
    subject_id = subj_active[idx]
    
    print(f"\n--> Selected {subset_name} Sample Index: {idx} (Subject ID: {subject_id})")
    print(f"--> True Activity Label: {true_label}")
    
    # 1. Convert features to text
    feat_text = features_to_text(feat_vec, feature_names)
    
    # 2. Retrieve context via RAG
    context = get_rag_context(feat_text)
    
    # Select task to demo
    print("\nSELECT TASK TO DEMO IN FRONT OF ADVISOR:")
    print("  1. Activity Classification (classification task)")
    print("  2. Step-by-Step Chain-of-Thought Reasoning (reasoning task)")
    print("  3. Dynamic vs Static Posture QnA (QA task)")
    print("  4. Personalized Health Recommendations (recommendation task)")
    print("  5. Run all 4 tasks sequentially")
    
    task_choice = input("Select task (1-5): ").strip()
    if task_choice not in ['1', '2', '3', '4', '5']:
        print("Invalid task selection.")
        continue

    tasks_to_run = []
    if task_choice == '1':
        tasks_to_run = [('classification', "Classify this activity (respond with activity name only):", 30)]
    elif task_choice == '2':
        tasks_to_run = [('reasoning', "Analyze these IMU features and explain the activity with reasoning:", 200)]
    elif task_choice == '3':
        tasks_to_run = [('qa', "Based on these sensor readings, is the person static or dynamic?", 100)]
    elif task_choice == '4':
        tasks_to_run = [('recommendation', f"The person is performing {true_label}. Give health recommendations.", 100)]
    elif task_choice == '5':
        tasks_to_run = [
            ('classification', "Classify this activity (respond with activity name only):", 30),
            ('reasoning', "Analyze these IMU features and explain the activity with reasoning:", 200),
            ('qa', "Based on these sensor readings, is the person static or dynamic?", 100),
            ('recommendation', f"The person is performing {true_label}. Give health recommendations.", 100)
        ]

    # Show prompt structure details
    print("\n" + "#"*70)
    print("PROMPT STRUCTURE (Explain this to your Advisor):")
    print("#"*70)
    print(f"System Prompt:\n{SYS_PROMPT}\n")
    print(f"Features converted to Text:\n{feat_text}\n")
    print(f"RAG Retrieved Context (domain patterns):\n{context}")
    print("#"*70 + "\n")

    for task_name, instruction, max_tok in tasks_to_run:
        print("\n" + "-"*50)
        print(f"TASK: {task_name.upper()}")
        print(f"User Query: {instruction}")
        print("-"*50)
        
        user_prompt = f"{instruction}\n\nSensor Features:\n{feat_text}\n\nContext:\n{context}"
        
        # Exact Meta-Llama-3 prompt template
        full_prompt = (
            "<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{SYS_PROMPT}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        
        # Tokenize and generate
        inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        print(f"\nLLaMA-3 Response:\n{response}")
        print("-"*50)

    input("\nPress Enter to return to main menu...")