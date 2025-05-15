import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os # <-- Import os module
from datetime import datetime # <-- For timestamped filenames (optional)

# --- Configuration & Hyperparameters ---
CONFIG = {
    "data_path": "CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv",
    "use_full_dataset_if_small": True,
    "sample_size": 100000,
    "random_state": 42,
    "ip_cols": ['id.orig_h', 'id.resp_h'],
    "categorical_cols_initial": ['proto', 'conn_state', 'history'],
    "numerical_cols_initial": ['id.orig_p', 'id.resp_p', 'missed_bytes',
                               'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'],
    "label_col": 'label',
    "batch_size": 128,
    "latent_dim": 100,
    "label_embedding_dim": 10,
    "g_hidden_dim": 256,
    "d_hidden_dim": 256,
    "num_epochs": 150,
    "lr_g": 0.0002,
    "lr_d": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "num_generated_samples": 100 # Increased samples for a more substantial CSV
}

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 0. Load Dataset ---

df_full = pd.read_csv(CONFIG["data_path"])
print(f"Successfully loaded dataset from: {CONFIG['data_path']} with {len(df_full)} rows.")
if len(df_full) < 100:
    print("Warning: Loaded dataset is very small. Check the path and file content.")

# --- 1. Preprocessing the Data ---
print("\nStarting Preprocessing...")

if not CONFIG["use_full_dataset_if_small"] and len(df_full) > CONFIG["sample_size"]:
    df = df_full.sample(n=CONFIG["sample_size"], random_state=CONFIG["random_state"]).reset_index(drop=True)
    print(f"Sampled {len(df)} rows from the dataset.")
elif len(df_full) <= CONFIG["sample_size"] and CONFIG["use_full_dataset_if_small"]:
    df = df_full.copy().reset_index(drop=True)
    print(f"Using full dataset of {len(df)} rows.")
else:
    df = df_full.sample(n=CONFIG["sample_size"], random_state=CONFIG["random_state"]).reset_index(drop=True)
    print(f"Sampled {len(df)} rows from the dataset.")

print("Performing robust NaN filling and type conversion...")
for col in CONFIG["numerical_cols_initial"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    else:
        print(f"Warning: Initial numerical column {col} not found for NaN/type processing.")

for col in CONFIG["categorical_cols_initial"]:
    if col in df.columns:
        df[col] = df[col].fillna('MISSING').astype(str)
    else:
        print(f"Warning: Initial categorical column {col} not found for NaN/type processing.")

original_feature_columns = [col for col in df.columns if col != CONFIG["label_col"]]

processed_ip_cols = []
ip_octet_names_map = {}
for ip_col_name in CONFIG["ip_cols"]:
    if ip_col_name in df.columns:
        current_ip_octets = []
        octets = df[ip_col_name].astype(str).str.split('.', expand=True, n=3)
        for i in range(4):
            col_name = f"{ip_col_name}_octet_{i+1}"
            if i < octets.shape[1]:
                df[col_name] = pd.to_numeric(octets[i], errors='coerce').fillna(0).astype(int)
            else:
                df[col_name] = 0
            processed_ip_cols.append(col_name)
            current_ip_octets.append(col_name)
        ip_octet_names_map[ip_col_name] = current_ip_octets
        if ip_col_name in df.columns:
            df = df.drop(columns=[ip_col_name])
    else:
        print(f"Warning: IP column {ip_col_name} not found in DataFrame.")

label_encoder_target = LabelEncoder()
if CONFIG["label_col"] in df.columns:
    df[CONFIG["label_col"]] = label_encoder_target.fit_transform(df[CONFIG["label_col"]])
    print(f"Label mapping: {dict(zip(label_encoder_target.classes_, label_encoder_target.transform(label_encoder_target.classes_)))}")
    num_classes = len(label_encoder_target.classes_)
else:
    raise ValueError(f"Label column '{CONFIG['label_col']}' not found in the dataset.")

df_categorical_encoded_list = []
one_hot_encoders_map = {}
final_categorical_cols = []
for col in CONFIG["categorical_cols_initial"]:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False, dtype=bool)
        df_categorical_encoded_list.append(dummies)
        one_hot_encoders_map[col] = dummies.columns.tolist()
        final_categorical_cols.extend(dummies.columns.tolist())
        if col in df.columns:
             df = df.drop(columns=[col])
    else:
        print(f"Warning: Categorical column {col} not found for one-hot encoding.")

if df_categorical_encoded_list:
    df_categorical_encoded = pd.concat(df_categorical_encoded_list, axis=1)
else:
    df_categorical_encoded = pd.DataFrame(index=df.index)

numerical_features_to_scale = [col for col in CONFIG["numerical_cols_initial"] if col in df.columns] + \
                              [col for col in processed_ip_cols if col in df.columns]
numerical_features_to_scale = list(dict.fromkeys(numerical_features_to_scale))

if numerical_features_to_scale:
    for col in numerical_features_to_scale:
        if df[col].dtype == 'object':
            print(f"Warning: Column '{col}' intended for numerical scaling is still object type. Attempting conversion.")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df_numerical = df[numerical_features_to_scale].astype(float)
    scaler = MinMaxScaler()
    df_numerical_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=numerical_features_to_scale, index=df.index)
else:
    df_numerical_scaled = pd.DataFrame(index=df.index)
    scaler = None

features_df = pd.concat([df_numerical_scaled, df_categorical_encoded], axis=1)
labels_series = df[CONFIG["label_col"]]

if features_df.isnull().any().any():
    print("Warning: NaNs found in features_df after scaling/encoding. Filling with 0.")
    features_df = features_df.fillna(0)

print("Converting features_df to float32 for PyTorch tensor.")
try:
    features_df = features_df.astype(np.float32)
except Exception as e:
    print(f"Error during explicit astype(np.float32): {e}")
    print("Columns with issues might be:")
    for col_idx, col_name in enumerate(features_df.columns):
        try:
            features_df.iloc[:, col_idx].astype(np.float32)
        except Exception as col_e:
            print(f"  - Column '{col_name}' (index {col_idx}, dtype: {features_df.iloc[:, col_idx].dtype}): {col_e}")
    raise

data_features = torch.FloatTensor(features_df.values)
data_labels = torch.LongTensor(labels_series.values)

feature_dim = features_df.shape[1]
if feature_dim == 0:
    raise ValueError("No features to train on after preprocessing. Check column names and data.")
print(f"Total feature dimension after preprocessing: {feature_dim}")

dataset = TensorDataset(data_features, data_labels)
dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, label_embedding_dim, hidden_dim, num_classes):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_emb), -1)
        return self.model(gen_input)

class Discriminator(nn.Module):
    def __init__(self, feature_dim, label_embedding_dim, hidden_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(feature_dim + label_embedding_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, features, labels):
        label_emb = self.label_embedding(labels)
        disc_input = torch.cat((features, label_emb), -1)
        return self.model(disc_input)

generator = Generator(CONFIG["latent_dim"], feature_dim, CONFIG["label_embedding_dim"], CONFIG["g_hidden_dim"], num_classes).to(device)
discriminator = Discriminator(feature_dim, CONFIG["label_embedding_dim"], CONFIG["d_hidden_dim"], num_classes).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG["lr_g"], betas=(CONFIG["beta1"], CONFIG["beta2"]))
optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG["lr_d"], betas=(CONFIG["beta1"], CONFIG["beta2"]))
adversarial_loss = nn.BCELoss().to(device)

print("\nStarting GAN Training...")
for epoch in range(CONFIG["num_epochs"]):
    d_loss_last_batch = torch.tensor(float('nan'))
    g_loss_last_batch = torch.tensor(float('nan'))

    for i, (real_samples_batch, labels_batch) in enumerate(dataloader):
        real_samples_batch = real_samples_batch.to(device)
        labels_batch = labels_batch.to(device)
        current_batch_size = real_samples_batch.size(0)
        valid = torch.ones(current_batch_size, 1, device=device, dtype=torch.float)
        fake = torch.zeros(current_batch_size, 1, device=device, dtype=torch.float)

        optimizer_D.zero_grad()
        real_pred = discriminator(real_samples_batch, labels_batch)
        d_real_loss = adversarial_loss(real_pred, valid)
        noise = torch.randn(current_batch_size, CONFIG["latent_dim"], device=device)
        gen_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
        fake_samples = generator(noise, gen_labels)
        fake_pred = discriminator(fake_samples.detach(), gen_labels)
        d_fake_loss = adversarial_loss(fake_pred, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2
        if not torch.isnan(d_loss) and not torch.isinf(d_loss):
            d_loss.backward()
            optimizer_D.step()
        else:
            print(f"Warning: NaN or Inf in D_loss at epoch {epoch+1}, batch {i}. Skipping backward/step.")
        d_loss_last_batch = d_loss

        optimizer_G.zero_grad()
        noise_g = torch.randn(current_batch_size, CONFIG["latent_dim"], device=device)
        gen_labels_for_G = torch.randint(0, num_classes, (current_batch_size,), device=device)
        fake_samples_for_G = generator(noise_g, gen_labels_for_G)
        g_pred = discriminator(fake_samples_for_G, gen_labels_for_G)
        g_loss = adversarial_loss(g_pred, valid)
        if not torch.isnan(g_loss) and not torch.isinf(g_loss):
            g_loss.backward()
            optimizer_G.step()
        else:
            print(f"Warning: NaN or Inf in G_loss at epoch {epoch+1}, batch {i}. Skipping backward/step.")
        g_loss_last_batch = g_loss

    print(
        f"[Epoch {epoch+1}/{CONFIG['num_epochs']}] "
        f"[Last Batch D loss: {d_loss_last_batch.item() if not torch.isnan(d_loss_last_batch) else 'NaN' :.4f}] "
        f"[Last Batch G loss: {g_loss_last_batch.item() if not torch.isnan(g_loss_last_batch) else 'NaN' :.4f}]"
    )

print("Training Finished.")

print("\nGenerating Synthetic Traffic Samples...")
generator.eval()
generated_data_human_readable_list = []
with torch.no_grad():
    for i in range(CONFIG["num_generated_samples"]):
        noise_gen = torch.randn(1, CONFIG["latent_dim"], device=device)
        desired_label_encoded = torch.tensor([i % num_classes], device=device)
        synthetic_sample_scaled = generator(noise_gen, desired_label_encoded)
        synthetic_sample_scaled_np = synthetic_sample_scaled.cpu().numpy()
        synthetic_df_scaled = pd.DataFrame(synthetic_sample_scaled_np, columns=features_df.columns)

        synthetic_sample_unscaled_dict = {}
        if numerical_features_to_scale and scaler is not None:
            cols_to_inverse_transform = [col for col in numerical_features_to_scale if col in synthetic_df_scaled.columns]
            if cols_to_inverse_transform:
                synthetic_numerical_df_scaled_part = synthetic_df_scaled[cols_to_inverse_transform]
                synthetic_numerical_unscaled_np = scaler.inverse_transform(synthetic_numerical_df_scaled_part)
                synthetic_numerical_unscaled_df = pd.DataFrame(synthetic_numerical_unscaled_np, columns=cols_to_inverse_transform)
                for col in cols_to_inverse_transform:
                    value = synthetic_numerical_unscaled_df[col].iloc[0]
                    is_int_col = any(k_ for k_ in ['_p', '_bytes', '_pkts', '_octet_'] if k_ in col) or col == 'missed_bytes'
                    synthetic_sample_unscaled_dict[col] = int(round(value)) if is_int_col else value
        for original_cat_col, one_hot_cols in one_hot_encoders_map.items():
            if not one_hot_cols: continue
            relevant_one_hot_cols = [ohc for ohc in one_hot_cols if ohc in synthetic_df_scaled.columns]
            if not relevant_one_hot_cols:
                synthetic_sample_unscaled_dict[original_cat_col] = "UNKNOWN_CAT_NO_COLS"
                continue
            one_hot_vector_for_cat_series = synthetic_df_scaled[relevant_one_hot_cols].iloc[0]
            if not one_hot_vector_for_cat_series.empty:
                original_value_col_name = one_hot_vector_for_cat_series.idxmax()
                original_value = original_value_col_name.replace(f"{original_cat_col}_", "")
                synthetic_sample_unscaled_dict[original_cat_col] = original_value
            else:
                synthetic_sample_unscaled_dict[original_cat_col] = "UNKNOWN_CAT_EMPTY_SERIES"

        for original_ip_col, octet_names in ip_octet_names_map.items():
            octets = [str(synthetic_sample_unscaled_dict.get(o_name, 0)) for o_name in octet_names]
            synthetic_sample_unscaled_dict[original_ip_col] = ".".join(octets)
            for o_name in octet_names:
                if o_name in synthetic_sample_unscaled_dict:
                    del synthetic_sample_unscaled_dict[o_name]

        generated_label_str = label_encoder_target.inverse_transform(desired_label_encoded.cpu().numpy())[0]
        synthetic_sample_unscaled_dict[CONFIG["label_col"]] = generated_label_str
        generated_data_human_readable_list.append(synthetic_sample_unscaled_dict)

generated_df_human_readable = pd.DataFrame(generated_data_human_readable_list)

if not generated_df_human_readable.empty:
    final_columns_ordered = [CONFIG["label_col"]]
    for col in original_feature_columns:
        if col in generated_df_human_readable.columns:
            final_columns_ordered.append(col)
        elif col in CONFIG["ip_cols"] and col in generated_df_human_readable.columns:
             if col not in final_columns_ordered:
                final_columns_ordered.append(col)
    for col in generated_df_human_readable.columns:
        if col not in final_columns_ordered:
            final_columns_ordered.append(col)
    final_columns_ordered = [col for col in final_columns_ordered if col in generated_df_human_readable.columns]
    if final_columns_ordered:
        generated_df_human_readable = generated_df_human_readable[final_columns_ordered]

    print("\n--- Generated Human-Readable Traffic Samples (Preview) ---")
    print(generated_df_human_readable.head().to_string())

    output_dir = "generated_samples"
    os.makedirs(output_dir, exist_ok=True)

    # Using a timestamp for unique filenames each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"generated_traffic_{timestamp}.csv"
    # output_filename = "generated_traffic_samples.csv" # For a fixed filename

    output_filepath = os.path.join(output_dir, output_filename)

    try:
        generated_df_human_readable.to_csv(output_filepath, index=False)
        print(f"\nSuccessfully saved generated samples to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving generated samples to CSV: {e}")
else:
    print("\nNo human-readable samples were generated (DataFrame is empty), so no CSV file created.")


print("\n--- Expected Outcomes Summary ---")
print(f"1. Trained GAN: {'Yes' if 'generator' in locals() and 'discriminator' in locals() else 'No'}")
print(f"2. Preprocessing pipeline: Implemented")
print(f"3. Generated Samples: {len(generated_df_human_readable)} samples generated.")
print(f"   Human Readable: {'Yes, attempted inverse transformations' if not generated_df_human_readable.empty else 'No (or empty output)'}")
print(f"   Saved to CSV: {'Yes, at ' + output_filepath if not generated_df_human_readable.empty and 'output_filepath' in locals() else 'No'}")
print("\n--- Further Considerations ---")
print("- Evaluate generated data quality.")
print("- Experiment with GAN architectures and hyperparameters.")