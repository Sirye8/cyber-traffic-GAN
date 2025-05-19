# Cyber Traffic GAN

A Generative Adversarial Network (GAN) for synthesizing realistic network traffic (benign and malicious) to enhance cybersecurity systems like Intrusion Detection Systems (IDS).

## 📝 Description

This project trains a GAN model using PyTorch to generate synthetic network traffic samples. The generated data mimics real-world traffic patterns, enabling researchers and practitioners to:
- Augment limited datasets for training IDS models.
- Stress-test IDS robustness against novel attack patterns.
- Study traffic dynamics without exposing sensitive real-world data.

**Key Features**:
- **Preprocessing Pipeline**: Handles IP address splitting, categorical encoding, and numerical normalization.
- **Conditional GAN**: Generates labeled traffic (benign/malicious) by incorporating class labels during training.
- **Human-Readable Output**: Converts generated numerical samples back into interpretable formats (e.g., reconstructed IP addresses).

## 📁 Dataset

The model is trained on the **CTU-IoT-Malware-Capture-1-1** dataset. Key preprocessing steps include:
- Splitting IP addresses into octets (e.g., `192.168.1.1` → `[192, 168, 1, 1]`).
- One-hot encoding categorical features (`proto`, `conn_state`, etc.).
- Min-max scaling for numerical features (`duration`, `bytes`, etc.).

Download the dataset and ensure it is placed in the project root as `CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv`.

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/cyber-traffic-GAN.git
   cd cyber-traffic-GAN
   ```

2. **Install dependencies**:
   ```bash
   pip install torch pandas numpy scikit-learn
   ```
    **Or if you have a gpu that supports cuda** (use your supported cuda version):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install pandas numpy scikit-learn
   ```
## 🚀 Usage

1. **Run the GAN training script**:
   ```bash
   python traffic_gan.py
   ```

2. **Expected Output**:
   - Preprocessed dataset statistics.
   - Training progress with generator/discriminator loss values.
   - Generated human-readable samples printed to the console.

3. **Customize Configuration**:
   Modify `CONFIG` in `main_gan_script.py` to adjust:
   - Batch size, latent dimension, and training epochs.
   - Columns to include for preprocessing.
   - Number of samples to generate.

## 📂 File Structure

```
├── main_gan_script.py       # Main training and generation script
├── README.md
├── CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv  # Dataset
└── generated_samples/       # Output directory for generated traffic
```

## 📊 Results

After training, the script will:
- Print synthetic traffic samples in a human-readable format, including:
   - Reconstructed IP addresses (e.g., `id.orig_h: 192.168.1.1`).
   - Categorical features (e.g., `proto: tcp`).
   - Labels (`benign`/`malicious`).