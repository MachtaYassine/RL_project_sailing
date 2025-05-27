# encode_weights_to_txt.py

import base64

# Path to your model weights file (e.g., PyTorch .pt file)
input_file = "src/agents/comb_student_agent_bc.pth"
output_txt = "encoded_weights.txt"

# Encode the file
with open(input_file, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

# Optionally wrap lines (e.g., 80 characters per line)
import textwrap
wrapped = textwrap.wrap(encoded, width=80)

# Save to .txt
with open(output_txt, "w") as f:
    for line in wrapped:
        f.write(line + "\n")

print(f"Encoded weights saved to {output_txt}")
