import sys
sys.path.append("/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/caffemodel2pytorch")
import caffemodel2pytorch
import torch
import os

# Paths to your Caffe model and prototxt
caffe_model_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/caffemodel2pytorch/AOD_Net.caffemodel"
caffe_proto_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/caffemodel2pytorch/deploy.prototxt"
output_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/caffemodel2pytorch/aod_net.pth"

# Debug: Check if files exist
if not os.path.exists(caffe_model_path):
    print(f"Error: Caffe model not found at {caffe_model_path}")
    sys.exit(1)
if not os.path.exists(caffe_proto_path):
    print(f"Error: Prototxt not found at {caffe_proto_path}")
    sys.exit(1)

# Convert Caffe model to PyTorch
net = caffemodel2pytorch.Net(caffe_proto_path, caffe_model_path)
torch.save(net.state_dict(), output_path)
print(f"Converted model saved to {output_path}")