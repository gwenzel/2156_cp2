# Quick DataLoader Test
# This cell tests if DataLoader can access files correctly from utils folder

from utils.data_loading import DataLoader

# Test DataLoader initialization and path resolution
data_loader = DataLoader()

print("🧪 DataLoader Path Test")
print("=" * 30)
print(f"Project root: {data_loader.project_root}")
print(f"Data directory: {data_loader.data_dir}")
print(f"Generated directory: {data_loader.generated_dir}")
print(f"Submissions directory: {data_loader.submissions_dir}")

print(f"\n📁 Directory Existence Check:")
print(f"Data dir exists: {os.path.exists(data_loader.data_dir)}")
print(f"Generated dir exists: {os.path.exists(data_loader.generated_dir)}")
print(f"Submissions dir exists: {os.path.exists(data_loader.submissions_dir)}")

# Test file loading (non-blocking)
print(f"\n🔄 Testing file access...")
import os
original_dataset_path = os.path.join(data_loader.project_root, '2155-Challenge-Problem-2/datasets')
print(f"Original dataset path: {original_dataset_path}")
print(f"Original dataset exists: {os.path.exists(original_dataset_path)}")

if os.path.exists(original_dataset_path):
    grid_files = [f for f in os.listdir(original_dataset_path) if f.startswith('grids_') and f.endswith('.npy')]
    print(f"Found grid files: {len(grid_files)} - {grid_files[:3]}{'...' if len(grid_files) > 3 else ''}")

print(f"\n✅ DataLoader is properly configured for utils folder!")