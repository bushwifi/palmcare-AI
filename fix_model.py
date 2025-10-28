import h5py
import json
import os

def inspect_h5_file(filepath):
    """Inspect the structure of an H5 model file."""
    print(f"\n{'='*60}")
    print(f"📂 Inspecting: {filepath}")
    print(f"{'='*60}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\n🔑 Top-level keys: {list(f.keys())}")
            
            # Check for model config
            if 'model_config' in f.attrs:
                print("\n📋 Model Config Found:")
                config = json.loads(f.attrs['model_config'])
                print(f"  Class: {config.get('class_name', 'Unknown')}")
                
                if 'config' in config:
                    model_config = config['config']
                    print(f"  Name: {model_config.get('name', 'Unknown')}")
                    
                    if 'layers' in model_config:
                        print(f"\n  📚 Layers ({len(model_config['layers'])} total):")
                        for i, layer in enumerate(model_config['layers'][:10]):  # First 10
                            layer_class = layer.get('class_name', 'Unknown')
                            layer_name = layer.get('config', {}).get('name', 'Unknown')
                            print(f"    {i+1}. {layer_class}: {layer_name}")
                        
                        if len(model_config['layers']) > 10:
                            print(f"    ... and {len(model_config['layers']) - 10} more layers")
            
            # Check for model weights
            if 'model_weights' in f:
                print(f"\n⚖️  Model Weights Found:")
                weights_group = f['model_weights']
                
                def count_weights(group, prefix=""):
                    count = 0
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            count += count_weights(item, prefix + key + "/")
                        else:
                            count += 1
                    return count
                
                total_weights = count_weights(weights_group)
                print(f"  Total weight tensors: {total_weights}")
                
                # Show first few weight groups
                print(f"\n  First few weight groups:")
                for i, key in enumerate(list(weights_group.keys())[:5]):
                    print(f"    - {key}")
            
            # Check training config
            if 'training_config' in f.attrs:
                print("\n🎓 Training Config Found")
                training_config = json.loads(f.attrs['training_config'])
                print(f"  Optimizer: {training_config.get('optimizer_config', {}).get('class_name', 'Unknown')}")
                print(f"  Loss: {training_config.get('loss', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Error reading H5 file: {e}")

def inspect_keras_file(filepath):
    """Inspect the structure of a Keras 3 file."""
    print(f"\n{'='*60}")
    print(f"📂 Inspecting: {filepath}")
    print(f"{'='*60}")
    
    try:
        import zipfile
        
        with zipfile.ZipFile(filepath, 'r') as z:
            print(f"\n📦 Archive contents:")
            files = z.namelist()
            
            for file in files:
                size = z.getinfo(file).file_size
                print(f"  - {file} ({size:,} bytes)")
            
            # Try to read config
            if 'config.json' in files:
                print(f"\n📋 Model Config:")
                with z.open('config.json') as config_file:
                    config = json.load(config_file)
                    print(f"  Class: {config.get('class_name', 'Unknown')}")
                    print(f"  Name: {config.get('config', {}).get('name', 'Unknown')}")
                    
                    if 'layers' in config.get('config', {}):
                        layers = config['config']['layers']
                        print(f"\n  📚 Layers ({len(layers)} total):")
                        for i, layer in enumerate(layers[:10]):
                            layer_class = layer.get('class_name', 'Unknown')
                            layer_name = layer.get('config', {}).get('name', 'Unknown')
                            print(f"    {i+1}. {layer_class}: {layer_name}")
                        
                        if len(layers) > 10:
                            print(f"    ... and {len(layers) - 10} more layers")
            
            # Check for metadata
            if 'metadata.json' in files:
                print(f"\n📊 Metadata:")
                with z.open('metadata.json') as meta_file:
                    metadata = json.load(meta_file)
                    print(f"  Keras version: {metadata.get('keras_version', 'Unknown')}")
                    print(f"  Date saved: {metadata.get('date_saved', 'Unknown')}")
                    
    except Exception as e:
        print(f"❌ Error reading Keras file: {e}")

def main():
    print("🔍 Palm Disease Model Inspector")
    print("="*60)
    
    # Check for model files
    keras_file = 'palm_disease_model.keras'
    h5_file = 'palm_disease_model.h5'
    
    found_files = []
    
    if os.path.exists(keras_file):
        found_files.append(keras_file)
    
    if os.path.exists(h5_file):
        found_files.append(h5_file)
    
    if not found_files:
        print("\n❌ No model files found!")
        print("Looking for:")
        print(f"  - {keras_file}")
        print(f"  - {h5_file}")
        return
    
    print(f"\n✅ Found {len(found_files)} model file(s)")
    
    # Inspect each file
    for filepath in found_files:
        if filepath.endswith('.keras'):
            inspect_keras_file(filepath)
        elif filepath.endswith('.h5'):
            inspect_h5_file(filepath)
        
        # Show file size
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\n📏 File size: {size_mb:.2f} MB")
    
    print("\n" + "="*60)
    print("✅ Inspection complete!")
    print("="*60)

if __name__ == "__main__":
    main()